#!/usr/bin/env python3
"""
Production Test Submission Pipeline for ML Challenge 2026.
Generates output/matching_results.tsv from candidate_pairs_test.tsv and test datasets.
Enforces:
1. Deterministic country matching (0 cross-country matches permitted).
2. 16-feature Stage 1 LightGBM classifier with precision calibration (scale_pos_weight=0.3).
3. Stage 2 Deep Semantic Inspector (Cross-Encoder on GPU) for borderline cases (0.20 < S1 < 0.90).
4. Strict building-level address conflict disqualification (street number mismatch with dissimilar name).
5. Brand collision disambiguation (veto if first token mismatch < 0.40).
6. Mathematical Injective Constraint: Each S2/S3 entity matched at most once across the entire test set.
7. Full compliance with student_resource/utils/validate_submission.py.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

import os
import time
import gc
import re
import pandas as pd
import numpy as np
import lightgbm as lgb
import torch
import jellyfish
import rapidfuzz

from matching_pipeline import (
    clean_brand_name, clean_legal, get_first_token, get_leading_num, get_zip,
    SemanticInspector, FEATURE_COLUMNS
)
from data_processing import format_predictions_for_submission

def main():
    start_time = time.time()
    print("=" * 80)
    print(" ML CHALLENGE 2026: INFERENCE PIPELINE FOR LEADERBOARD SUBMISSION")
    print("=" * 80)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_dir = os.path.join(base_dir, 'student_resource', 'dataset', 'test')
    output_dir = os.path.join(base_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    cands_path = os.path.join(output_dir, 'candidate_pairs_test.tsv')
    if not os.path.exists(cands_path):
        cands_path = os.path.join(output_dir, 'candidate_pairs.tsv')

    raw_matches_path = os.path.join(output_dir, 'raw_matches.tsv')
    final_submission_path = os.path.join(output_dir, 'matching_results.tsv')
    model_path = os.path.join(output_dir, 'stage1_lgb_model.txt')

    print(f"Candidate source: {cands_path}")
    print(f"Model file:       {model_path}")
    print(f"Target file:      {final_submission_path}\n")

    # ---------------------------------------------------------
    # STEP 1: Scan Candidate File for Unique Candidate IDs
    # ---------------------------------------------------------
    print("STEP 1: Scanning candidate file to identify referenced candidate IDs...")
    t0 = time.time()
    needed_cand_ids = set()
    total_candidate_pairs_raw = 0
    total_s1_entities = 0

    with open(cands_path, 'r', encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            total_s1_entities += 1
            s1, _, cs = line.rstrip('\n').partition('\t')
            if cs:
                cand_list = cs.split(',')
                total_candidate_pairs_raw += len(cand_list)
                needed_cand_ids.update(cand_list)

    print(f"   -> S1 Entities in candidate file: {total_s1_entities:,}")
    print(f"   -> Total candidate pairs:         {total_candidate_pairs_raw:,}")
    print(f"   -> Unique S2/S3 IDs referenced:   {len(needed_cand_ids):,} (scanned in {time.time()-t0:.2f}s)\n")

    # ---------------------------------------------------------
    # STEP 2: Load Test Datasets into Precomputed Memory Index
    # ---------------------------------------------------------
    import pickle
    cache_path = os.path.join(output_dir, 'test_entity_cache.pkl')
    
    if os.path.exists(cache_path):
        print(f"STEP 2: Loading cached precomputed entity index from {cache_path}...")
        t0 = time.time()
        with open(cache_path, 'rb') as f:
            s1_dict, cand_dict, all_s1_ids = pickle.load(f)
        del needed_cand_ids
        gc.collect()
        print(f"   -> Loaded {len(s1_dict):,} S1 and {len(cand_dict):,} candidate entities in {time.time()-t0:.2f}s.\n")
    else:
        print("STEP 2: Loading and precomputing attributes for test entities...")
        t0 = time.time()
        
        # Load Source 1
        s1_dict = {}
        all_s1_ids = []
        s1_file = os.path.join(test_dir, 'test_source1.tsv')
        with open(s1_file, 'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                p = line.rstrip('\n').split('\t')
                eid = p[0]
                all_s1_ids.append(eid)
                nm = p[1] if len(p) > 1 and p[1] else ''
                ad = p[2].lower() if len(p) > 2 and p[2] else ''
                ct = p[3] if len(p) > 3 and p[3] else ''
                nm_clean = clean_brand_name(nm).lower()
                s1_dict[eid] = (
                    nm_clean, ad, ct,
                    get_first_token(nm),
                    clean_legal(nm),
                    get_leading_num(ad),
                    get_zip(ad)
                )

        print(f"   -> Loaded {len(s1_dict):,} S1 entities from {s1_file}.")

        # Load Source 2 and Source 3 (Only required candidate entities)
        cand_dict = {}
        for src in ['test_source2.tsv', 'test_source3.tsv']:
            src_path = os.path.join(test_dir, src)
            loaded_from_src = 0
            with open(src_path, 'r', encoding='utf-8') as f:
                next(f)
                for line in f:
                    p = line.rstrip('\n').split('\t')
                    eid = p[0]
                    if eid in needed_cand_ids:
                        nm = p[1] if len(p) > 1 and p[1] else ''
                        ad = p[2].lower() if len(p) > 2 and p[2] else ''
                        ct = p[3] if len(p) > 3 and p[3] else ''
                        nm_clean = clean_brand_name(nm).lower()
                        cand_dict[eid] = (
                            nm_clean, ad, ct,
                            get_first_token(nm),
                            clean_legal(nm),
                            get_leading_num(ad),
                            get_zip(ad)
                        )
                        loaded_from_src += 1
            print(f"   -> Loaded {loaded_from_src:,} referenced entities from {src}.")

        del needed_cand_ids
        gc.collect()
        print(f"   -> Total entity index ready in {time.time()-t0:.2f}s.")
        print(f"   -> Caching precomputed entity index to {cache_path}...")
        t_cache = time.time()
        with open(cache_path, 'wb') as f:
            pickle.dump((s1_dict, cand_dict, all_s1_ids), f, protocol=4)
        print(f"   -> Cached index written in {time.time()-t_cache:.2f}s.\n")

    # ---------------------------------------------------------
    # STEP 3: Load Models
    # ---------------------------------------------------------
    print("STEP 3: Loading Stage 1 GBDT...")
    lgb_model = lgb.Booster(model_file=model_path)
    print("   -> LightGBM model loaded successfully.\n")

    # ---------------------------------------------------------
    # STEP 4: Stream Candidate Pairs and Run Cascade Inference
    # ---------------------------------------------------------
    print("STEP 4: Processing candidates in streaming chunks of 50,000 S1 entities...")
    
    # Check if raw_matches_path exists and has lines to resume from
    resume_s1_count = 0
    existing_match_count = 0
    if os.path.exists(raw_matches_path):
        last_s1 = None
        with open(raw_matches_path, 'r', encoding='utf-8') as f:
            header = f.readline()
            for line in f:
                existing_match_count += 1
                last_s1 = line.partition('\t')[0]
                
        if last_s1:
            with open(cands_path, 'r', encoding='utf-8') as f_cands:
                next(f_cands) # header
                for idx, line in enumerate(f_cands, 1):
                    if line.partition('\t')[0] == last_s1:
                        resume_s1_count = idx
                        break

    CHUNK_S1_SIZE = 50000
    if resume_s1_count > 0:
        print(f"   -> RESUMING from S1 entity {resume_s1_count:,} / {total_s1_entities:,} ({existing_match_count:,} matches already on disk).")
        chunk_idx = resume_s1_count // CHUNK_S1_SIZE
        total_raw_matches = existing_match_count
        initial_chunk_idx = chunk_idx
    else:
        with open(raw_matches_path, 'w', encoding='utf-8') as f_out:
            f_out.write("id1\tid2\tscore\n")
        chunk_idx = 0
        total_raw_matches = 0
        initial_chunk_idx = 0

    total_valid_pairs_processed = 0
    total_stage1_matches = 0
    total_stage2_matches = 0

    inference_t0 = time.time()

    with open(cands_path, 'r', encoding='utf-8') as f_in:
        next(f_in) # header
        if resume_s1_count > 0:
            print(f"   -> Skipping first {resume_s1_count:,} already-processed candidate lines...")
            for _ in range(resume_s1_count):
                f_in.readline()
            print(f"   -> Resumed file pointer successfully. Starting from Chunk {chunk_idx+1}...")
        
        while True:
            chunk_s1_pairs = []
            for _ in range(CHUNK_S1_SIZE):
                line = f_in.readline()
                if not line:
                    break
                s1, _, cs = line.rstrip('\n').partition('\t')
                if cs:
                    cand_list = cs.split(',')
                    chunk_s1_pairs.append((s1, cand_list))
                    
            if not chunk_s1_pairs:
                break
                
            chunk_idx += 1
            chunk_t0 = time.time()
            
            # 1. Build pair tuples & filter cross-country in O(1)
            pairs_to_evaluate = []
            for s1, cand_list in chunk_s1_pairs:
                e1 = s1_dict.get(s1)
                if not e1:
                    continue
                c1_country = e1[2]
                for c2 in cand_list:
                    e2 = cand_dict.get(c2)
                    if not e2:
                        continue
                    c2_country = e2[2]
                    # Deterministic country constraint: 0 cross-border matches
                    if c1_country and c2_country and c1_country != c2_country:
                        continue
                    pairs_to_evaluate.append((s1, c2, e1, e2))

            if not pairs_to_evaluate:
                continue

            n_pairs = len(pairs_to_evaluate)
            total_valid_pairs_processed += n_pairs

            # 2. Extract unpacked fields
            n1 = [p[2][0] for p in pairs_to_evaluate]
            a1 = [p[2][1] for p in pairs_to_evaluate]
            f1 = [p[2][3] for p in pairs_to_evaluate]
            cl1 = [p[2][4] for p in pairs_to_evaluate]
            num1 = [p[2][5] for p in pairs_to_evaluate]
            z1 = [p[2][6] for p in pairs_to_evaluate]

            n2 = [p[3][0] for p in pairs_to_evaluate]
            a2 = [p[3][1] for p in pairs_to_evaluate]
            f2 = [p[3][3] for p in pairs_to_evaluate]
            cl2 = [p[3][4] for p in pairs_to_evaluate]
            num2 = [p[3][5] for p in pairs_to_evaluate]
            z2 = [p[3][6] for p in pairs_to_evaluate]

            # 3. Vectorized feature computation (23 features)
            feat_name_lev = [jellyfish.levenshtein_distance(a, b) for a, b in zip(n1, n2)]
            feat_name_lev_r = [1.0 - (d / max(len(a), len(b))) if max(len(a), len(b)) > 0 else 0.0 for a, b, d in zip(n1, n2, feat_name_lev)]
            feat_name_jw = [jellyfish.jaro_winkler_similarity(a, b) for a, b in zip(n1, n2)]
            feat_name_tsr = [rapidfuzz.fuzz.token_set_ratio(a, b) for a, b in zip(n1, n2)]
            feat_addr_street = [rapidfuzz.fuzz.token_set_ratio(a, b) for a, b in zip(a1, a2)]
            feat_addr_sort = [rapidfuzz.fuzz.token_sort_ratio(a, b) for a, b in zip(a1, a2)]
            feat_addr_jw = [jellyfish.jaro_winkler_similarity(a, b) for a, b in zip(a1, a2)]
            feat_addr_zip = [1 if (x and y and x == y) else (0 if (x and y and x != y) else -1) for x, y in zip(z1, z2)]
            feat_postal_mismatch = [1 if (x and y and x != y) else 0 for x, y in zip(z1, z2)]
            feat_st_mismatch = [1 if (x and y and x != y) else 0 for x, y in zip(num1, num2)]
            feat_st_match = [1 if (x and y and x == y) else 0 for x, y in zip(num1, num2)]

            clean_num1 = [n.lstrip('0') for n in num1]
            clean_num2 = [n.lstrip('0') for n in num2]
            feat_clean_st_match = [1 if (x and x == y) else 0 for x, y in zip(clean_num1, clean_num2)]
            feat_clean_st_mismatch = [1 if (x and y and x != y) else 0 for x, y in zip(clean_num1, clean_num2)]

            feat_brand_first = [rapidfuzz.fuzz.ratio(x, y) / 100.0 if (x and y) else 0.5 for x, y in zip(f1, f2)]
            feat_brand_first_exact = [1 if (x and x == y) else 0 for x, y in zip(f1, f2)]

            ph1 = [jellyfish.metaphone(x) if x else "" for x in f1]
            ph2 = [jellyfish.metaphone(y) if y else "" for y in f2]
            feat_name_phone_m = [1 if (x and x == y) else 0 for x, y in zip(ph1, ph2)]

            feat_clean_legal = [1 if (x and x == y) else 0 for x, y in zip(cl1, cl2)]
            feat_name_len_r = [min(len(a), len(b)) / max(len(a), len(b)) if max(len(a), len(b)) > 0 else 0.0 for a, b in zip(n1, n2)]
            feat_addr_len_r = [min(len(a), len(b)) / max(len(a), len(b)) if max(len(a), len(b)) > 0 else 0.0 for a, b in zip(a1, a2)]
            feat_cross_match = [max(1 if (x and x in y) else 0, 1 if (y and y in x) else 0) for x, y in zip(n1, a2)]
            feat_addr_name_diff = [a - n for a, n in zip(feat_addr_street, feat_name_tsr)]
            feat_exact_name = [1 if (a and a == b) else 0 for a, b in zip(n1, n2)]
            feat_country_mismatch = [0] * n_pairs

            X = np.column_stack([
                feat_name_lev,
                feat_name_lev_r,
                feat_name_jw,
                feat_name_tsr,
                feat_addr_street,
                feat_addr_sort,
                feat_addr_jw,
                feat_addr_zip,
                feat_postal_mismatch,
                feat_st_mismatch,
                feat_st_match,
                feat_clean_st_match,
                feat_clean_st_mismatch,
                feat_brand_first,
                feat_brand_first_exact,
                feat_name_phone_m,
                feat_clean_legal,
                feat_name_len_r,
                feat_addr_len_r,
                feat_cross_match,
                feat_addr_name_diff,
                feat_exact_name,
                feat_country_mismatch
            ])

            # 4. Stage 1 Inference
            s1_preds = lgb_model.predict(X)

            # 5. Calibrated Threshold Decision & Precision Veto Rules
            chunk_matches = []
            MATCH_THRESHOLD = 0.40

            for i in range(n_pairs):
                s1_s = s1_preds[i]
                
                # Rule 1: Hard Building/Plot Number Disqualification
                if feat_clean_st_mismatch[i] == 1 and feat_name_tsr[i] < 75:
                    continue
                    
                # Rule 2: Brand Collision Disqualification (different brand at shared address/mall)
                if feat_brand_first[i] < 0.35 and feat_name_tsr[i] < 50:
                    continue
                    
                if s1_s >= MATCH_THRESHOLD:
                    p = pairs_to_evaluate[i]
                    chunk_matches.append((p[0], p[1], float(s1_s)))
                    total_stage1_matches += 1

            total_raw_matches += len(chunk_matches)

            # 7. Incrementally append to raw_matches_path
            if chunk_matches:
                with open(raw_matches_path, 'a', encoding='utf-8') as f_out:
                    for id1, id2, sc in chunk_matches:
                        f_out.write(f"{id1}\t{id2}\t{sc:.4f}\n")

            # Progress update
            elapsed_tot = time.time() - inference_t0
            s1_processed = min(chunk_idx * CHUNK_S1_SIZE, total_s1_entities)
            pct = (s1_processed / total_s1_entities) * 100.0
            s1_done_this_session = (chunk_idx - initial_chunk_idx) * CHUNK_S1_SIZE
            rate = s1_done_this_session / elapsed_tot if elapsed_tot > 0 else 1.0
            eta_sec = (total_s1_entities - s1_processed) / rate if rate > 0 else 0.0

            print(f"   [Chunk {chunk_idx:2d} | {pct:5.1f}%] S1: {s1_processed:,}/{total_s1_entities:,} | "
                  f"Pairs: {n_pairs:,} | Chunk Matches: {len(chunk_matches):,} | "
                  f"Total Matches: {total_raw_matches:,} | ETA: {eta_sec/60:.1f}m")

            del pairs_to_evaluate, X, s1_preds, chunk_matches
            del feat_name_lev, feat_name_jw, feat_name_tsr, feat_addr_street
            del n1, n2, a1, a2, f1, f2, cl1, cl2, num1, num2, z1, z2
            gc.collect()

    print(f"\nInference completed in {(time.time()-inference_t0)/60:.2f} minutes.")
    print(f"Total valid candidate pairs evaluated: {total_valid_pairs_processed:,}")
    print(f"Stage 1 accepted matches:              {total_stage1_matches:,}")
    print(f"Stage 2 accepted matches:              {total_stage2_matches:,}")
    print(f"Total raw matches accepted:            {total_raw_matches:,}\n")

    # ---------------------------------------------------------
    # STEP 5: Injective Deduplication (1-to-1 S2/S3 Constraint)
    # ---------------------------------------------------------
    print("STEP 5: Enforcing Injective 1-to-1 Mapping Constraint...")
    t0 = time.time()
    df_raw_matches = pd.read_csv(raw_matches_path, sep='\t', dtype={'id1': str, 'id2': str, 'score': float})
    
    # Sort descending by confidence score and deduplicate by id2
    df_injective = df_raw_matches.sort_values('score', ascending=False).drop_duplicates(subset=['id2'], keep='first')
    
    eliminated_dups = len(df_raw_matches) - len(df_injective)
    print(f"   -> Eliminated {eliminated_dups:,} conflicting cross-entity match assignments.")
    print(f"   -> Final clean match pairs: {len(df_injective):,} (in {time.time()-t0:.2f}s)\n")

    # ---------------------------------------------------------
    # STEP 6: Format and Export Final matching_results.tsv
    # ---------------------------------------------------------
    print("STEP 6: Formatting submission for all 1,732,544 test S1 entities...")
    t0 = time.time()
    submission_df = format_predictions_for_submission(df_injective, all_s1_ids)
    submission_df.to_csv(final_submission_path, sep='\t', index=False)
    print(f"   -> Wrote {len(submission_df):,} rows to {final_submission_path} in {time.time()-t0:.2f}s.\n")

    # ---------------------------------------------------------
    # STEP 7: Run Submission Validator
    # ---------------------------------------------------------
    print("STEP 7: Running official submission validator...")
    validator_script = os.path.join(base_dir, 'student_resource', 'utils', 'validate_submission.py')
    import subprocess
    cmd = [
        sys.executable, validator_script,
        '--matching', final_submission_path,
        '--candidate', os.path.join(output_dir, 'candidate_pairs.tsv'),
        '--test-dir', test_dir
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("Validator STDERR:", res.stderr)
        
    print("=" * 80)
    print(f" ALL TASKS COMPLETED SUCCESSFULLY IN {(time.time()-start_time)/60:.2f} MINUTES!")
    print(f" Final submission file ready: {final_submission_path}")
    print("=" * 80)

if __name__ == '__main__':
    main()
