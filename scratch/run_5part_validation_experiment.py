#!/usr/bin/env python3
"""
5-Part Out-of-Sample Generalization Experiment for ML Challenge 2026.
Tests for bias and overfitting by:
1. Selecting 60,000 fresh S1 entities strictly disjoint from any previous validation set.
2. Splitting into 5 equal parts:
   - Parts 1-4 (48,000 entities, ~2M candidate pairs) -> Training
   - Part 5 (12,000 entities, ~500k candidate pairs) -> Unseen Holdout Test
3. Running multi-channel blocking on both partitions.
4. Training a brand new LightGBM model from scratch on Parts 1-4.
5. Evaluating inference strictly on Part 5 against train_ground_truth.tsv.
6. Reporting exact competition Macro F0.5, Precision, and Recall.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True, encoding='utf-8', errors='replace')

import os
import time
import random
from collections import defaultdict
import pandas as pd
import numpy as np
import lightgbm as lgb

# Ensure src is on path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(base_dir, 'src'))

from blocking import run_blocking_for_country, merge_country_files
from matching_pipeline import FEATURE_COLUMNS, compute_features_fast
from evaluate_local import compute_macro_f05, print_evaluation_report

def main():
    start_time = time.time()
    print("=" * 85, flush=True)
    print(" 5-PART GENERALIZATION EXPERIMENT: TESTING MODEL ON UNSEEN DATA (2M PAIRS)", flush=True)
    print("=" * 85, flush=True)

    train_dir = os.path.join(base_dir, 'student_resource', 'dataset', 'train')
    exp_dir = os.path.join(base_dir, 'data', 'experiment_5part')
    os.makedirs(exp_dir, exist_ok=True)

    s1_full_file = os.path.join(train_dir, 'train_source1.tsv')
    gt_full_file = os.path.join(train_dir, 'train_ground_truth.tsv')
    cand_sources = [
        os.path.join(train_dir, 'train_source2.tsv'),
        os.path.join(train_dir, 'train_source3.tsv')
    ]

    # -------------------------------------------------------------
    # STEP 1: Select 60,000 Fresh S1 Entities (Exclude prior 50k val)
    # -------------------------------------------------------------
    print("\n[Step 1/6] Selecting 60,000 fresh S1 entities disjoint from previous splits...", flush=True)
    prev_val_file = os.path.join(base_dir, 'data', 'processed', 'val_s1_sample.tsv')
    prev_seen_ids = set()
    if os.path.exists(prev_val_file):
        with open(prev_val_file, 'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                prev_seen_ids.add(line.split('\t')[0])
    print(f"   -> Excluded {len(prev_seen_ids):,} previously used validation IDs.", flush=True)

    fresh_s1_records = []
    with open(s1_full_file, 'r', encoding='utf-8') as f:
        header = f.readline().rstrip('\n')
        for line in f:
            eid = line.split('\t')[0]
            if eid not in prev_seen_ids:
                fresh_s1_records.append(line)
            if len(fresh_s1_records) >= 60000:
                break

    print(f"   -> Extracted {len(fresh_s1_records):,} fresh S1 entities.", flush=True)

    # Split into 5 equal parts (4 parts Train = 48,000, 1 part Test = 12,000)
    random.seed(1337)
    random.shuffle(fresh_s1_records)

    train_records = fresh_s1_records[:48000]
    test_records = fresh_s1_records[48000:]

    s1_train_file = os.path.join(exp_dir, 'exp_s1_train_parts1to4.tsv')
    s1_test_file = os.path.join(exp_dir, 'exp_s1_test_part5.tsv')

    with open(s1_train_file, 'w', encoding='utf-8') as f:
        f.write(header + '\n')
        f.writelines(train_records)

    with open(s1_test_file, 'w', encoding='utf-8') as f:
        f.write(header + '\n')
        f.writelines(test_records)

    test_s1_ids = [line.split('\t')[0] for line in test_records]
    print(f"   -> Parts 1-4 (Train Split): {len(train_records):,} entities saved to {os.path.basename(s1_train_file)}", flush=True)
    print(f"   -> Part 5 (Holdout Test):   {len(test_records):,} entities saved to {os.path.basename(s1_test_file)}", flush=True)

    # -------------------------------------------------------------
    # STEP 2: Multi-Channel Blocking on Train (Parts 1-4) & Test (Part 5)
    # -------------------------------------------------------------
    print("\n[Step 2/6] Running multi-channel blocking for Train (Parts 1-4) and Test (Part 5)...", flush=True)

    # Blocking Train (Parts 1-4)
    train_cands_file = os.path.join(exp_dir, 'candidate_pairs_train_parts1to4.tsv')
    if not os.path.exists(train_cands_file):
        train_country_files = []
        for c in ['India', 'US']:
            c_out = os.path.join(exp_dir, f'candidate_pairs_train_{c}.tsv')
            run_blocking_for_country(c, s1_train_file, cand_sources, c_out, max_cands_per_s1=60)
            train_country_files.append(c_out)
        merge_country_files(train_country_files, train_cands_file)
    else:
        print(f"   -> Existing {os.path.basename(train_cands_file)} found, reusing.", flush=True)

    # Blocking Test (Part 5)
    test_cands_file = os.path.join(exp_dir, 'candidate_pairs_test_part5.tsv')
    if not os.path.exists(test_cands_file):
        test_country_files = []
        for c in ['India', 'US']:
            c_out = os.path.join(exp_dir, f'candidate_pairs_test_{c}.tsv')
            run_blocking_for_country(c, s1_test_file, cand_sources, c_out, max_cands_per_s1=60)
            test_country_files.append(c_out)
        merge_country_files(test_country_files, test_cands_file)
    else:
        print(f"   -> Existing {os.path.basename(test_cands_file)} found, reusing.", flush=True)

    # -------------------------------------------------------------
    # STEP 3: Load Ground Truth for Evaluation
    # -------------------------------------------------------------
    print("\n[Step 3/6] Loading Ground Truth Labels...", flush=True)
    gt_dict = {}
    with open(gt_full_file, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split('\t')
            s1 = p[0]
            ms = p[1].split(',') if len(p) > 1 and p[1].strip() else []
            gt_dict[s1] = set(m.strip() for m in ms if m.strip())
    print(f"   -> Loaded ground truth mapping for {len(gt_dict):,} S1 entities.", flush=True)

    # Check blocking recall on Part 5 holdout
    test_true_matches = 0
    test_captured_matches = 0
    with open(test_cands_file, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            s1, _, cs = line.rstrip('\n').partition('\t')
            true_s = gt_dict.get(s1, set())
            test_true_matches += len(true_s)
            c_set = set(cs.split(',')) if cs.strip() else set()
            test_captured_matches += len(true_s.intersection(c_set))

    print(f"   -> Part 5 Blocking Recall: {test_captured_matches:,} / {test_true_matches:,} ({test_captured_matches/test_true_matches*100:.2f}%)", flush=True)

    # -------------------------------------------------------------
    # STEP 4: Train Fresh LightGBM Model Strictly on Parts 1-4 (~2M pairs)
    # -------------------------------------------------------------
    print("\n[Step 4/6] Parsing training pairs from Parts 1-4...", flush=True)
    train_pairs = []
    needed_s1_train = set()
    needed_cand_train = set()

    with open(train_cands_file, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split('\t')
            s1 = p[0]
            if len(p) > 1 and p[1].strip():
                for c2 in p[1].split(','):
                    c2 = c2.strip()
                    if c2:
                        train_pairs.append((s1, c2))
                        needed_s1_train.add(s1)
                        needed_cand_train.add(c2)

    print(f"   -> Total candidate pairs in Parts 1-4: {len(train_pairs):,}", flush=True)

    # Sample 400,000 pairs from Parts 1-4 for fast booster training
    random.seed(42)
    sample_train_pairs = random.sample(train_pairs, min(400000, len(train_pairs)))
    needed_s1_sample = set(p[0] for p in sample_train_pairs)
    needed_cand_sample = set(p[1] for p in sample_train_pairs)

    print(f"   -> Loading entity attributes for {len(sample_train_pairs):,} training pairs...", flush=True)
    s1_train_attrs = {}
    with open(s1_full_file, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split('\t')
            if p[0] in needed_s1_sample:
                s1_train_attrs[p[0]] = (
                    p[1] if len(p) > 1 else '',
                    p[2] if len(p) > 2 else '',
                    p[3] if len(p) > 3 else ''
                )

    cand_train_attrs = {}
    for src in cand_sources:
        with open(src, 'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                p = line.rstrip('\n').split('\t')
                if p[0] in needed_cand_sample:
                    cand_train_attrs[p[0]] = (
                        p[1] if len(p) > 1 else '',
                        p[2] if len(p) > 2 else '',
                        p[3] if len(p) > 3 else ''
                    )

    # Build Training DataFrame
    all_s1, all_c2, targets = [], [], []
    for s1, c2 in sample_train_pairs:
        if s1 in s1_train_attrs and c2 in cand_train_attrs:
            all_s1.append(s1)
            all_c2.append(c2)
            targets.append(1 if (s1 in gt_dict and c2 in gt_dict[s1]) else 0)

    print(f"   -> Labeled Training Set: {len(targets):,} pairs (Positives: {sum(targets):,}, Negatives: {len(targets)-sum(targets):,})", flush=True)
    df_train = pd.DataFrame({
        'id1': all_s1, 'id2': all_c2,
        'name1': [s1_train_attrs[x][0] for x in all_s1],
        'addr1': [s1_train_attrs[x][1] for x in all_s1],
        'country1': [s1_train_attrs[x][2] for x in all_s1],
        'name2': [cand_train_attrs[x][0] for x in all_c2],
        'addr2': [cand_train_attrs[x][1] for x in all_c2],
        'country2': [cand_train_attrs[x][2] for x in all_c2],
        'target': targets
    })

    df_train_feats = compute_features_fast(df_train)
    X_tr = df_train_feats[FEATURE_COLUMNS]
    y_tr = df_train_feats['target']

    print(" Training fresh LightGBM model strictly on Parts 1-4...", flush=True)
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.08,
        'num_leaves': 63,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 5,
        'scale_pos_weight': 0.35,
        'verbose': -1,
        'seed': 42
    }
    model = lgb.train(params, dtrain, num_boost_round=120)
    exp_model_path = os.path.join(exp_dir, 'exp_parts1to4_lgb_model.txt')
    model.save_model(exp_model_path)
    print(f" Fresh model saved to {os.path.basename(exp_model_path)}", flush=True)

    # -------------------------------------------------------------
    # STEP 5: Run Inference on Part 5 (The 5th Part - Never Seen)
    # -------------------------------------------------------------
    print("\n[Step 5/6] Running inference on Part 5 holdout (12,000 entities, ~500k candidate pairs)...", flush=True)
    test_pairs = []
    needed_s1_test = set()
    needed_cand_test = set()

    with open(test_cands_file, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split('\t')
            s1 = p[0]
            if len(p) > 1 and p[1].strip():
                for c2 in p[1].split(','):
                    c2 = c2.strip()
                    if c2:
                        test_pairs.append((s1, c2))
                        needed_s1_test.add(s1)
                        needed_cand_test.add(c2)

    print(f"   -> Part 5 candidate pairs: {len(test_pairs):,} across {len(needed_s1_test):,} entities.", flush=True)

    s1_test_attrs = {}
    with open(s1_test_file, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split('\t')
            s1_test_attrs[p[0]] = (
                p[1] if len(p) > 1 else '',
                p[2] if len(p) > 2 else '',
                p[3] if len(p) > 3 else ''
            )

    cand_test_attrs = {}
    for src in cand_sources:
        with open(src, 'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                p = line.rstrip('\n').split('\t')
                if p[0] in needed_cand_test:
                    cand_test_attrs[p[0]] = (
                        p[1] if len(p) > 1 else '',
                        p[2] if len(p) > 2 else '',
                        p[3] if len(p) > 3 else ''
                    )

    test_s1_valid = [p[0] for p in test_pairs if p[0] in s1_test_attrs and p[1] in cand_test_attrs]
    test_c2_valid = [p[1] for p in test_pairs if p[0] in s1_test_attrs and p[1] in cand_test_attrs]

    df_test = pd.DataFrame({
        'id1': test_s1_valid, 'id2': test_c2_valid,
        'name1': [s1_test_attrs[x][0] for x in test_s1_valid],
        'addr1': [s1_test_attrs[x][1] for x in test_s1_valid],
        'country1': [s1_test_attrs[x][2] for x in test_s1_valid],
        'name2': [cand_test_attrs[x][0] for x in test_c2_valid],
        'addr2': [cand_test_attrs[x][1] for x in test_c2_valid],
        'country2': [cand_test_attrs[x][2] for x in test_c2_valid],
    })

    print(f" Computing features for all {len(df_test):,} Part 5 test pairs...", flush=True)
    df_test_feats = compute_features_fast(df_test)
    test_probs = model.predict(df_test_feats[FEATURE_COLUMNS])
    df_test_feats['prob'] = test_probs

    # -------------------------------------------------------------
    # STEP 6: Cross-Check Against Ground Truth File on Part 5
    # -------------------------------------------------------------
    print("\n" + "=" * 85, flush=True)
    print(" [Step 6/6] EVALUATING UNSEEN PART 5 RESULTS AGAINST GROUND TRUTH", flush=True)
    print("=" * 85, flush=True)

    for t in [0.18, 0.25, 0.30, 0.35, 0.40, 0.50]:
        matched_subset = df_test_feats[df_test_feats['prob'] >= t]
        preds_dict = defaultdict(list)
        for _, row in matched_subset.iterrows():
            preds_dict[row['id1']].append(row['id2'])

        res = compute_macro_f05(gt_dict, preds_dict, s1_ids=test_s1_ids)
        print(f" Threshold {t:.2f} | Macro F0.5: {res['macro_f05']:.4f} | Macro Prec: {res['macro_precision']:.4f} | Macro Rec: {res['macro_recall']:.4f} | Micro Prec: {res['micro_precision']:.4f}", flush=True)

    # Detailed report at calibrated threshold 0.35
    matched_035 = df_test_feats[df_test_feats['prob'] >= 0.35]
    final_preds_dict = defaultdict(list)
    for _, row in matched_035.iterrows():
        final_preds_dict[row['id1']].append(row['id2'])

    report_035 = compute_macro_f05(gt_dict, final_preds_dict, s1_ids=test_s1_ids)
    print_evaluation_report(report_035, title="FINAL OUT-OF-SAMPLE TEST PERFORMANCE (PART 5)")

    print(f" EXPERIMENT COMPLETED IN {(time.time()-start_time)/60:.2f} MINUTES!", flush=True)

if __name__ == '__main__':
    main()
