#!/usr/bin/env python3
import sys, os, time, gc
from collections import defaultdict
import pandas as pd
import numpy as np
import lightgbm as lgb
import jellyfish, rapidfuzz

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(base_dir, 'src'))

from matching_pipeline import (
    clean_legal, clean_brand_name, get_first_token, get_leading_num, get_zip,
    FEATURE_COLUMNS
)
from evaluate_local import compute_macro_f05, print_evaluation_report

def main():
    t_start = time.time()
    print("=" * 80)
    print(" EVALUATING EXACT VALIDATION BENCHMARK (50,000 S1 Entities)")
    print("=" * 80)

    val_s1_file = os.path.join(base_dir, 'data', 'processed', 'val_s1_sample.tsv')
    val_cand_file = os.path.join(base_dir, 'data', 'processed', 'candidate_pairs_val.tsv')
    val_gt_file = os.path.join(base_dir, 'data', 'processed', 'val_ground_truth.tsv')
    model_file = os.path.join(base_dir, 'output', 'stage1_lgb_model.txt')
    train_dir = os.path.join(base_dir, 'student_resource', 'dataset', 'train')

    # 1. Load Ground Truth
    print("[1/6] Loading Ground Truth Labels...")
    gt_dict = {}
    with open(val_gt_file, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split('\t')
            s1 = p[0]
            ms = p[1].split(',') if len(p) > 1 and p[1].strip() else []
            gt_dict[s1] = set(m.strip() for m in ms if m.strip())
    print(f"   -> Loaded GT for {len(gt_dict):,} S1 entities.")

    # 2. Parse candidates and find needed IDs
    print("[2/6] Parsing candidate pairs...")
    raw_pairs = []
    needed_cand_ids = set()
    with open(val_cand_file, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split('\t')
            s1 = p[0]
            if len(p) > 1 and p[1].strip():
                c_list = [c.strip() for c in p[1].split(',') if c.strip()]
                for c2 in c_list:
                    raw_pairs.append((s1, c2))
                    needed_cand_ids.add(c2)
    print(f"   -> Total candidate pairs: {len(raw_pairs):,} referencing {len(needed_cand_ids):,} unique candidates.")

    # 3. Load attributes
    print("[3/6] Loading entity attributes...")
    s1_dict = {}
    with open(val_s1_file, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split('\t')
            eid = p[0]
            nm = p[1] if len(p) > 1 and p[1] else ''
            ad = p[2].lower() if len(p) > 2 and p[2] else ''
            ct = p[3] if len(p) > 3 and p[3] else ''
            nm_clean = clean_brand_name(nm).lower()
            s1_dict[eid] = (nm_clean, ad, ct, get_first_token(nm), clean_legal(nm), get_leading_num(ad), get_zip(ad))

    cand_dict = {}
    for src in ['train_source2.tsv', 'train_source3.tsv']:
        with open(os.path.join(train_dir, src), 'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                p = line.rstrip('\n').split('\t')
                eid = p[0]
                if eid in needed_cand_ids:
                    nm = p[1] if len(p) > 1 and p[1] else ''
                    ad = p[2].lower() if len(p) > 2 and p[2] else ''
                    ct = p[3] if len(p) > 3 else ''
                    nm_clean = clean_brand_name(nm).lower()
                    cand_dict[eid] = (nm_clean, ad, ct, get_first_token(nm), clean_legal(nm), get_leading_num(ad), get_zip(ad))
    print(f"   -> Loaded {len(s1_dict):,} S1 and {len(cand_dict):,} candidate attributes.")

    # 4. Filter cross-country & build arrays
    print("[4/6] Vectorizing features for valid pairs...")
    valid_pairs = []
    for s1, c2 in raw_pairs:
        e1 = s1_dict.get(s1)
        e2 = cand_dict.get(c2)
        if not e1 or not e2:
            continue
        if e1[2] and e2[2] and e1[2] != e2[2]:
            continue
        valid_pairs.append((s1, c2, e1, e2))
    print(f"   -> Valid same-country pairs: {len(valid_pairs):,}")

    n_pairs = len(valid_pairs)
    n1 = [p[2][0] for p in valid_pairs]
    a1 = [p[2][1] for p in valid_pairs]
    f1 = [p[2][3] for p in valid_pairs]
    cl1 = [p[2][4] for p in valid_pairs]
    num1 = [p[2][5] for p in valid_pairs]
    z1 = [p[2][6] for p in valid_pairs]

    n2 = [p[3][0] for p in valid_pairs]
    a2 = [p[3][1] for p in valid_pairs]
    f2 = [p[3][3] for p in valid_pairs]
    cl2 = [p[3][4] for p in valid_pairs]
    num2 = [p[3][5] for p in valid_pairs]
    z2 = [p[3][6] for p in valid_pairs]

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

    # 5. Predict with Model
    print("[5/6] Predicting with LightGBM model...")
    model = lgb.Booster(model_file=model_file)
    scores = model.predict(X)

    cache_path = os.path.join(base_dir, 'data', 'processed', 'val_cache_df_scores.parquet')
    
    if os.path.exists(cache_path):
        print(f"[4-5/6] Loading precomputed predictions from {cache_path} in 1s...")
        df_scores = pd.read_parquet(cache_path)
    else:
        pair_s1 = [p[0] for p in valid_pairs]
        pair_c2 = [p[1] for p in valid_pairs]
        df_scores = pd.DataFrame({
            'id1': pair_s1,
            'id2': pair_c2,
            'score': scores,
            'exact_name': feat_exact_name,
            'name_lev_r': feat_name_lev_r,
            'name_tsr': feat_name_tsr,
            'brand_first': feat_brand_first,
            'clean_st_mismatch': feat_clean_st_mismatch,
            'clean_st_match': feat_clean_st_match,
            'a1_len': [len(a) for a in a1],
            'a2_len': [len(a) for a in a2]
        })
        print(f"Caching precomputed scores to {cache_path}...")
        df_scores.to_parquet(cache_path)

    # Clean rules without indiscriminate empty address over-boosting
    # Rule 1: Hard Building/Plot Number Disqualification
    bldg_disqualify_mask = (df_scores['clean_st_mismatch'] == 1) & (df_scores['name_tsr'] < 85)
    df_scores.loc[bldg_disqualify_mask, 'score'] = 0.0

    # Rule 2: Brand Collision Disqualification (different brand first token at same address/locality)
    brand_collision_mask = (df_scores['brand_first'] < 0.35) & (df_scores['name_tsr'] < 50)
    df_scores.loc[brand_collision_mask, 'score'] = 0.0

    # 6. Evaluate with Injective Deduplication across thresholds
    print("[6/6] Evaluating with Injective 1-to-1 Deduplication across thresholds...")
    all_s1_ids = list(gt_dict.keys())
    
    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    best_f05 = -1
    best_t = 0.35
    best_rep = None

    for t in thresholds:
        matched = df_scores[df_scores['score'] >= t]
        # Injective deduplication: keep highest scoring match per candidate id2
        inj = matched.sort_values('score', ascending=False).drop_duplicates(subset=['id2'], keep='first')
        
        preds_dict = defaultdict(list)
        for _, row in inj.iterrows():
            preds_dict[row['id1']].append(row['id2'])

        res = compute_macro_f05(gt_dict, preds_dict, s1_ids=all_s1_ids)
        print(f" Threshold {t:.2f} | Macro F0.5: {res['macro_f05']:.4f} | Prec: {res['macro_precision']:.4f} | Rec: {res['macro_recall']:.4f} | Micro Prec: {res['micro_precision']:.4f} | Matches: {len(inj):,}")
        
        if res['macro_f05'] > best_f05:
            best_f05 = res['macro_f05']
            best_t = t
            best_rep = res

    print("\n" + "=" * 80)
    print(f" BEST LOCAL VALIDATION MACRO F0.5: {best_f05:.4f} at Threshold {best_t:.2f}")
    print("=" * 80)
    print_evaluation_report(best_rep, title=f"VALIDATION REPORT AT BEST THRESHOLD {best_t:.2f}")
    print(f" Evaluation completed in {(time.time()-t_start)/60:.2f} minutes.")

if __name__ == '__main__':
    main()
