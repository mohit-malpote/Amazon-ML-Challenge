import os, sys, time
from collections import defaultdict
import pandas as pd
import numpy as np

sys.path.insert(0, '.')
from src.evaluate_local import compute_macro_f05

val_gt_file = 'data/processed/val_ground_truth.tsv'
cache_path = 'data/processed/val_cache_df_scores.parquet'

gt_dict = {}
with open(val_gt_file, 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\n').split('\t')
        gt_dict[p[0]] = set(m.strip() for m in p[1].split(',') if m.strip()) if len(p) > 1 and p[1].strip() else set()

df = pd.read_parquet(cache_path)

# Apply precision veto rules
bldg_disqualify_mask = (df['clean_st_mismatch'] == 1) & (df['name_tsr'] < 85)
df.loc[bldg_disqualify_mask, 'score'] = 0.0

brand_collision_mask = (df['brand_first'] < 0.35) & (df['name_tsr'] < 50)
df.loc[brand_collision_mask, 'score'] = 0.0

all_s1 = list(gt_dict.keys())

print("Testing Strategy A: Standard Threshold with Injective Deduplication")
for t in [0.35, 0.40, 0.45, 0.50, 0.55]:
    matched = df[df['score'] >= t]
    inj = matched.sort_values('score', ascending=False).drop_duplicates(subset=['id2'], keep='first')
    
    preds_dict = defaultdict(list)
    for r in zip(inj['id1'], inj['id2']):
        preds_dict[r[0]].append(r[1])
        
    res = compute_macro_f05(gt_dict, preds_dict, s1_ids=all_s1)
    print(f"Base T={t:.2f} | Macro F0.5: {res['macro_f05']:.4f} | Prec: {res['macro_precision']:.4f} | Rec: {res['macro_recall']:.4f} | SingAcc: {res['singleton_accuracy']:.4f}")

print("\nTesting Strategy B: Minimum Peak Confidence per S1 (Singleton Filter)")
# If an S1's best match is below min_peak, drop ALL matches for that S1 (treat as singleton)
for base_t in [0.35, 0.40]:
    for min_peak in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        matched = df[df['score'] >= base_t]
        inj = matched.sort_values('score', ascending=False).drop_duplicates(subset=['id2'], keep='first')
        
        # Group by id1 and find max score
        max_scores = inj.groupby('id1')['score'].max()
        valid_s1 = set(max_scores[max_scores >= min_peak].index)
        
        inj_filtered = inj[inj['id1'].isin(valid_s1)]
        
        preds_dict = defaultdict(list)
        for r in zip(inj_filtered['id1'], inj_filtered['id2']):
            preds_dict[r[0]].append(r[1])
            
        res = compute_macro_f05(gt_dict, preds_dict, s1_ids=all_s1)
        print(f"Base T={base_t:.2f} + MinPeak={min_peak:.2f} | Macro F0.5: {res['macro_f05']:.4f} | Prec: {res['macro_precision']:.4f} | Rec: {res['macro_recall']:.4f} | SingAcc: {res['singleton_accuracy']:.4f}")

print("\nTesting Strategy C: Relative Threshold (Keep matches within margin of top match)")
# E.g. score >= base_t AND score >= max_score * ratio
for base_t in [0.35, 0.40]:
    for ratio in [0.60, 0.70, 0.80, 0.85]:
        matched = df[df['score'] >= base_t]
        inj = matched.sort_values('score', ascending=False).drop_duplicates(subset=['id2'], keep='first')
        
        inj['max_s'] = inj.groupby('id1')['score'].transform('max')
        inj_rel = inj[inj['score'] >= inj['max_s'] * ratio]
        
        preds_dict = defaultdict(list)
        for r in zip(inj_rel['id1'], inj_rel['id2']):
            preds_dict[r[0]].append(r[1])
            
        res = compute_macro_f05(gt_dict, preds_dict, s1_ids=all_s1)
        print(f"Base T={base_t:.2f} + RelRatio={ratio:.2f} | Macro F0.5: {res['macro_f05']:.4f} | Prec: {res['macro_precision']:.4f} | Rec: {res['macro_recall']:.4f} | Matches: {len(inj_rel):,}")
