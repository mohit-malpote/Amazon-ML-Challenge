import os, sys
from collections import defaultdict
import pandas as pd
import numpy as np

sys.path.insert(0, '.')
from src.evaluate_local import compute_macro_f05

val_gt_file = 'data/processed/val_ground_truth.tsv'
cache_path = 'data/processed/val_cache_df_scores.parquet'
val_s1_file = 'data/processed/val_s1_sample.tsv'

country_map = {}
with open(val_s1_file, 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\n').split('\t')
        country_map[p[0]] = p[3] if len(p) > 3 else 'Unknown'

gt_dict = {}
with open(val_gt_file, 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\n').split('\t')
        gt_dict[p[0]] = set(m.strip() for m in p[1].split(',') if m.strip()) if len(p) > 1 and p[1].strip() else set()

df = pd.read_parquet(cache_path)
df['country'] = df['id1'].map(country_map)

# Veto rules
bldg_disqualify_mask = (df['clean_st_mismatch'] == 1) & (df['name_tsr'] < 85)
df.loc[bldg_disqualify_mask, 'score'] = 0.0

brand_collision_mask = (df['brand_first'] < 0.35) & (df['name_tsr'] < 50)
df.loc[brand_collision_mask, 'score'] = 0.0

all_s1 = list(gt_dict.keys())

print("=== Experiment 1: Country-Calibrated Thresholds ===")
# Try US thresh from 0.35 to 0.45, India thresh from 0.45 to 0.60
for us_t in [0.35, 0.40, 0.45]:
    for in_t in [0.45, 0.50, 0.55]:
        mask = ((df['country'] == 'US') & (df['score'] >= us_t)) | \
               ((df['country'] == 'India') & (df['score'] >= in_t))
        matched = df[mask]
        inj = matched.sort_values('score', ascending=False).drop_duplicates(subset=['id2'], keep='first')
        
        preds_dict = defaultdict(list)
        for r in zip(inj['id1'], inj['id2']):
            preds_dict[r[0]].append(r[1])
            
        res = compute_macro_f05(gt_dict, preds_dict, s1_ids=all_s1)
        print(f"US_T={us_t:.2f}, IN_T={in_t:.2f} | Macro F0.5: {res['macro_f05']:.4f} | Prec: {res['macro_precision']:.4f} | Rec: {res['macro_recall']:.4f}")

print("\n=== Experiment 2: Score Margin from Best Match per S1 ===")
# score >= base_t AND score >= max_score - margin
for base_t in [0.35, 0.40, 0.45]:
    for margin in [0.20, 0.25, 0.30, 0.35]:
        matched = df[df['score'] >= base_t].copy()
        matched['max_s'] = matched.groupby('id1')['score'].transform('max')
        matched = matched[matched['score'] >= matched['max_s'] - margin]
        
        inj = matched.sort_values('score', ascending=False).drop_duplicates(subset=['id2'], keep='first')
        preds_dict = defaultdict(list)
        for r in zip(inj['id1'], inj['id2']):
            preds_dict[r[0]].append(r[1])
            
        res = compute_macro_f05(gt_dict, preds_dict, s1_ids=all_s1)
        print(f"Base_T={base_t:.2f}, Margin={margin:.2f} | Macro F0.5: {res['macro_f05']:.4f} | Prec: {res['macro_precision']:.4f} | Rec: {res['macro_recall']:.4f}")

print("\n=== Experiment 3: Max Matches per S1 Cap (Top-K) ===")
for base_t in [0.35, 0.40, 0.45]:
    for k in [4, 5, 6, 7]:
        matched = df[df['score'] >= base_t].sort_values('score', ascending=False)
        inj = matched.drop_duplicates(subset=['id2'], keep='first')
        # Cap top-k per id1
        inj_k = inj.groupby('id1').head(k)
        
        preds_dict = defaultdict(list)
        for r in zip(inj_k['id1'], inj_k['id2']):
            preds_dict[r[0]].append(r[1])
            
        res = compute_macro_f05(gt_dict, preds_dict, s1_ids=all_s1)
        print(f"Base_T={base_t:.2f}, Top-{k} | Macro F0.5: {res['macro_f05']:.4f} | Prec: {res['macro_precision']:.4f} | Rec: {res['macro_recall']:.4f}")
