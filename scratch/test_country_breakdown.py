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

bldg_disqualify_mask = (df['clean_st_mismatch'] == 1) & (df['name_tsr'] < 85)
df.loc[bldg_disqualify_mask, 'score'] = 0.0

brand_collision_mask = (df['brand_first'] < 0.35) & (df['name_tsr'] < 50)
df.loc[brand_collision_mask, 'score'] = 0.0

for c in ['US', 'India']:
    c_s1 = [k for k, v in country_map.items() if v == c]
    print(f"\n=== Country: {c} ({len(c_s1):,} S1) ===")
    for t in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        matched = df[df['score'] >= t]
        inj = matched.sort_values('score', ascending=False).drop_duplicates(subset=['id2'], keep='first')
        preds_dict = defaultdict(list)
        for r in zip(inj['id1'], inj['id2']):
            preds_dict[r[0]].append(r[1])
        res = compute_macro_f05(gt_dict, preds_dict, s1_ids=c_s1)
        print(f"T={t:.2f} | Macro F0.5: {res['macro_f05']:.4f} | Prec: {res['macro_precision']:.4f} | Rec: {res['macro_recall']:.4f}")
