import sys
import pandas as pd
import numpy as np
from collections import defaultdict

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
all_s1 = list(gt_dict.keys())

# Veto rules
bldg_disqualify_mask = (df['clean_st_mismatch'] == 1) & (df['name_tsr'] < 85)
df.loc[bldg_disqualify_mask, 'score'] = 0.0

brand_collision_mask = (df['brand_first'] < 0.35) & (df['name_tsr'] < 50)
df.loc[brand_collision_mask, 'score'] = 0.0

# Baseline at T=0.45
base_df = df[df['score'] >= 0.45].sort_values('score', ascending=False)
base_inj = base_df.drop_duplicates(subset=['id2'], keep='first')
preds_dict = defaultdict(list)
for r in zip(base_inj['id1'], base_inj['id2']):
    preds_dict[r[0]].append(r[1])
base_res = compute_macro_f05(gt_dict, preds_dict, s1_ids=all_s1)
print(f"Baseline | Macro F0.5: {base_res['macro_f05']:.4f} | Prec: {base_res['macro_precision']:.4f} | Rec: {base_res['macro_recall']:.4f} | Matches: {len(base_inj):,}")

# Test dynamic relative pruning:
# If top_score >= 0.80, only keep other matches if score >= top_score * ratio
for ratio in [0.55, 0.60, 0.65, 0.70, 0.75]:
    test_df = base_inj.copy()
    test_df['max_s'] = test_df.groupby('id1')['score'].transform('max')
    
    # Prune weak tail when strong top match exists
    keep_mask = (test_df['max_s'] < 0.75) | (test_df['score'] >= test_df['max_s'] * ratio)
    pruned = test_df[keep_mask]
    
    preds_dict = defaultdict(list)
    for r in zip(pruned['id1'], pruned['id2']):
        preds_dict[r[0]].append(r[1])
    res = compute_macro_f05(gt_dict, preds_dict, s1_ids=all_s1)
    print(f"Top-Match Relative Ratio >= {ratio:.2f} | Macro F0.5: {res['macro_f05']:.4f} | Prec: {res['macro_precision']:.4f} | Rec: {res['macro_recall']:.4f} | Matches: {len(pruned):,}")

# Test source-level constraint:
# An S1 entity can have matches from Source 2 and Source 3.
# Can an S1 entity match 10 entities from Source 2? No!
# What if we cap max matches per source (e.g. max 3 S2 and max 3 S3)?
test_df = base_inj.copy()
test_df['src'] = test_df['id2'].str[:2]
test_df['src_rank'] = test_df.groupby(['id1', 'src']).cumcount() + 1
for max_per_src in [2, 3, 4, 5]:
    src_pruned = test_df[test_df['src_rank'] <= max_per_src]
    preds_dict = defaultdict(list)
    for r in zip(src_pruned['id1'], src_pruned['id2']):
        preds_dict[r[0]].append(r[1])
    res = compute_macro_f05(gt_dict, preds_dict, s1_ids=all_s1)
    print(f"Max {max_per_src} matches per source | Macro F0.5: {res['macro_f05']:.4f} | Prec: {res['macro_precision']:.4f} | Rec: {res['macro_recall']:.4f} | Matches: {len(src_pruned):,}")
