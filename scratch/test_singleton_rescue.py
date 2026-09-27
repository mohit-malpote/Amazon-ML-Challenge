import sys
import pandas as pd
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

# Current Tier 1.5 logic:
matched = df[df['score'] >= 0.50].sort_values('score', ascending=False)
inj = matched.drop_duplicates(subset=['id2'], keep='first')
preds_dict_base = defaultdict(list)
for r in zip(inj['id1'], inj['id2']):
    preds_dict_base[r[0]].append(r[1])
base_res = compute_macro_f05(gt_dict, preds_dict_base, s1_ids=all_s1)
print(f"Base Tier 1.5 | Macro F0.5: {base_res['macro_f05']:.4f} | Prec: {base_res['macro_precision']:.4f} | Rec: {base_res['macro_recall']:.4f} | Empty: {50000 - len(preds_dict_base):,}")

# What if we rescue empty S1 entities that have a candidate with score >= T_rescue?
# E.g. T_rescue = 0.35, 0.30, 0.25, 0.20
for t_rescue in [0.45, 0.40, 0.35, 0.30, 0.25, 0.20]:
    preds_dict_test = defaultdict(list)
    for k, v in preds_dict_base.items():
        preds_dict_test[k] = list(v)
        
    empty_s1 = set(all_s1) - set(preds_dict_test.keys())
    
    # For empty S1, find top candidate if score >= t_rescue
    empty_cands = df[(df['id1'].isin(empty_s1)) & (df['score'] >= t_rescue)].sort_values('score', ascending=False)
    # Deduplicate id2
    used_id2 = set(inj['id2'])
    empty_cands = empty_cands[~empty_cands['id2'].isin(used_id2)]
    top_rescue = empty_cands.drop_duplicates(subset=['id1'], keep='first')
    
    for r in zip(top_rescue['id1'], top_rescue['id2']):
        preds_dict_test[r[0]].append(r[1])
        
    res = compute_macro_f05(gt_dict, preds_dict_test, s1_ids=all_s1)
    rescued = len(top_rescue)
    print(f"Rescue T>={t_rescue:.2f} (rescued {rescued:,} S1) | Macro F0.5: {res['macro_f05']:.4f} | Prec: {res['macro_precision']:.4f} | Rec: {res['macro_recall']:.4f} | Empty: {50000 - len(preds_dict_test):,}")
