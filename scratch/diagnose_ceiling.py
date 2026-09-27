import sys
from collections import defaultdict
import pandas as pd

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

print(f"{'T':<6} | {'Macro F0.5':<10} | {'Prec':<8} | {'Rec':<8} | {'FN Sing (True match, pred 0)':<30} | {'FP Sing (True 0, pred match)'}")
print("-" * 95)

for t in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
    matched = df[df['score'] >= t]
    inj = matched.sort_values('score', ascending=False).drop_duplicates(subset=['id2'], keep='first')
    preds_dict = defaultdict(list)
    for r in zip(inj['id1'], inj['id2']):
        preds_dict[r[0]].append(r[1])
    res = compute_macro_f05(gt_dict, preds_dict, s1_ids=all_s1)
    
    fn_sing = 0
    fp_sing = 0
    for s1 in all_s1:
        gt_len = len(gt_dict.get(s1, set()))
        pr_len = len(preds_dict.get(s1, []))
        if gt_len > 0 and pr_len == 0:
            fn_sing += 1
        elif gt_len == 0 and pr_len > 0:
            fp_sing += 1
            
    print(f"{t:<6.2f} | {res['macro_f05']:<10.4f} | {res['macro_precision']:<8.4f} | {res['macro_recall']:<8.4f} | {fn_sing:<30,} | {fp_sing:,}")
