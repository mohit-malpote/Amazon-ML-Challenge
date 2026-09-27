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

# Standard veto rules
bldg_disqualify_mask = (df['clean_st_mismatch'] == 1) & (df['name_tsr'] < 85)
df.loc[bldg_disqualify_mask, 'score'] = 0.0

brand_collision_mask = (df['brand_first'] < 0.35) & (df['name_tsr'] < 50)
df.loc[brand_collision_mask, 'score'] = 0.0

print("=== Base Performance at T=0.45 ===")
matched = df[df['score'] >= 0.45]
inj = matched.sort_values('score', ascending=False).drop_duplicates(subset=['id2'], keep='first')
preds_dict = defaultdict(list)
for r in zip(inj['id1'], inj['id2']):
    preds_dict[r[0]].append(r[1])
res = compute_macro_f05(gt_dict, preds_dict, s1_ids=all_s1)
print(f"Base T=0.45 | Macro F0.5: {res['macro_f05']:.4f} | Prec: {res['macro_precision']:.4f} | Rec: {res['macro_recall']:.4f} | Matches: {len(inj):,}")

print("\n=== Testing Name Sanity Guard (Reject Address Collisions with Different Names) ===")
# A pair is an address collision if name_tsr < MIN_TSR and brand_first < MIN_BRAND and name_lev_r < MIN_LEV
for min_tsr in [40, 45, 50]:
    for min_brand in [0.50, 0.60, 0.70]:
        df_test = df.copy()
        # Veto rule: completely disjoint name despite address overlap
        name_veto = (df_test['name_tsr'] < min_tsr) & (df_test['brand_first'] < min_brand) & (df_test['name_lev_r'] < 0.40)
        df_test.loc[name_veto, 'score'] = 0.0
        
        for t in [0.40, 0.45, 0.50]:
            matched = df_test[df_test['score'] >= t]
            inj = matched.sort_values('score', ascending=False).drop_duplicates(subset=['id2'], keep='first')
            preds_dict = defaultdict(list)
            for r in zip(inj['id1'], inj['id2']):
                preds_dict[r[0]].append(r[1])
            res = compute_macro_f05(gt_dict, preds_dict, s1_ids=all_s1)
            print(f"MinTSR={min_tsr}, MinBrand={min_brand:.2f}, T={t:.2f} | Macro F0.5: {res['macro_f05']:.4f} | Prec: {res['macro_precision']:.4f} | Rec: {res['macro_recall']:.4f} | Matches: {len(inj):,}")
