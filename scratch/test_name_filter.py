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

df['is_true'] = [c2 in gt_dict.get(s1, set()) for s1, c2 in zip(df['id1'], df['id2'])]
true_pairs = df[df['is_true'] == True]
false_pairs = df[df['is_true'] == False]

print(f"Total True Pairs in Val: {len(true_pairs):,}")
true_low = len(true_pairs[true_pairs['name_tsr'] < 40])
print(f"True pairs with name_tsr < 40: {true_low:,} ({true_low/len(true_pairs)*100:.2f}%)")
false_high = len(false_pairs[(false_pairs['score'] >= 0.50) & (false_pairs['name_tsr'] < 40)])
print(f"False pairs with score >= 0.50 and name_tsr < 40: {false_high:,}")

all_s1 = list(gt_dict.keys())

print("\n--- Testing Impact on Local Validation Macro F0.5 ---")
for min_name in [0, 30, 35, 40, 45, 50]:
    matched = df[(df['score'] >= 0.45) & (df['name_tsr'] >= min_name)]
    inj = matched.sort_values('score', ascending=False).drop_duplicates(subset=['id2'], keep='first')
    
    preds_dict = defaultdict(list)
    for r in zip(inj['id1'], inj['id2']):
        preds_dict[r[0]].append(r[1])
        
    res = compute_macro_f05(gt_dict, preds_dict, s1_ids=all_s1)
    print(f"Min Name TSR >= {min_name:2d} | Macro F0.5: {res['macro_f05']:.4f} | Prec: {res['macro_precision']:.4f} | Rec: {res['macro_recall']:.4f} | Matches: {len(inj):,}")
