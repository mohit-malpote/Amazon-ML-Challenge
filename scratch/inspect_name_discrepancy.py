import sys
import pandas as pd
from collections import defaultdict

val_gt_file = 'data/processed/val_ground_truth.tsv'
cache_path = 'data/processed/val_cache_df_scores.parquet'
val_s1_file = 'data/processed/val_s1_sample.tsv'

s1_names = {}
with open(val_s1_file, 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\n').split('\t')
        s1_names[p[0]] = p[1] if len(p) > 1 else ''

cand_names = {}
for src in ['train_source2.tsv', 'train_source3.tsv']:
    p = 'student_resource/dataset/train/' + src
    with open(p, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split('\t')
            cand_names[p[0]] = p[1] if len(p) > 1 else ''

gt_dict = {}
with open(val_gt_file, 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\n').split('\t')
        gt_dict[p[0]] = set(m.strip() for m in p[1].split(',') if m.strip()) if len(p) > 1 and p[1].strip() else set()

df = pd.read_parquet(cache_path)
df['is_true'] = [c2 in gt_dict.get(s1, set()) for s1, c2 in zip(df['id1'], df['id2'])]

# Sample 10 true matches with name_tsr < 40
true_low = df[(df['is_true'] == True) & (df['name_tsr'] < 40)].head(10)
print("=== True Matches with name_tsr < 40 ===")
for _, r in true_low.iterrows():
    n1 = s1_names.get(r['id1'], '')
    n2 = cand_names.get(r['id2'], '')
    print(f"[{n1}] <==> [{n2}] | score={r['score']:.4f} | name_tsr={r['name_tsr']:.1f} | lev_r={r['name_lev_r']:.2f}")

# Sample 10 false matches with score >= 0.50 and name_tsr < 40
false_high = df[(df['is_true'] == False) & (df['score'] >= 0.50) & (df['name_tsr'] < 40)].head(10)
print("\n=== False Matches with score >= 0.50 and name_tsr < 40 ===")
for _, r in false_high.iterrows():
    n1 = s1_names.get(r['id1'], '')
    n2 = cand_names.get(r['id2'], '')
    print(f"[{n1}] <==> [{n2}] | score={r['score']:.4f} | name_tsr={r['name_tsr']:.1f} | lev_r={r['name_lev_r']:.2f}")
