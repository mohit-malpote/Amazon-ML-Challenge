import os, sys
from collections import defaultdict
import rapidfuzz

val_gt_file = 'data/processed/val_ground_truth.tsv'
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

gt_pairs = set()
with open(val_gt_file, 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\n').split('\t')
        s1 = p[0]
        if len(p) > 1 and p[1].strip():
            for m in p[1].split(','):
                gt_pairs.add((s1, m.strip()))

total_gt = len(gt_pairs)
print(f"Total True GT Pairs in Validation: {total_gt:,}")

def has_common_word(n1, n2, min_len=4):
    w1 = set(w.lower() for w in n1.split() if len(w) >= min_len)
    w2 = set(w.lower() for w in n2.split() if len(w) >= min_len)
    return len(w1.intersection(w2)) > 0

# Check how many GT pairs violate the rule
dropped_gt = 0
for s1, c2 in gt_pairs:
    n1 = s1_names.get(s1, '')
    n2 = cand_names.get(c2, '')
    if not n1 or not n2:
        continue
    tsr = rapidfuzz.fuzz.token_set_ratio(n1, n2)
    f1 = n1.split()[0].lower() if n1.split() else ''
    f2 = n2.split()[0].lower() if n2.split() else ''
    fr = rapidfuzz.fuzz.ratio(f1, f2) / 100.0 if (f1 and f2) else 0.0
    common = has_common_word(n1, n2, 4)
    
    # If disjoint
    if tsr < 35 and fr < 0.40 and not common:
        dropped_gt += 1

print(f"GT pairs with tsr < 35, fr < 0.40, and no common word: {dropped_gt:,} / {total_gt:,} ({dropped_gt/total_gt*100:.2f}%)")
