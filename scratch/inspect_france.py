import pandas as pd

# Load test attributes
s1_attr = {}
with open('student_resource/dataset/test/test_source1.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\n').split('\t')
        s1_attr[p[0]] = (p[1] if len(p)>1 else '', p[2] if len(p)>2 else '', p[3] if len(p)>3 else '')

cand_attr = {}
for src in ['test_source2.tsv', 'test_source3.tsv']:
    with open('student_resource/dataset/test/' + src, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split('\t')
            cand_attr[p[0]] = (p[1] if len(p)>1 else '', p[2] if len(p)>2 else '')

df = pd.read_csv('output/raw_matches.tsv', sep='\t')
df['country'] = df['id1'].map(lambda x: s1_attr.get(x, ('','',''))[2])

france_df = df[df['country'] == 'France']
print('--- France Sample with score 0.40 - 0.50 (low confidence) ---')
sample_low = france_df[(france_df['score'] >= 0.40) & (france_df['score'] <= 0.50)].head(5)
for _, r in sample_low.iterrows():
    s1 = s1_attr.get(r['id1'], ('','',''))
    c2 = cand_attr.get(r['id2'], ('',''))
    print(f"Score: {r['score']:.4f}")
    print(f"  S1: [{s1[0]}] | [{s1[1]}]")
    print(f"  C2: [{c2[0]}] | [{c2[1]}]")

print('\n--- France Sample with score >= 0.80 (high confidence) ---')
sample_high = france_df[france_df['score'] >= 0.80].head(5)
for _, r in sample_high.iterrows():
    s1 = s1_attr.get(r['id1'], ('','',''))
    c2 = cand_attr.get(r['id2'], ('',''))
    print(f"Score: {r['score']:.4f}")
    print(f"  S1: [{s1[0]}] | [{s1[1]}]")
    print(f"  C2: [{c2[0]}] | [{c2[1]}]")
