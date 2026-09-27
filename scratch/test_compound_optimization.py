import pandas as pd
from collections import Counter

s1_country = {}
with open('student_resource/dataset/test/test_source1.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\n').split('\t')
        s1_country[p[0]] = p[3] if len(p) > 3 else 'Unknown'

total_s1 = len(s1_country)

df = pd.read_csv('output/raw_matches.tsv', sep='\t')
df['country'] = df['id1'].map(s1_country)

print(f"Loaded {len(df):,} raw matches.")

configs = [
    {
        "name": "Current Leaderboard File (US 0.50, IN 0.54, FR 0.72, Top 8)",
        "us": 0.50, "in": 0.54, "fr": 0.72, "rel_ratio": 0.0, "max_k": 8
    },
    {
        "name": "Config A: Precision Boost (US 0.52, IN 0.56, FR 0.76, Top 6)",
        "us": 0.52, "in": 0.56, "fr": 0.76, "rel_ratio": 0.0, "max_k": 6
    },
    {
        "name": "Config B: Dynamic Margin Pruning (US 0.50, IN 0.54, FR 0.72, Rel 0.65, Top 6)",
        "us": 0.50, "in": 0.54, "fr": 0.72, "rel_ratio": 0.65, "max_k": 6
    },
    {
        "name": "Config C: Precision Sweet Spot (US 0.54, IN 0.58, FR 0.78, Rel 0.65, Top 6)",
        "us": 0.54, "in": 0.58, "fr": 0.78, "rel_ratio": 0.65, "max_k": 6
    },
    {
        "name": "Config D: Ultra-Clean Precision Ceiling (US 0.58, IN 0.62, FR 0.80, Rel 0.70, Top 5)",
        "us": 0.58, "in": 0.62, "fr": 0.80, "rel_ratio": 0.70, "max_k": 5
    }
]

print(f"{'Config Name':<60} | {'Matches':<10} | {'Non-Empty S1':<22} | {'Singletons':<22} | {'Avg/S1'}")
print("-" * 130)

for cfg in configs:
    mask = ((df['country'] == 'US') & (df['score'] >= cfg['us'])) | \
           ((df['country'] == 'India') & (df['score'] >= cfg['in'])) | \
           ((df['country'] == 'France') & (df['score'] >= cfg['fr']))
           
    sub = df[mask].sort_values('score', ascending=False)
    inj = sub.drop_duplicates(subset=['id2'], keep='first')
    
    if cfg['rel_ratio'] > 0:
        inj = inj.copy()
        inj['max_s'] = inj.groupby('id1')['score'].transform('max')
        inj = inj[(inj['max_s'] < 0.75) | (inj['score'] >= inj['max_s'] * cfg['rel_ratio'])]
        
    if cfg['max_k'] < 99:
        inj = inj.groupby('id1').head(cfg['max_k'])
        
    num_matches = len(inj)
    non_empty = inj['id1'].nunique()
    singletons = total_s1 - non_empty
    avg_m = num_matches / non_empty if non_empty > 0 else 0
    
    print(f"{cfg['name']:<60} | {num_matches:<10,} | {non_empty:<10,} ({non_empty/total_s1*100:5.2f}%) | {singletons:<10,} ({singletons/total_s1*100:5.2f}%) | {avg_m:.2f}")
