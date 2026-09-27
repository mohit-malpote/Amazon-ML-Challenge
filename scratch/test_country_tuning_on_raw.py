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

configs = [
    # Baseline (0.799 run)
    {"name": "Flat 0.40 (Current 0.799 submission)", "us": 0.40, "in": 0.40, "fr": 0.40, "max_k": 99},
    # Calibrated country thresholds
    {"name": "Calibrated Tier 1 (US 0.45, IN 0.50, FR 0.65)", "us": 0.45, "in": 0.50, "fr": 0.65, "max_k": 99},
    {"name": "Calibrated Tier 2 (US 0.45, IN 0.50, FR 0.70)", "us": 0.45, "in": 0.50, "fr": 0.70, "max_k": 99},
    {"name": "Calibrated Tier 3 (US 0.48, IN 0.52, FR 0.72)", "us": 0.48, "in": 0.52, "fr": 0.72, "max_k": 99},
    {"name": "Calibrated Tier 4 (US 0.50, IN 0.55, FR 0.75)", "us": 0.50, "in": 0.55, "fr": 0.75, "max_k": 99},
    # With Top-K cap per S1
    {"name": "Tier 2 + Cap Top-7 per S1", "us": 0.45, "in": 0.50, "fr": 0.70, "max_k": 7},
    {"name": "Tier 3 + Cap Top-7 per S1", "us": 0.48, "in": 0.52, "fr": 0.72, "max_k": 7},
]

print(f"{'Config Name':<45} | {'Matches':<10} | {'Non-Empty S1':<22} | {'Singletons':<22} | {'Avg/S1'}")
print("-" * 115)

for cfg in configs:
    mask = ((df['country'] == 'US') & (df['score'] >= cfg['us'])) | \
           ((df['country'] == 'India') & (df['score'] >= cfg['in'])) | \
           ((df['country'] == 'France') & (df['score'] >= cfg['fr']))
           
    sub_df = df[mask].sort_values('score', ascending=False)
    inj = sub_df.drop_duplicates(subset=['id2'], keep='first')
    
    if cfg['max_k'] < 99:
        inj = inj.groupby('id1').head(cfg['max_k'])
        
    num_matches = len(inj)
    non_empty = inj['id1'].nunique()
    singletons = total_s1 - non_empty
    avg_m = num_matches / non_empty if non_empty > 0 else 0
    
    print(f"{cfg['name']:<45} | {num_matches:<10,} | {non_empty:<10,} ({non_empty/total_s1*100:5.2f}%) | {singletons:<10,} ({singletons/total_s1*100:5.2f}%) | {avg_m:.2f}")
