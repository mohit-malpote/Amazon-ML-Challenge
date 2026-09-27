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

options = [
    {
        "name": "Option 1: Balanced High-Precision (US: 0.46, IN: 0.50, FR: 0.70)",
        "us": 0.46, "in": 0.50, "fr": 0.70
    },
    {
        "name": "Option 2: Sweet-Spot Calibrated (US: 0.50, IN: 0.54, FR: 0.72)",
        "us": 0.50, "in": 0.54, "fr": 0.72
    },
    {
        "name": "Option 3: Conservative Precision-Ceiling (US: 0.54, IN: 0.58, FR: 0.75)",
        "us": 0.54, "in": 0.58, "fr": 0.75
    },
    {
        "name": "Option 4: Global T=0.62 (Safe Uniform Cutoff)",
        "us": 0.62, "in": 0.62, "fr": 0.62
    },
    {
        "name": "Option 5: Global T=0.65 (Matches 0.801 Run Volume: 4.28M)",
        "us": 0.65, "in": 0.65, "fr": 0.65
    }
]

for opt in options:
    mask = ((df['country'] == 'US') & (df['score'] >= opt['us'])) | \
           ((df['country'] == 'India') & (df['score'] >= opt['in'])) | \
           ((df['country'] == 'France') & (df['score'] >= opt['fr']))
           
    sub = df[mask].sort_values('score', ascending=False).drop_duplicates(subset=['id2'], keep='first')
    
    us_m = len(sub[sub['country'] == 'US'])
    in_m = len(sub[sub['country'] == 'India'])
    fr_m = len(sub[sub['country'] == 'France'])
    tot_m = len(sub)
    non_empty = sub['id1'].nunique()
    singletons = total_s1 - non_empty
    
    print(f"\n=== {opt['name']} ===")
    print(f"Total Matches: {tot_m:,} | Non-Empty S1: {non_empty:,} ({non_empty/total_s1*100:.2f}%) | Singletons: {singletons:,} ({singletons/total_s1*100:.2f}%)")
    print(f"  US Matches:     {us_m:,} (for 663k S1, avg {us_m/sub[sub['country']=='US']['id1'].nunique():.2f}/S1)")
    print(f"  India Matches:  {in_m:,} (for 810k S1, avg {in_m/sub[sub['country']=='India']['id1'].nunique():.2f}/S1)")
    print(f"  France Matches: {fr_m:,} (for 259k S1, avg {fr_m/sub[sub['country']=='France']['id1'].nunique():.2f}/S1)")
