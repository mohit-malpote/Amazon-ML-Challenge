import sys, os
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(base_dir, 'src'))
from blocking import normalize_name, extract_address_keys, get_phonetic_keys, get_token_keys
import pandas as pd

print("Loading data...")
gt = pd.read_csv(os.path.join(base_dir, 'student_resource', 'dataset', 'train', 'train_ground_truth.tsv'), sep='\t', nrows=200)
gt_clean = gt.dropna(subset=['matched_entity_ids']).copy()
gt_clean['matched_entity_ids'] = gt_clean['matched_entity_ids'].astype(str).str.split(',')
gt_flat = gt_clean.explode('matched_entity_ids')
gt_flat['matched_entity_ids'] = gt_flat['matched_entity_ids'].str.strip()
sample_pairs = list(zip(gt_flat['source1_entity_id'], gt_flat['matched_entity_ids']))[:15]

s1 = pd.read_csv(os.path.join(base_dir, 'student_resource', 'dataset', 'train', 'train_source1.tsv'), sep='\t', nrows=500).set_index('entity_id')
s2 = pd.read_csv(os.path.join(base_dir, 'student_resource', 'dataset', 'train', 'train_source2.tsv'), sep='\t', nrows=50000).set_index('entity_id')
s3 = pd.read_csv(os.path.join(base_dir, 'student_resource', 'dataset', 'train', 'train_source3.tsv'), sep='\t', nrows=50000).set_index('entity_id')

token_hits = 0
phonetic_hits = 0
address_hits = 0
tested = 0

for s1_id, c_id in sample_pairs:
    if s1_id not in s1.index: continue
    r2_table = s2 if c_id.startswith('S2') else s3
    if c_id not in r2_table.index: continue
    
    r1 = s1.loc[s1_id]
    r2 = r2_table.loc[c_id]
    tested += 1
    
    n1, n2 = r1['business_name'], r2['business_name']
    a1, a2 = r1['business_address'], r2['business_address']
    
    norm1, norm2 = normalize_name(n1), normalize_name(n2)
    tok1, tok2 = set(get_token_keys(norm1)), set(get_token_keys(norm2))
    ph1, ph2 = set(get_phonetic_keys(norm1)), set(get_phonetic_keys(norm2))
    
    z1, num1 = extract_address_keys(a1)
    z2, num2 = extract_address_keys(a2)
    addr_match = (z1 and z2 and z1 == z2 and num1 and num2 and num1 == num2)
    
    t_hit = bool(tok1 & tok2)
    p_hit = bool(ph1 & ph2)
    
    if t_hit: token_hits += 1
    if p_hit: phonetic_hits += 1
    if addr_match: address_hits += 1
    
    print(f"[{tested}]")
    print(f"  S1:   {n1} | {a1}")
    print(f"  S23:  {n2} | {a2}")
    print(f"  Token overlap: {t_hit} ({tok1 & tok2}) | Phonetic: {p_hit} ({ph1 & ph2}) | Addr: {addr_match}")
    print("-" * 70)

if tested > 0:
    print(f"\nSummary on {tested} testable pairs:")
    print(f"  Token hits:    {token_hits} ({token_hits/tested*100:.1f}%)")
    print(f"  Phonetic hits: {phonetic_hits} ({phonetic_hits/tested*100:.1f}%)")
    print(f"  Combined (Token OR Phonetic): {token_hits + phonetic_hits} hits")
else:
    print("\nNo sample candidate pairs matched within the first nrows loaded.")
