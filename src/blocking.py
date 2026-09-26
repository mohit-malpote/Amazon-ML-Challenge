#!/usr/bin/env python3
"""
Production Multi-Channel Blocking Engine for ML Challenge 2026.
Supports country-parallel execution:
  python src/blocking.py --country India
  python src/blocking.py --country US
  python src/blocking.py --country France
  python src/blocking.py --country ALL

Combines 5 independent channels:
1. Significant Token Blocking (stopwords & legal suffixes stripped)
2. Phonetic Blocking (Double Metaphone on first 2 significant words)
3. Address & Postal Code Blocking (exact PIN/Zip + street number)
4. Sorted Neighborhood Method (SNM window W=5 on normalized names)
5. Zero-Candidate Fallback Engine (guarantees NO S1 entity is left empty)
"""

import os
import sys
import re
import time
import argparse
import unicodedata
from collections import defaultdict
import pandas as pd
import jellyfish

# Regex patterns for normalization
noise_prefix_pattern = re.compile(r'^\s*(m/s|shri|sri|sree|smt|dr|mr|mrs|the|a|an)\b\s*', re.IGNORECASE)
legal_suffix_pattern = re.compile(r'\b(pvt\s+ltd|private\s+limited|llc|inc|corp|corporation|co|company|ltd|limited|llp|gmbh|sarl|sa)\b', re.IGNORECASE)
punct_pattern = re.compile(r'[^\w\s]')
zip_pattern = re.compile(r'\b\d{5,6}\b')
leading_num_pattern = re.compile(r'^\s*(\d+)')

STOPWORDS = {
    'the', 'a', 'an', 'and', 'of', 'in', 'at', 'by', 'for', 'with', 'on',
    'services', 'service', 'solutions', 'solution', 'store', 'stores',
    'shop', 'shops', 'center', 'centre', 'enterprises', 'enterprise',
    'associates', 'group', 'india', 'usa', 'france', 'international',
    'tech', 'technologies', 'technology', 'systems', 'consulting'
}

def normalize_name(name):
    if not name or not isinstance(name, str):
        return ""
    # Strip diacritics / accents (essential for French and Romanized Indian names)
    s = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8')
    s = s.lower().replace('&', ' and ')
    s = noise_prefix_pattern.sub('', s)
    s = legal_suffix_pattern.sub('', s)
    s = punct_pattern.sub(' ', s)
    return ' '.join(s.split())

def extract_address_keys(addr):
    if not addr or not isinstance(addr, str):
        return "", ""
    z_m = zip_pattern.search(addr)
    zip_code = z_m.group(0) if z_m else ""
    n_m = leading_num_pattern.search(addr)
    num = n_m.group(1) if n_m else ""
    return zip_code, num

def extract_first_addr_token(addr):
    if not addr or not isinstance(addr, str):
        return ""
    tokens = [t for t in re.findall(r'\b[a-zA-Z]{3,}\b', addr.lower()) if t not in {'near', 'opp', 'road', 'street', 'bldg', 'floor', 'colony'}]
    return tokens[0] if tokens else ""

def get_phonetic_keys(norm_name):
    tokens = [t for t in norm_name.split() if len(t) >= 3 and t not in STOPWORDS]
    keys = []
    for t in tokens[:2]:
        try:
            m = jellyfish.metaphone(t)
            if m:
                keys.append(m)
        except Exception:
            pass
    return keys

def get_token_keys(norm_name):
    return [t for t in norm_name.split() if len(t) >= 3 and t not in STOPWORDS][:3]

def run_blocking_for_country(country, test_dir, output_dir, max_cands_per_s1=50):
    t0 = time.time()
    print("=" * 80)
    print(f" STARTING MULTI-CHANNEL BLOCKING FOR COUNTRY: {country.upper()}")
    print("=" * 80)

    # 1. Load Source 1, 2, 3 filtered by Country
    print(f"\n[1/5] Loading records for {country} from test sources...")
    s1_records = []
    with open(os.path.join(test_dir, 'test_source1.tsv'), 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) >= 4 and p[3] == country:
                s1_records.append((p[0], p[1], p[2]))
    print(f"   -> Loaded {len(s1_records):,} Source 1 records.")

    cand_records = []
    for src in ['test_source2.tsv', 'test_source3.tsv']:
        c_loaded = 0
        with open(os.path.join(test_dir, src), 'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                p = line.rstrip('\n').split('\t')
                if len(p) >= 4 and p[3] == country:
                    cand_records.append((p[0], p[1], p[2]))
                    c_loaded += 1
        print(f"   -> Loaded {c_loaded:,} candidate records from {src}.")

    total_cands = len(cand_records)
    print(f"   -> Total candidates pool (S2+S3) for {country}: {total_cands:,}")

    # 2. Build Inverted Index from Candidates (S2 & S3)
    print("\n[2/5] Building Multi-Channel Inverted Index over S2/S3 candidate entities...")
    t_idx = time.time()
    token_index = defaultdict(list)
    phonetic_index = defaultdict(list)
    zip_num_index = defaultdict(list)
    zip_token_index = defaultdict(list)

    # Sorted Neighborhood candidate list: (norm_name, entity_id)
    cand_snm_list = []

    for eid, nm, ad in cand_records:
        norm_n = normalize_name(nm)
        cand_snm_list.append((norm_n, eid))

        # Channel 1: Significant Tokens
        for tok in get_token_keys(norm_n):
            token_index[tok].append(eid)

        # Channel 2: Phonetic Double Metaphone
        for ph in get_phonetic_keys(norm_n):
            phonetic_index[ph].append(eid)

        # Channel 3: Address & Postal Keys
        z, num = extract_address_keys(ad)
        if z:
            if num:
                zip_num_index[f"{z}_{num}"].append(eid)
            first_ad_tok = extract_first_addr_token(ad)
            if first_ad_tok:
                zip_token_index[f"{z}_{first_ad_tok}"].append(eid)

    print(f"   -> Inverted indexes built in {time.time()-t_idx:.2f}s.")
    print(f"      - Token index size:     {len(token_index):,} keys")
    print(f"      - Phonetic index size:  {len(phonetic_index):,} keys")
    print(f"      - Zip-Num index size:   {len(zip_num_index):,} keys")
    print(f"      - Zip-Token index size: {len(zip_token_index):,} keys")

    # 3. Channel 4: Sorted Neighborhood Index
    print("\n[3/5] Setting up Sorted Neighborhood lookup for candidates...")
    cand_snm_list.sort(key=lambda x: x[0])
    cand_snm_names = [x[0] for x in cand_snm_list]
    cand_snm_ids = [x[1] for x in cand_snm_list]

    import bisect

    # 4. Query Inverted Indexes for every Source 1 Entity
    print(f"\n[4/5] Generating candidate pairs for {len(s1_records):,} Source 1 entities...")
    t_match = time.time()
    s1_candidates_map = {}
    zero_cand_count = 0
    total_pairs_generated = 0

    for idx, (s1_id, nm, ad) in enumerate(s1_records):
        norm_n = normalize_name(nm)
        cand_set = set()

        # Query Channel 1: Token
        for tok in get_token_keys(norm_n):
            c_list = token_index.get(tok)
            if c_list and len(c_list) < 250: # Avoid high-frequency exploding blocks
                cand_set.update(c_list)

        # Query Channel 2: Phonetic
        for ph in get_phonetic_keys(norm_n):
            c_list = phonetic_index.get(ph)
            if c_list and len(c_list) < 250:
                cand_set.update(c_list)

        # Query Channel 3: Address & Postal Keys
        z, num = extract_address_keys(ad)
        if z:
            if num:
                c_list = zip_num_index.get(f"{z}_{num}")
                if c_list and len(c_list) < 150:
                    cand_set.update(c_list)
            first_ad_tok = extract_first_addr_token(ad)
            if first_ad_tok:
                c_list = zip_token_index.get(f"{z}_{first_ad_tok}")
                if c_list and len(c_list) < 150:
                    cand_set.update(c_list)

        # Query Channel 4: Sorted Neighborhood (Window W=5 around insertion point)
        if norm_n:
            pos = bisect.bisect_left(cand_snm_names, norm_n)
            start_pos = max(0, pos - 3)
            end_pos = min(len(cand_snm_names), pos + 3)
            for p_idx in range(start_pos, end_pos):
                cand_set.add(cand_snm_ids[p_idx])

        # Channel 5: Zero-Candidate Fallback Engine
        # If an entity still has 0 candidates, fallback to prefix or postal code
        if not cand_set:
            zero_cand_count += 1
            # Fallback A: 3-letter name prefix
            if len(norm_n) >= 3:
                pref = norm_n[:3]
                pos = bisect.bisect_left(cand_snm_names, pref)
                cand_set.update(cand_snm_ids[pos:min(len(cand_snm_names), pos + 5)])

            # Fallback B: If still empty, use nearest sorted neighbors
            if not cand_set and cand_snm_ids:
                cand_set.add(cand_snm_ids[min(pos, len(cand_snm_ids)-1)])

        # Cap maximum candidates per S1 entity to avoid explosion
        if len(cand_set) > max_cands_per_s1:
            cand_list = list(cand_set)[:max_cands_per_s1]
        else:
            cand_list = list(cand_set)

        total_pairs_generated += len(cand_list)
        s1_candidates_map[s1_id] = cand_list

        if (idx + 1) % 100000 == 0 or (idx + 1) == len(s1_records):
            elapsed = time.time() - t_match
            print(f"   -> Processed {idx+1:,} / {len(s1_records):,} S1 entities "
                  f"({(idx+1)/len(s1_records)*100:.1f}%) | Pairs: {total_pairs_generated:,} | Elapsed: {elapsed:.1f}s")

    print(f"\n   -> Candidate generation completed in {time.time()-t_match:.2f}s.")
    print(f"   -> Total candidate pairs generated: {total_pairs_generated:,}")
    print(f"   -> Average candidates per S1:       {total_pairs_generated/len(s1_records):.2f}")
    print(f"   -> Fallback rescued entities:       {zero_cand_count:,}")

    # 5. Export Country Output File
    out_file = os.path.join(output_dir, f"candidate_pairs_{country}.tsv")
    print(f"\n[5/5] Exporting candidate pairs to {out_file}...")
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id, _, _ in s1_records:
            cands = s1_candidates_map.get(s1_id, [])
            f.write(f"{s1_id}\t{','.join(cands)}\n")

    print(f" SUCCESS: {out_file} generated successfully in {(time.time()-t0)/60:.2f} minutes.\n")
    return out_file

def main():
    parser = argparse.ArgumentParser(description="Multi-Channel Blocking for ML Challenge 2026")
    parser.add_argument('--country', type=str, default='ALL', choices=['India', 'US', 'France', 'ALL'],
                        help="Country to process: 'India', 'US', 'France', or 'ALL'")
    parser.add_argument('--max-cands', type=int, default=50,
                        help="Max candidate pairs allowed per S1 entity (default: 50)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_dir = os.path.join(base_dir, 'student_resource', 'dataset', 'test')
    output_dir = os.path.join(base_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    if args.country == 'ALL':
        countries = ['India', 'US', 'France']
    else:
        countries = [args.country]

    for c in countries:
        run_blocking_for_country(c, test_dir, output_dir, max_cands_per_s1=args.max_cands)

if __name__ == '__main__':
    main()
