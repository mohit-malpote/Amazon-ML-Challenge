#!/usr/bin/env python3
"""
Production Multi-Channel High-Recall Blocking Engine for ML Challenge 2026.
Supports country-parallel execution:
  python src/blocking.py --country India
  python src/blocking.py --country US
  python src/blocking.py --country France
  python src/blocking.py --country ALL

Also supports local validation benchmarking:
  python src/blocking.py --mode val --country ALL

Combines 6 complementary blocking channels:
1. Significant Token Blocking (stopwords & legal suffixes stripped, up to 4 tokens)
2. Compact Name Prefix Blocking (strips spaces to catch e.g. agracare.com vs Agra Care)
3. Phonetic Blocking (Double Metaphone on primary significant tokens)
4. Address & Postal Code Blocking (exact PIN/Zip + building/house number)
5. Building + Locality Token Blocking (recovers matches when PIN code is missing)
6. Sorted Neighborhood Method (SNM window W=5 on normalized names)
7. Score-based Prioritized Selection: Multi-channel hits ranked first via heapq.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True, encoding='utf-8', errors='replace')

import os
import re
import time
import argparse
import unicodedata
import bisect
import heapq
from collections import defaultdict
import pandas as pd
import jellyfish

# Regex patterns for normalization across US, India, and France
noise_prefix_pattern = re.compile(r'^\s*(m/s|shri|sri|sree|smt|dr|mr|mrs|the|a|an|ste|societe|ets|cie|cabinet|atelier)\b\s*', re.IGNORECASE)
legal_suffix_pattern = re.compile(
    r'\b(pvt\s+ltd|private\s+limited|llc|inc|corp|corporation|co|company|ltd|limited|llp|gmbh|sarl|sas|sasu|eurl|sci|snc|sa|ei|fils|societe|ste|association|ets|etablissements)\b',
    re.IGNORECASE
)
punct_pattern = re.compile(r'[^\w\s]')
zip_pattern = re.compile(r'\b\d{5,6}\b')
building_pattern = re.compile(r'\b(?:h\.?no\.?|bldg|plot|flat|room|shop|survey|b-)?\s*([a-zA-Z]?\d+[\w-]*)\b', re.IGNORECASE)
num_pattern = re.compile(r'\b\d+\b')
domain_regex = re.compile(r'\.(?:com|in|org|net|co\.in|fr|io)\b', re.IGNORECASE)
phone_regex = re.compile(r'[\s\-]+\d{10}\b')
symbol_noise_regex = re.compile(r'^[^\w]+|[^\w]+$')

# Smart Building/Unit Number extraction patterns across US, India, France
bldg_prefix_pattern = re.compile(
    r'\b(?:plot\s*no\.?|house\s*no\.?|h\.?\s*no\.?|flat\s*no\.?|door\s*no\.?|shop\s*no\.?|no\.?|kh\.?\s*no\.?|unit|suite|ste|apt|bldg|building|room|rn|hn)\s*[:#\-\/]?\s*([0-9]+[a-zA-Z]?)\b',
    re.IGNORECASE
)
bldg_adj_pattern = re.compile(
    r'\b(\d+)\s*(?:bis|ter|quater)?\s*(?:st|street|ave|avenue|rd|road|lane|dr|drive|ter|terrace|rue|bd|boulevard|blvd|chemin|impasse|allee|place|pl|cours|route|rte|quai|passage|square)\b',
    re.IGNORECASE
)
bldg_post_pattern = re.compile(
    r'\b(?:st|street|ave|avenue|rd|road|lane|dr|drive|ter|terrace|rue|bd|boulevard|blvd|chemin|impasse|allee|place|pl|cours|route|rte|quai|passage|square)\s+(\d+)\b',
    re.IGNORECASE
)
leading_num_pattern = re.compile(r'^\s*(\d+)\b')

# Comprehensive Brahmic Romanizer mapping covering all Indic scripts
INDIC_VOWELS = {
    'A': 'a', 'AA': 'aa', 'I': 'i', 'II': 'ee', 'U': 'u', 'UU': 'oo',
    'VOCALIC R': 'ri', 'VOCALIC RR': 'ri', 'E': 'e', 'AI': 'ai', 'O': 'o', 'AU': 'au',
    'CANDRA E': 'e', 'CANDRA O': 'o', 'SHORT E': 'e', 'SHORT O': 'o'
}

INDIC_CONSONANTS = {
    'KA': 'k', 'KHA': 'kh', 'GA': 'g', 'GHA': 'gh', 'NGA': 'ng',
    'CA': 'ch', 'CHA': 'chh', 'JA': 'j', 'JHA': 'jh', 'NYA': 'ny',
    'TTA': 't', 'TTHA': 'th', 'DDA': 'd', 'DDHA': 'dh', 'NNA': 'n',
    'TA': 't', 'THA': 'th', 'DA': 'd', 'DHA': 'dh', 'NA': 'n',
    'PA': 'p', 'PHA': 'ph', 'BA': 'b', 'BHA': 'bh', 'MA': 'm',
    'YA': 'y', 'RA': 'r', 'LA': 'l', 'LLA': 'l', 'VA': 'v', 'WA': 'w',
    'SHA': 'sh', 'SSA': 'sh', 'SA': 's', 'HA': 'h'
}

def romanize_indic(text):
    if not text:
        return ""
    out = []
    for ch in text:
        if ord(ch) < 128:
            out.append(ch)
            continue
        try:
            uname = unicodedata.name(ch)
        except ValueError:
            continue
        if any(uname.startswith(script) for script in ['DEVANAGARI', 'BENGALI', 'GURMUKHI', 'GUJARATI', 'ORIYA', 'TAMIL', 'TELUGU', 'KANNADA', 'MALAYALAM']):
            if 'VIRAMA' in uname:
                continue
            if 'ANUSVARA' in uname or 'CANDRABINDU' in uname:
                out.append('n')
                continue
            if 'VOWEL SIGN ' in uname:
                v = uname.split('VOWEL SIGN ')[1].strip()
                if v in INDIC_VOWELS:
                    out.append(INDIC_VOWELS[v])
                    continue
            if 'LETTER ' in uname:
                l = uname.split('LETTER ')[1].strip()
                if l in INDIC_VOWELS:
                    out.append(INDIC_VOWELS[l])
                    continue
                if l in INDIC_CONSONANTS:
                    out.append(INDIC_CONSONANTS[l])
                    continue
            out.append(' ')
        else:
            out.append(ch)
    return ''.join(out)

STOPWORDS = {
    'the', 'a', 'an', 'and', 'of', 'in', 'at', 'by', 'for', 'with', 'on',
    'services', 'service', 'solutions', 'solution', 'store', 'stores',
    'shop', 'shops', 'center', 'centre', 'enterprises', 'enterprise',
    'associates', 'group', 'india', 'usa', 'france', 'international',
    'tech', 'technologies', 'technology', 'systems', 'consulting',
    'pvt', 'ltd', 'limited', 'private', 'llc', 'inc', 'corp', 'co',
    'le', 'la', 'les', 'de', 'du', 'des', 'ste', 'societe', 'ets'
}

ADDR_STOPWORDS = {
    'near', 'opp', 'opposite', 'behind', 'road', 'rd', 'street', 'st', 'lane',
    'floor', 'flr', 'building', 'bldg', 'nagar', 'colony', 'vihar', 'plot',
    'flat', 'shop', 'no', 'hno', 'house', 'main', 'cross', 'post', 'dist',
    'india', 'france', 'usa', 'state', 'city', 'town', 'village', 'avenue', 'ave',
    'rue', 'bd', 'boulevard', 'blvd', 'chemin', 'impasse', 'allee', 'place', 'pl',
    'cours', 'route', 'rte', 'quai', 'passage', 'square', 'cedex'
}

def normalize_name(name):
    if not name or not isinstance(name, str):
        return ""
    # 1. Romanize Indic scripts before ascii normalization
    name_rom = romanize_indic(name)
    # 2. Strip diacritics / accents (essential for French and Romanized Indian names)
    s = unicodedata.normalize('NFKD', name_rom).encode('ascii', 'ignore').decode('utf-8')
    s = s.lower().replace('&', ' and ')
    s = domain_regex.sub('', s)
    s = phone_regex.sub('', s)
    s = symbol_noise_regex.sub(' ', s)
    s = noise_prefix_pattern.sub('', s)
    s = legal_suffix_pattern.sub('', s)
    s = punct_pattern.sub(' ', s)
    return ' '.join(s.split())

def extract_address_keys(addr):
    if not addr or not isinstance(addr, str):
        return "", ""
    z_m = zip_pattern.search(addr)
    zip_code = z_m.group(0) if z_m else ""
    ad = addr.lower()
    num = ""
    m = bldg_prefix_pattern.search(ad) or bldg_adj_pattern.search(ad) or leading_num_pattern.search(ad)
    if m:
        num = m.group(1).lstrip('0')
        if len(num) in [5, 6] and num.isdigit():
            num = ""
    return zip_code, num

def extract_addr_features(addr):
    if not addr or not isinstance(addr, str):
        return "", "", [], []
    z_m = zip_pattern.search(addr)
    z = z_m.group(0) if z_m else ""
    
    ad = addr.lower()
    b = ""
    m_b = bldg_prefix_pattern.search(ad) or bldg_adj_pattern.search(ad) or bldg_post_pattern.search(ad) or leading_num_pattern.search(ad)
    if m_b:
        num_cand = m_b.group(1).lstrip('0')
        if not (len(num_cand) in [5, 6] and num_cand.isdigit()):
            b = num_cand
    if not b:
        b_m = building_pattern.search(addr)
        b = b_m.group(1).lower().lstrip('0') if b_m else ""
    
    clean_ad = punct_pattern.sub(' ', addr.lower())
    ad_tokens = [t for t in clean_ad.split() if len(t) >= 4 and t not in ADDR_STOPWORDS and not t.isdigit()]
    ad_nums = [m.lstrip('0') for m in num_pattern.findall(addr) if m.lstrip('0') and len(m.lstrip('0')) <= 6]
    return z, b, ad_tokens[:4], ad_nums[:2]

def extract_first_addr_token(addr):
    if not addr or not isinstance(addr, str):
        return ""
    tokens = [t for t in re.findall(r'\b[a-zA-Z]{3,}\b', addr.lower()) if t not in ADDR_STOPWORDS]
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
    return [t for t in norm_name.split() if len(t) >= 3 and t not in STOPWORDS][:4]

def run_blocking_for_country(country, s1_file, cand_files, output_file, max_cands_per_s1=60):
    t0 = time.time()
    print("=" * 80, flush=True)
    print(f" STARTING MULTI-CHANNEL BLOCKING FOR COUNTRY: {country.upper()}", flush=True)
    print(f" Source 1 File: {s1_file}", flush=True)
    print(f" Target Output: {output_file}", flush=True)
    print("=" * 80, flush=True)

    # 1. Load Source 1 filtered by Country
    print(f"\n[1/5] Loading records for {country} from Source 1...", flush=True)
    s1_records = []
    with open(s1_file, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split('\t')
            eid = p[0]
            nm = p[1] if len(p) > 1 else ''
            ad = p[2] if len(p) > 2 else ''
            ct = p[3] if len(p) > 3 else ''
            if country == 'ALL' or ct == country:
                s1_records.append((eid, nm, ad))
    print(f"   -> Loaded {len(s1_records):,} Source 1 records.", flush=True)

    if not s1_records:
        print(f"   -> No records found for country {country}. Skipping.", flush=True)
        return output_file

    # 2. Load candidate records (Source 2 & 3)
    cand_records = []
    for src in cand_files:
        c_loaded = 0
        if not os.path.exists(src):
            continue
        with open(src, 'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                p = line.rstrip('\n').split('\t')
                eid = p[0]
                nm = p[1] if len(p) > 1 else ''
                ad = p[2] if len(p) > 2 else ''
                ct = p[3] if len(p) > 3 else ''
                if country == 'ALL' or ct == country:
                    cand_records.append((eid, nm, ad))
                    c_loaded += 1
        print(f"   -> Loaded {c_loaded:,} candidate records from {os.path.basename(src)}.", flush=True)

    total_cands = len(cand_records)
    print(f"   -> Total candidates pool (S2+S3) for {country}: {total_cands:,}", flush=True)

    # 3. Build Multi-Channel Inverted Indexes
    print("\n[2/5] Building Multi-Channel Inverted Indexes over candidates...", flush=True)
    t_idx = time.time()
    token_index = defaultdict(list)
    bigram_index = defaultdict(list)
    compact_index = defaultdict(list)
    phone_index = defaultdict(list)
    zip_bldg_index = defaultdict(list)
    zip_token_index = defaultdict(list)
    bldg_token_index = defaultdict(list)
    addr_num_tok_index = defaultdict(list)
    addr_pair_index = defaultdict(list)
    cand_snm_list = []

    for eid, nm, ad in cand_records:
        norm = normalize_name(nm)
        compact = norm.replace(' ', '')
        cand_snm_list.append((norm, eid))

        # Channel 1: Significant Tokens & Bigrams
        toks = [t for t in norm.split() if len(t) >= 3 and t not in STOPWORDS]
        for t in toks[:4]:
            token_index[t].append(eid)
        if len(toks) >= 2:
            bigram_index[f"{toks[0]}_{toks[1]}"].append(eid)

        # Channel 2: Compact Name Prefix (catches agracare.com vs Agra Care)
        if len(compact) >= 4:
            compact_index[compact[:8]].append(eid)

        # Channel 3: Phonetic (Double Metaphone)
        for t in toks[:2]:
            try:
                m = jellyfish.metaphone(t)
                if m:
                    phone_index[m].append(eid)
            except Exception:
                pass

        # Channel 4 & 5: Address and Building Features
        z, bldg, ad_toks, ad_nums = extract_addr_features(ad)
        if z and bldg:
            zip_bldg_index[f"{z}_{bldg}"].append(eid)
        if z and ad_toks:
            for at in ad_toks[:2]:
                zip_token_index[f"{z}_{at}"].append(eid)
        if bldg and ad_toks:
            bldg_token_index[f"{bldg}_{ad_toks[0]}"].append(eid)

        # Channel 7: High-Precision Locality + Number Index
        for num in ad_nums[:2]:
            for at in ad_toks[:3]:
                addr_num_tok_index[f"{num}_{at}"].append(eid)
        if not ad_nums and len(ad_toks) >= 2:
            addr_pair_index[f"{ad_toks[0]}_{ad_toks[1]}"].append(eid)

    print(f"   -> Inverted indexes built in {time.time()-t_idx:.2f}s.", flush=True)
    print(f"      - Token index size:        {len(token_index):,} keys", flush=True)
    print(f"      - Bigram index size:       {len(bigram_index):,} keys", flush=True)
    print(f"      - Compact prefix index:    {len(compact_index):,} keys", flush=True)
    print(f"      - Phonetic index size:     {len(phone_index):,} keys", flush=True)
    print(f"      - Zip-Bldg index size:     {len(zip_bldg_index):,} keys", flush=True)
    print(f"      - Zip-Token index size:    {len(zip_token_index):,} keys", flush=True)
    print(f"      - Bldg-Token index size:   {len(bldg_token_index):,} keys", flush=True)
    print(f"      - Num-Token index size:    {len(addr_num_tok_index):,} keys", flush=True)

    # 4. Set up Sorted Neighborhood Method
    print("\n[3/5] Setting up Sorted Neighborhood lookup...", flush=True)
    cand_snm_list.sort(key=lambda x: x[0])
    cand_snm_names = [x[0] for x in cand_snm_list]
    cand_snm_ids = [x[1] for x in cand_snm_list]

    # 5. Query Multi-Channel Indexes with Priority Scoring
    print(f"\n[4/5] Generating candidate pairs for {len(s1_records):,} Source 1 entities...", flush=True)
    t_match = time.time()
    s1_candidates_map = {}
    zero_cand_count = 0
    total_pairs_generated = 0

    for idx, (s1_id, nm, ad) in enumerate(s1_records):
        norm = normalize_name(nm)
        compact = norm.replace(' ', '')
        scores = {}

        # Channel 1: Significant Tokens (allow blocks up to 3000)
        toks = [t for t in norm.split() if len(t) >= 3 and t not in STOPWORDS]
        for t in toks[:4]:
            cl = token_index.get(t)
            if cl and len(cl) < 3000:
                w = 4.0 if len(cl) < 50 else (2.5 if len(cl) < 250 else (1.2 if len(cl) < 1000 else 0.6))
                for c in cl:
                    scores[c] = scores.get(c, 0.0) + w

        # Channel 1B: Bigrams (High Precision Anchor)
        if len(toks) >= 2:
            cl = bigram_index.get(f"{toks[0]}_{toks[1]}")
            if cl and len(cl) < 400:
                for c in cl:
                    scores[c] = scores.get(c, 0.0) + 6.0

        # Channel 2: Compact Name
        if len(compact) >= 4:
            cl = compact_index.get(compact[:8])
            if cl and len(cl) < 500:
                for c in cl:
                    scores[c] = scores.get(c, 0.0) + 4.0

        # Channel 3: Phonetic
        for t in toks[:2]:
            try:
                m = jellyfish.metaphone(t)
                cl = phone_index.get(m)
                if cl and len(cl) < 600:
                    for c in cl:
                        scores[c] = scores.get(c, 0.0) + 1.5
            except Exception:
                pass

        # Channel 4 & 5: Address Features
        z, bldg, ad_toks, ad_nums = extract_addr_features(ad)
        if z and bldg:
            cl = zip_bldg_index.get(f"{z}_{bldg}")
            if cl and len(cl) < 500:
                for c in cl:
                    scores[c] = scores.get(c, 0.0) + 5.0
        if z and ad_toks:
            for at in ad_toks[:2]:
                cl = zip_token_index.get(f"{z}_{at}")
                if cl and len(cl) < 500:
                    for c in cl:
                        scores[c] = scores.get(c, 0.0) + 3.0
        if bldg and ad_toks:
            cl = bldg_token_index.get(f"{bldg}_{ad_toks[0]}")
            if cl and len(cl) < 400:
                for c in cl:
                    scores[c] = scores.get(c, 0.0) + 4.0

        # Channel 7: Locality Number + Token Query
        for num in ad_nums[:2]:
            for at in ad_toks[:3]:
                cl = addr_num_tok_index.get(f"{num}_{at}")
                if cl and len(cl) < 400:
                    for c in cl:
                        scores[c] = scores.get(c, 0.0) + 5.0
        if not ad_nums and len(ad_toks) >= 2:
            cl = addr_pair_index.get(f"{ad_toks[0]}_{ad_toks[1]}")
            if cl and len(cl) < 300:
                for c in cl:
                    scores[c] = scores.get(c, 0.0) + 4.0

        # Channel 6: Sorted Neighborhood Method (W=9)
        if norm:
            pos = bisect.bisect_left(cand_snm_names, norm)
            start_pos = max(0, pos - 8)
            end_pos = min(len(cand_snm_names), pos + 8)
            for p_idx in range(start_pos, end_pos):
                cid = cand_snm_ids[p_idx]
                scores[cid] = scores.get(cid, 0.0) + 1.0

        # Fallback if zero candidates
        if not scores and cand_snm_ids:
            zero_cand_count += 1
            if len(norm) >= 3:
                pos = bisect.bisect_left(cand_snm_names, norm[:3])
                for p_idx in range(pos, min(len(cand_snm_names), pos + 5)):
                    cid = cand_snm_ids[p_idx]
                    scores[cid] = scores.get(cid, 0.0) + 0.5
            if not scores:
                cid = cand_snm_ids[min(pos, len(cand_snm_ids) - 1)]
                scores[cid] = 0.1

        # Priority selection: heap top max_cands_per_s1
        if len(scores) <= max_cands_per_s1:
            top_candidates = list(scores.keys())
        else:
            top_candidates = heapq.nlargest(max_cands_per_s1, scores, key=scores.get)

        s1_candidates_map[s1_id] = top_candidates
        total_pairs_generated += len(top_candidates)

        if (idx + 1) % 25000 == 0 or (idx + 1) == len(s1_records):
            elapsed = time.time() - t_match
            print(f"   -> Processed {idx+1:,} / {len(s1_records):,} S1 entities "
                  f"({(idx+1)/len(s1_records)*100:.1f}%) | Pairs: {total_pairs_generated:,} | Elapsed: {elapsed:.1f}s", flush=True)

    print(f"\n   -> Blocking finished in {time.time()-t_match:.2f}s.", flush=True)
    print(f"   -> Total candidate pairs:     {total_pairs_generated:,}", flush=True)
    print(f"   -> Avg candidates per S1:     {total_pairs_generated/len(s1_records):.2f}", flush=True)
    print(f"   -> Fallback rescued entities: {zero_cand_count:,}", flush=True)

    # 6. Export Output File
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    print(f"\n[5/5] Exporting candidate pairs to {output_file}...", flush=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id, _, _ in s1_records:
            cands = s1_candidates_map.get(s1_id, [])
            f.write(f"{s1_id}\t{','.join(cands)}\n")

    print(f" SUCCESS: {output_file} generated in {(time.time()-t0)/60:.2f} minutes.\n", flush=True)
    return output_file

def merge_country_files(input_files, output_file):
    print(f"Merging country candidate files into {output_file}...", flush=True)
    with open(output_file, 'w', encoding='utf-8') as fout:
        fout.write("source1_entity_id\tcandidate_entity_ids\n")
        for fpath in input_files:
            if not os.path.exists(fpath):
                continue
            with open(fpath, 'r', encoding='utf-8') as fin:
                next(fin) # skip header
                for line in fin:
                    fout.write(line)
    print(f"Merged output saved to {output_file}", flush=True)

def main():
    parser = argparse.ArgumentParser(description="Multi-Channel Blocking for ML Challenge 2026")
    parser.add_argument('--mode', type=str, default='test', choices=['test', 'val'],
                        help="Execution mode: 'test' for test dataset, 'val' for local validation benchmark")
    parser.add_argument('--country', type=str, default='ALL', choices=['India', 'US', 'France', 'ALL'],
                        help="Country to process: 'India', 'US', 'France', or 'ALL'")
    parser.add_argument('--max-cands', type=int, default=60,
                        help="Max candidate pairs allowed per S1 entity (default: 60)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    if args.mode == 'val':
        # Local validation benchmark on data/processed/val_s1_sample.tsv
        val_s1_file = os.path.join(base_dir, 'data', 'processed', 'val_s1_sample.tsv')
        train_cand_files = [
            os.path.join(base_dir, 'student_resource', 'dataset', 'train', 'train_source2.tsv'),
            os.path.join(base_dir, 'student_resource', 'dataset', 'train', 'train_source3.tsv')
        ]
        
        countries = ['India', 'US'] if args.country == 'ALL' else [args.country]
        country_out_files = []
        for c in countries:
            c_out = os.path.join(base_dir, 'data', 'processed', f"candidate_pairs_val_{c}.tsv")
            run_blocking_for_country(c, val_s1_file, train_cand_files, c_out, max_cands_per_s1=args.max_cands)
            country_out_files.append(c_out)
            
        final_val_cands = os.path.join(base_dir, 'data', 'processed', 'candidate_pairs_val.tsv')
        merge_country_files(country_out_files, final_val_cands)

    else:
        # Production test set blocking
        test_dir = os.path.join(base_dir, 'student_resource', 'dataset', 'test')
        s1_file = os.path.join(test_dir, 'test_source1.tsv')
        cand_files = [
            os.path.join(test_dir, 'test_source2.tsv'),
            os.path.join(test_dir, 'test_source3.tsv')
        ]
        countries = ['France', 'US', 'India'] if args.country == 'ALL' else [args.country]
        country_out_files = []
        for c in countries:
            c_out = os.path.join(output_dir, f"candidate_pairs_{c}.tsv")
            run_blocking_for_country(c, s1_file, cand_files, c_out, max_cands_per_s1=args.max_cands)
            country_out_files.append(c_out)

        all_country_files = [os.path.join(output_dir, f"candidate_pairs_{c}.tsv") for c in ['France', 'US', 'India']]
        if all(os.path.exists(f) for f in all_country_files):
            import shutil
            final_test_cands = os.path.join(output_dir, 'candidate_pairs_test.tsv')
            merge_country_files(all_country_files, final_test_cands)
            canonical_cands = os.path.join(output_dir, 'candidate_pairs.tsv')
            print(f"Syncing canonical candidate file: {canonical_cands}...", flush=True)
            shutil.copyfile(final_test_cands, canonical_cands)
            print(f"Canonical candidates ready: {canonical_cands}", flush=True)
        elif args.country == 'ALL':
            import shutil
            final_test_cands = os.path.join(output_dir, 'candidate_pairs_test.tsv')
            merge_country_files(country_out_files, final_test_cands)
            canonical_cands = os.path.join(output_dir, 'candidate_pairs.tsv')
            print(f"Syncing canonical candidate file: {canonical_cands}...", flush=True)
            shutil.copyfile(final_test_cands, canonical_cands)
            print(f"Canonical candidates ready: {canonical_cands}", flush=True)

if __name__ == '__main__':
    main()
