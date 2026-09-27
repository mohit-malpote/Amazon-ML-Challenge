import sys, os, time, re, unicodedata, bisect, heapq
from collections import defaultdict
sys.stdout.reconfigure(encoding='utf-8')

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(base_dir, 'src'))

from blocking import (
    noise_prefix_pattern, legal_suffix_pattern, punct_pattern, zip_pattern,
    building_pattern, leading_num_pattern, STOPWORDS, ADDR_STOPWORDS
)

# Indic Romanizer
INDIC_MAP = {
    'A': 'a', 'AA': 'aa', 'I': 'i', 'II': 'ee', 'U': 'u', 'UU': 'oo',
    'R': 'r', 'RR': 'ri', 'E': 'e', 'AI': 'ai', 'O': 'o', 'AU': 'au',
    'KA': 'k', 'KHA': 'kh', 'GA': 'g', 'GHA': 'gh', 'NGA': 'ng',
    'CA': 'ch', 'CHA': 'chh', 'JA': 'j', 'JHA': 'jh', 'NYA': 'ny',
    'TTA': 't', 'TTHA': 'th', 'DDA': 'd', 'DDHA': 'dh', 'NNA': 'n',
    'TA': 't', 'THA': 'th', 'DA': 'd', 'DHA': 'dh', 'NA': 'n',
    'PA': 'p', 'PHA': 'ph', 'BA': 'b', 'BHA': 'bh', 'MA': 'm',
    'YA': 'y', 'RA': 'r', 'LA': 'l', 'LLA': 'l', 'VA': 'v', 'WA': 'w',
    'SHA': 'sh', 'SSA': 'sh', 'SA': 's', 'HA': 'h',
    'SIGN VIRAMA': '', 'VIRAMA': '', 'SIGN ANUSVARA': 'n', 'SIGN CANDRABINDU': 'n'
}

num_pattern = re.compile(r'\b\d+\b')

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
        parts = uname.split()
        if len(parts) >= 3 and parts[0] in ['DEVANAGARI', 'BENGALI', 'GURMUKHI', 'GUJARATI', 'ORIYA', 'TAMIL', 'TELUGU', 'KANNADA', 'MALAYALAM']:
            key = ' '.join(parts[2:])
            if key in INDIC_MAP:
                out.append(INDIC_MAP[key])
            else:
                out.append(' ')
        else:
            out.append(ch)
    return ''.join(out)

import jellyfish

def normalize_name_upgraded(name):
    if not name or not isinstance(name, str):
        return ""
    name_rom = romanize_indic(name)
    s = unicodedata.normalize('NFKD', name_rom).encode('ascii', 'ignore').decode('utf-8')
    s = s.lower().replace('&', ' and ')
    s = noise_prefix_pattern.sub('', s)
    s = legal_suffix_pattern.sub('', s)
    s = punct_pattern.sub(' ', s)
    return ' '.join(s.split())

def extract_addr_features_upgraded(addr):
    if not addr or not isinstance(addr, str):
        return "", "", [], []
    z_m = zip_pattern.search(addr)
    z = z_m.group(0) if z_m else ""
    b_m = building_pattern.search(addr)
    b = b_m.group(1).lower().lstrip('0') if b_m else ""
    
    clean_ad = punct_pattern.sub(' ', addr.lower())
    ad_tokens = [t for t in clean_ad.split() if len(t) >= 4 and t not in ADDR_STOPWORDS and not t.isdigit()]
    ad_nums = [m.lstrip('0') for m in num_pattern.findall(addr) if m.lstrip('0') and len(m.lstrip('0')) <= 6]
    return z, b, ad_tokens[:4], ad_nums[:2]

def run_test():
    print("=" * 80)
    print(" TESTING UPGRADED BLOCKING ON 50,000 S1 VALIDATION SET")
    print("=" * 80)
    
    val_s1_file = os.path.join(base_dir, 'data', 'processed', 'val_s1_sample.tsv')
    val_gt_file = os.path.join(base_dir, 'data', 'processed', 'val_ground_truth.tsv')
    train_dir = os.path.join(base_dir, 'student_resource', 'dataset', 'train')
    cand_files = [
        os.path.join(train_dir, 'train_source2.tsv'),
        os.path.join(train_dir, 'train_source3.tsv')
    ]

    # 1. Load GT pairs
    print("[1/4] Loading Ground Truth...")
    gt_pairs = set()
    with open(val_gt_file, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split('\t')
            s1 = p[0]
            if len(p) > 1 and p[1].strip():
                for m in p[1].split(','):
                    gt_pairs.add((s1, m.strip()))
    print(f"   -> Total Ground Truth Match Pairs: {len(gt_pairs):,}")

    # 2. Load S1 validation entities (first 10,000 for fast benchmarking)
    print("[2/4] Loading S1 validation records (10,000 sample for speed)...")
    s1_records = []
    with open(val_s1_file, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split('\t')
            s1_records.append((p[0], p[1] if len(p)>1 else '', p[2] if len(p)>2 else '', p[3] if len(p)>3 else ''))
            if len(s1_records) >= 10000:
                break
    s1_ids_tested = set(r[0] for r in s1_records)
    gt_tested = set(p for p in gt_pairs if p[0] in s1_ids_tested)
    print(f"   -> Testing on {len(s1_records):,} S1 entities ({len(gt_tested):,} GT matches).")

    # 3. Load Candidate Records from candidate pool
    print("[3/4] Loading candidates and building upgraded multi-channel indexes...")
    t0 = time.time()
    
    # We load candidates for US & India (where our 10k entities are)
    cand_records = []
    for src in cand_files:
        with open(src, 'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                p = line.rstrip('\n').split('\t')
                cand_records.append((p[0], p[1] if len(p)>1 else '', p[2] if len(p)>2 else '', p[3] if len(p)>3 else ''))
    print(f"   -> Loaded {len(cand_records):,} total candidates in {time.time()-t0:.2f}s.")

    # Build Inverted Indexes
    t_idx = time.time()
    token_index = defaultdict(list)
    compact_index = defaultdict(list)
    phone_index = defaultdict(list)
    zip_bldg_index = defaultdict(list)
    zip_token_index = defaultdict(list)
    bldg_token_index = defaultdict(list)
    addr_num_tok_index = defaultdict(list)
    addr_pair_index = defaultdict(list)
    cand_snm_list = []

    for eid, nm, ad, ct in cand_records:
        norm = normalize_name_upgraded(nm)
        compact = norm.replace(' ', '')
        cand_snm_list.append((norm, eid))

        toks = [t for t in norm.split() if len(t) >= 3 and t not in STOPWORDS]
        for t in toks[:4]:
            token_index[t].append(eid)

        if len(compact) >= 4:
            compact_index[compact[:8]].append(eid)

        for t in toks[:2]:
            try:
                m = jellyfish.metaphone(t)
                if m:
                    phone_index[m].append(eid)
            except Exception:
                pass

        z, b, ad_toks, ad_nums = extract_addr_features_upgraded(ad)
        if z and b:
            zip_bldg_index[f"{z}_{b}"].append(eid)
        if z and ad_toks:
            for at in ad_toks[:2]:
                zip_token_index[f"{z}_{at}"].append(eid)
        if b and ad_toks:
            bldg_token_index[f"{b}_{ad_toks[0]}"].append(eid)

        # New High-Precision Address Channels
        for num in ad_nums[:2]:
            for at in ad_toks[:3]:
                addr_num_tok_index[f"{num}_{at}"].append(eid)
        if not ad_nums and len(ad_toks) >= 2:
            addr_pair_index[f"{ad_toks[0]}_{ad_toks[1]}"].append(eid)

    cand_snm_list.sort(key=lambda x: x[0])
    cand_snm_names = [x[0] for x in cand_snm_list]
    cand_snm_ids = [x[1] for x in cand_snm_list]
    print(f"   -> Upgraded inverted indexes ready in {time.time()-t_idx:.2f}s.")

    # 4. Generate candidates for the 10,000 test entities
    print("[4/4] Querying candidates for 10,000 S1 entities...")
    t_match = time.time()
    MAX_CANDS = 60
    cand_hits = 0
    total_pairs = 0

    for idx, (s1_id, nm, ad, ct) in enumerate(s1_records):
        norm = normalize_name_upgraded(nm)
        compact = norm.replace(' ', '')
        scores = {}

        toks = [t for t in norm.split() if len(t) >= 3 and t not in STOPWORDS]
        for t in toks[:4]:
            cl = token_index.get(t)
            if cl and len(cl) < 1500: # expanded from 800
                w = 3.5 if len(cl) < 50 else (2.0 if len(cl) < 200 else 1.0)
                for c in cl:
                    scores[c] = scores.get(c, 0.0) + w

        if len(compact) >= 4:
            cl = compact_index.get(compact[:8])
            if cl and len(cl) < 500:
                for c in cl:
                    scores[c] = scores.get(c, 0.0) + 4.0

        for t in toks[:2]:
            try:
                m = jellyfish.metaphone(t)
                cl = phone_index.get(m)
                if cl and len(cl) < 600:
                    for c in cl:
                        scores[c] = scores.get(c, 0.0) + 1.5
            except Exception:
                pass

        z, b, ad_toks, ad_nums = extract_addr_features_upgraded(ad)
        if z and b:
            cl = zip_bldg_index.get(f"{z}_{b}")
            if cl and len(cl) < 500:
                for c in cl:
                    scores[c] = scores.get(c, 0.0) + 5.0
        if z and ad_toks:
            for at in ad_toks[:2]:
                cl = zip_token_index.get(f"{z}_{at}")
                if cl and len(cl) < 500:
                    for c in cl:
                        scores[c] = scores.get(c, 0.0) + 3.0
        if b and ad_toks:
            cl = bldg_token_index.get(f"{b}_{ad_toks[0]}")
            if cl and len(cl) < 400:
                for c in cl:
                    scores[c] = scores.get(c, 0.0) + 4.0

        # Query Address Number + Token Channel
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

        # SNM
        if norm:
            pos = bisect.bisect_left(cand_snm_names, norm)
            for p_idx in range(max(0, pos - 4), min(len(cand_snm_names), pos + 4)):
                cid = cand_snm_ids[p_idx]
                scores[cid] = scores.get(cid, 0.0) + 1.0

        if len(scores) <= MAX_CANDS:
            top_cands = list(scores.keys())
        else:
            top_cands = heapq.nlargest(MAX_CANDS, scores, key=scores.get)

        total_pairs += len(top_cands)
        for c in top_cands:
            if (s1_id, c) in gt_tested:
                cand_hits += 1

    print("\n" + "=" * 80)
    print(f" UPGRADED BLOCKING RESULTS (10,000 S1 Entities):")
    print(f"   -> Total Candidates Generated:     {total_pairs:,} ({total_pairs/len(s1_records):.1f} per entity)")
    print(f"   -> True Ground Truth Matches:      {len(gt_tested):,}")
    print(f"   -> True Matches Captured:          {cand_hits:,}")
    print(f"   -> UPGRADED BLOCKING RECALL:       {cand_hits / len(gt_tested) * 100:.2f}%")
    print(f"   -> Prior Baseline Blocking Recall: 74.68%")
    print(f"   -> Absolute Recall Gain:           +{cand_hits / len(gt_tested) * 100 - 74.68:.2f}%")
    print(f" Time elapsed: {(time.time()-t_match):.1f}s")
    print("=" * 80)

if __name__ == '__main__':
    run_test()
