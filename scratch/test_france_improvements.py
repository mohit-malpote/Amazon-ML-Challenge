import sys
sys.stdout.reconfigure(encoding='utf-8')
import os, time
from collections import defaultdict
import pandas as pd
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(base_dir, 'src'))

from blocking import normalize_name, extract_addr_features, extract_address_keys
from matching_pipeline import clean_brand_name, clean_legal, get_first_token, get_leading_num

test_dir = os.path.join(base_dir, 'student_resource', 'dataset', 'test')

# Load 15 French test entities
france_s1 = []
with open(os.path.join(test_dir, 'test_source1.tsv'), 'r', encoding='utf-8', errors='replace') as f:
    next(f)
    for line in f:
        p = line.rstrip('\n').split('\t')
        if len(p) > 3 and p[3] == 'France':
            france_s1.append((p[0], p[1], p[2]))
            if len(france_s1) >= 15:
                break

print("Testing French Attribute Extractions with Upgraded Pipeline:")
for eid, nm, ad in france_s1:
    nm_clean = clean_brand_name(nm)
    token = get_first_token(nm)
    legal = clean_legal(nm)
    num = get_leading_num(ad)
    print(f"ID: {eid}")
    print(f"  Raw:   [{nm}] | [{ad}]")
    print(f"  Clean: [{nm_clean}] | Token: [{token}] | Legal: [{legal}] | Bldg Num: [{num}]")
