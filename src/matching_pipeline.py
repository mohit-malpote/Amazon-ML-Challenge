import pandas as pd
import numpy as np
import rapidfuzz
import jellyfish
import usaddress
import lightgbm as lgb
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch
import gc
import json

# ==========================================
# STAGE 1: Feature Engineering
# ==========================================
import re

FEATURE_COLUMNS = [
    'name_levenshtein',
    'name_jaro_winkler',
    'name_token_set_ratio',
    'address_street_match',
    'address_zip_match',
    'postal_code_mismatch',
    'street_number_mismatch',
    'street_number_match',
    'brand_first_token_ratio',
    'clean_legal_name_match',
    'name_length_ratio',
    'address_length_ratio',
    'name_to_address_cross_match',
    'address_name_ratio_diff',
    'exact_name_match',
    'country_mismatch'
]

legal_suffixes_regex = re.compile(r'\b(pvt\s+ltd|private\s+limited|llc|inc|corp|corporation|co|company|ltd|limited|llp|gmbh|sarl|sa)\b', re.IGNORECASE)
punct_regex = re.compile(r'[^\w\s]')
noise_tokens = {'the', 'a', 'an', 'of', 'in', 'and'}
bracket_regex = re.compile(r'\[.*?\]')
prefix_noise_regex = re.compile(r'^\s*\b(shri|sri|sree|m/s|smt|dr|mr|mrs|partners|the|a|an|of)\b\s*', re.IGNORECASE)
leading_num_pattern = re.compile(r'^\s*(\d+)')
zip_pattern = re.compile(r'\b\d{5,6}\b')

def clean_brand_name(name):
    s = bracket_regex.sub(' ', name)
    s = prefix_noise_regex.sub(' ', s)
    s = punct_regex.sub(' ', s)
    s = legal_suffixes_regex.sub(' ', s)
    return ' '.join(s.split())

def clean_legal(name):
    s = bracket_regex.sub(' ', name)
    s = punct_regex.sub(' ', s.lower())
    s = legal_suffixes_regex.sub(' ', s)
    s = prefix_noise_regex.sub(' ', s)
    return ' '.join(s.split())

def get_first_token(name):
    cleaned = clean_brand_name(name)
    tokens = [t for t in re.findall(r'\b\w+\b', cleaned.lower()) if t not in noise_tokens]
    return tokens[0] if tokens else ""

def get_leading_num(addr):
    m = leading_num_pattern.match(addr)
    return m.group(1) if m else ""

def get_zip(addr):
    m = zip_pattern.search(addr)
    return m.group(0) if m else ""

def length_ratio(s1, s2):
    l1, l2 = len(s1), len(s2)
    mx = max(l1, l2)
    return min(l1, l2) / mx if mx > 0 else 0.0

def compute_features_fast(df):
    print(f"Vectorizing feature computation for {len(df):,} pairs...")
    n1 = df['name1'].fillna("").astype(str).str.lower().tolist()
    n2 = df['name2'].fillna("").astype(str).str.lower().tolist()
    a1 = df['addr1'].fillna("").astype(str).str.lower().tolist()
    a2 = df['addr2'].fillna("").astype(str).str.lower().tolist()
    
    # 1. Name Similarities
    print(" 1/8 Computing Name String Similarities (Levenshtein, JW, TSR)...")
    name_lev = [jellyfish.levenshtein_distance(a, b) for a, b in zip(n1, n2)]
    name_jw = [jellyfish.jaro_winkler_similarity(a, b) for a, b in zip(n1, n2)]
    name_tsr = [rapidfuzz.fuzz.token_set_ratio(a, b) for a, b in zip(n1, n2)]
    
    # 2. Address & Postal Code Features
    print(" 2/8 Computing Postal/PIN Code Match & Mismatch (US, France, India)...")
    z1 = [get_zip(a) for a in a1]
    z2 = [get_zip(a) for a in a2]
    addr_zip = [1 if (x and y and x == y) else (0 if (x and y and x != y) else -1) for x, y in zip(z1, z2)]
    postal_mismatch = [1 if (x and y and x != y) else 0 for x, y in zip(z1, z2)]
    
    print(" 3/8 Computing Address Street Match...")
    addr_street = [rapidfuzz.fuzz.token_set_ratio(a, b) for a, b in zip(a1, a2)]
    
    # 3. Leading Street Number Hard Disqualifiers
    print(" 4/8 Computing Street Number Match & Mismatch...")
    num1 = [get_leading_num(a) for a in a1]
    num2 = [get_leading_num(a) for a in a2]
    st_mismatch = [1 if (x and y and x != y) else 0 for x, y in zip(num1, num2)]
    st_match = [1 if (x and y and x == y) else 0 for x, y in zip(num1, num2)]
    
    # 4. Brand Disambiguation & Clean Legal Match
    print(" 5/8 Computing Brand First Token Ratio & Clean Legal Match...")
    f1 = [get_first_token(n) for n in n1]
    f2 = [get_first_token(n) for n in n2]
    brand_first = [rapidfuzz.fuzz.ratio(x, y) / 100.0 if (x and y) else 0.5 for x, y in zip(f1, f2)]
    
    clean1 = [clean_legal(n) for n in n1]
    clean2 = [clean_legal(n) for n in n2]
    clean_legal_m = [1 if (c1_ and c1_ == c2_) else 0 for c1_, c2_ in zip(clean1, clean2)]
    
    # 5. Length Ratios & Exact Match
    print(" 6/8 Computing Length Ratios & Exact Name Match...")
    name_len_r = [length_ratio(a, b) for a, b in zip(n1, n2)]
    addr_len_r = [length_ratio(a, b) for a, b in zip(a1, a2)]
    exact_name = [1 if (a and a == b) else 0 for a, b in zip(n1, n2)]
    
    # 6. Cross Match
    print(" 7/8 Computing Cross Matches...")
    cross_match = [1 if (x and x in y) or (y and y in x) else 0 for x, y in zip(n1, a2)]
    cross_match2 = [1 if (x and x in y) or (y and y in x) else 0 for x, y in zip(n2, a1)]
    final_cross = [max(c1, c2) for c1, c2 in zip(cross_match, cross_match2)]
    
    # 7. Discrepancy Feature & Country Mismatch
    print(" 8/8 Computing Address-Name Discrepancy & Country Mismatch...")
    addr_name_diff = [a - n for a, n in zip(addr_street, name_tsr)]
    
    if 'country1' in df.columns and 'country2' in df.columns:
        c1 = df['country1'].fillna("").astype(str).tolist()
        c2 = df['country2'].fillna("").astype(str).tolist()
        country_mismatch = [1 if (x and y and x != y) else 0 for x, y in zip(c1, c2)]
    else:
        country_mismatch = [0] * len(df)
        
    df['name_levenshtein'] = name_lev
    df['name_jaro_winkler'] = name_jw
    df['name_token_set_ratio'] = name_tsr
    df['address_street_match'] = addr_street
    df['address_zip_match'] = addr_zip
    df['postal_code_mismatch'] = postal_mismatch
    df['street_number_mismatch'] = st_mismatch
    df['street_number_match'] = st_match
    df['brand_first_token_ratio'] = brand_first
    df['clean_legal_name_match'] = clean_legal_m
    df['name_length_ratio'] = name_len_r
    df['address_length_ratio'] = addr_len_r
    df['name_to_address_cross_match'] = final_cross
    df['address_name_ratio_diff'] = addr_name_diff
    df['exact_name_match'] = exact_name
    df['country_mismatch'] = country_mismatch
    
    return df

# ==========================================
# STAGE 2: Deep Semantic Inspector
# ==========================================
class SemanticInspector:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        print(f"SemanticInspector initialized on device: {self.device}")
        
    def predict(self, pairs):
        # Pairs is a list of (name1, addr1, name2, addr2)
        if not pairs:
            return []
            
        text_a = [f"Name: {p[0]} | Addr: {p[1]}" for p in pairs]
        text_b = [f"Name: {p[2]} | Addr: {p[3]}" for p in pairs]
        
        batch_size = 256 if self.device.type == "cuda" else 64
        scores = []
        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                b_a = text_a[i:i+batch_size]
                b_b = text_b[i:i+batch_size]
                inputs = self.tokenizer(b_a, b_b, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                probs = torch.sigmoid(outputs.logits).squeeze(-1).cpu().numpy()
                if probs.ndim == 0:
                    scores.append(float(probs))
                else:
                    scores.extend(probs.tolist())
        return scores

# ==========================================
# ==========================================
# STAGE 3: Supreme Judge (Local LLM)
# ==========================================
class SupremeJudge:
    def __init__(self, model_name="Qwen/Qwen2.5-3B-Instruct", device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"Loading Stage 3 Supreme Judge ({model_name}) on device: {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if self.device.type == "cuda" else torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        )
        self.model.to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        print(f"Supreme Judge initialized successfully on {self.device}.")
        
    def judge_pairs(self, pairs, batch_size=None):
        """
        pairs: list of (name1, addr1, name2, addr2)
        returns: list of 'MATCH' or 'REJECT'
        """
        if not pairs:
            return []
            
        if batch_size is None:
            batch_size = 16 if self.device.type == "cuda" else 8
            
        system_prompt = (
            "You are an expert Entity Resolution Judge for business records.\n"
            "Determine whether Entity 1 and Entity 2 refer to the EXACT SAME real-world commercial entity.\n\n"
            "MATCH CRITERIA:\n"
            "1. Same core brand/business name with minor spelling variations, typos, transliterations, or punctuation differences.\n"
            "2. Different legal suffixes (Inc, LLC, Pvt Ltd, Ltd, Corp, Co, LLP) or prefixes (Shri, M/s, Partners, [Tags]).\n"
            "3. Matching street address with standard abbreviations (Rd vs Road, St vs Street, Ste vs Suite).\n\n"
            "REJECT CRITERIA:\n"
            "1. Different brands or competitors (e.g., McDonald's vs Subway, Target vs Starbucks, Nail Salon vs Barber) even if at the same address.\n"
            "2. Conflicting street/building numbers (e.g. 15 Main St vs 94 Main St) or completely different cities.\n\n"
            "Respond strictly with ONE word: 'MATCH' or 'REJECT'."
        )
        
        decisions = []
        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                if (i // batch_size) % 10 == 0 and len(pairs) > 30:
                    print(f"      Auditing progress: {min(i + batch_size, len(pairs))}/{len(pairs)} pairs...", flush=True)
                batch = pairs[i:i+batch_size]
                prompts = []
                for p in batch:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Entity 1: {p[0]} | {p[1]}\nEntity 2: {p[2]} | {p[3]}\nDecision:"}
                    ]
                    prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    prompts.append(prompt_text)
                    
                inputs = self.tokenizer(prompts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.device)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=4,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
                for j, out_seq in enumerate(outputs):
                    input_len = inputs.input_ids[j].shape[0]
                    generated_tokens = out_seq[input_len:]
                    text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip().upper()
                    if "MATCH" in text and "REJECT" not in text:
                        decisions.append("MATCH")
                    else:
                        decisions.append("REJECT")
                        
        return decisions

# ==========================================
# CASCADE PIPELINE EXECUTION
# ==========================================
def run_cascade_pipeline(candidate_df, lgb_model, semantic_inspector, supreme_judge=None,
                         s1_match_thresh=0.90, s1_reject_thresh=0.20,
                         s2_match_thresh=0.88, s2_reject_thresh=0.45):
    """
    candidate_df: DataFrame with columns [id1, id2, name1, addr1, name2, addr2]
    lgb_model: trained LightGBM model
    semantic_inspector: Stage 2 Cross-Encoder model
    supreme_judge: Stage 3 Local LLM reasoner (optional)
    """
    print("Stage 1: Feature Engineering & LightGBM")
    candidate_df = compute_features_fast(candidate_df)
    
    stage1_preds = lgb_model.predict(candidate_df[FEATURE_COLUMNS])
    candidate_df['stage1_score'] = stage1_preds
    
    # Decisions Stage 1
    candidate_df['decision'] = 'PENDING'
    candidate_df.loc[candidate_df['stage1_score'] >= s1_match_thresh, 'decision'] = 'MATCH'
    candidate_df.loc[candidate_df['stage1_score'] <= s1_reject_thresh, 'decision'] = 'REJECT'
    
    # Hard Disqualification: Country Mismatch is 100% Reject
    if 'country_mismatch' in candidate_df.columns:
        candidate_df.loc[candidate_df['country_mismatch'] == 1, 'decision'] = 'REJECT'
        
    # Hard Disqualification: Street Number Mismatch with low name similarity
    st_and_name_mismatch = (candidate_df['street_number_mismatch'] == 1) & (candidate_df['name_token_set_ratio'] < 80)
    candidate_df.loc[st_and_name_mismatch, 'decision'] = 'REJECT'
    
    # Pass to Stage 2
    stage2_mask = candidate_df['decision'] == 'PENDING'
    stage2_df = candidate_df[stage2_mask].copy()
    
    if len(stage2_df) > 0 and semantic_inspector is not None:
        print(f"Stage 2: Semantic Inspector on {len(stage2_df)} pairs")
        pairs_for_s2 = list(zip(stage2_df['name1'], stage2_df['addr1'], stage2_df['name2'], stage2_df['addr2']))
        s2_scores = semantic_inspector.predict(pairs_for_s2)
        candidate_df.loc[stage2_mask, 'stage2_score'] = s2_scores
        
        if supreme_judge is not None:
            # Stage 2 definite decisions
            candidate_df.loc[stage2_mask & (candidate_df['stage2_score'] >= s2_match_thresh), 'decision'] = 'MATCH'
            candidate_df.loc[stage2_mask & (candidate_df['stage2_score'] < s2_reject_thresh), 'decision'] = 'REJECT'
            
            # Suspect matches that need LLM precision audit:
            # 1. Uncertainty zone: s2_reject_thresh <= S2 < s2_match_thresh
            # 2. Or S2 is high, but brand first token is totally different
            suspect_high = stage2_mask & (candidate_df['stage2_score'] >= s2_match_thresh) & (candidate_df['brand_first_token_ratio'] < 0.40) & (candidate_df['name_token_set_ratio'] < 75)
            candidate_df.loc[suspect_high, 'decision'] = 'PENDING'
        else:
            # 2-stage cascade: Stage 2 makes the final call with calibrated threshold
            candidate_df.loc[stage2_mask & (candidate_df['stage2_score'] >= s2_match_thresh), 'decision'] = 'MATCH'
            candidate_df.loc[stage2_mask & (candidate_df['stage2_score'] < s2_match_thresh), 'decision'] = 'REJECT'
        
    # Pass to Stage 3 (if supreme_judge provided)
    stage3_mask = candidate_df['decision'] == 'PENDING'
    stage3_df = candidate_df[stage3_mask].copy()
    
    if len(stage3_df) > 0 and supreme_judge is not None:
        print(f"Stage 3: Supreme Judge Veto Auditor on {len(stage3_df)} pairs...")
        pairs_for_s3 = list(zip(stage3_df['name1'], stage3_df['addr1'], stage3_df['name2'], stage3_df['addr2']))
        s3_decisions = supreme_judge.judge_pairs(pairs_for_s3)
        candidate_df.loc[stage3_mask, 'decision'] = s3_decisions
        
    final_matches = candidate_df[candidate_df['decision'] == 'MATCH']
    return final_matches

if __name__ == "__main__":
    print("This script is a library for the cascade pipeline. Call run_cascade_pipeline with dataframes.")
