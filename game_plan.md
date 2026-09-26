# Amazon ML Challenge 2026: 0.99 F0.5 Score Master Plan (Detailed Precision Focus)

> **LATEST UPDATE & DETAILED IMPLEMENTATION PLAN:**  
> See [PRECISION_IMPROVEMENT_PLAN.md](file:///c:/Users/mohit/Downloads/6ab10eb3b23ba_student_resource/PRECISION_IMPROVEMENT_PLAN.md) for the complete, empirical post-mortem, negative disqualifier features, threshold calibration, and 7B/8B Precision Auditor implementation.

With the Blocking Team (Team 1) executing a highly aggressive, multi-method union strategy (Token, Prefix, Phonetic, Sorted Neighborhood, and Address blocking), we are guaranteed an excellent recall ceiling. 

However, because they are taking the **union** of all these methods, the `candidate_pairs.tsv` will be massive and filled with specific types of "tricky" false positives. 

Here is the **hyper-detailed plan for Team 2 (Matching/Precision)**, specifically designed to counter the exact noise patterns Team 1 will generate.

---

## 🏗️ Team 1: The Blocking Strategy (Summary)
*   **Methods:** Token, Prefix, Phonetic (Soundex/Metaphone), Sorted Neighborhood, Address parsing.
*   **The Output:** High recall, but it will pass pairs that:
    1.  *Sound the same but are different* (due to Phonetic blocking).
    2.  *Are located in the exact same building/street* (due to Address blocking).
    3.  *Share generic words like "Cafe" or "LLC"* (due to Token blocking).

---

## 🎯 Team 2: The Sequential Cascade Matching Pipeline (Deep Dive)

We will use the **Rejection Cascade** to maximize the $F_{0.5}$ score. The goal is to aggressively penalize false positives.

### Stage 1: The Gatekeeper (Feature Engineering + LightGBM)
Since Team 1 is throwing everything into the candidate pool, Engine A must be lightning-fast and explicitly designed to shoot down the false positives created by Team 1.

**1. Targeted Feature Engineering (using `rapidfuzz`, `jellyfish`, `usaddress`):**
To defeat Team 1's noisy blocks, we must compute pairwise similarity features for every candidate pair. 

*   **Defeating Phonetic Blocks (Name Features):**
    *   `name_levenshtein_distance`: Exact character edit distance.
    *   `name_jaro_winkler`: Heavily weights matches at the *beginning* of the string (great for catching typos in Phonetic blocks).
*   **Defeating Token Blocks (Token Features):**
    *   `rapidfuzz.fuzz.token_set_ratio`: If Name A is "Bright Cafe" and Name B is "Bright Cafe LLC", this scores 100%. If B is "Dark Cafe", it scores low. 
*   **Defeating Address Blocks (Address Features):**
    *   If Team 1 blocked by address, the names might be totally different. We use `usaddress` to parse both addresses.
    *   `address_street_match_score`: Boolean or fuzzy score of just the street name.
    *   `address_zip_match`: Exact match on numbers.
    *   `name_to_address_cross_match`: Sometimes a DBA name is in the address field.

**2. Training Strategy:**
*   **Target Variable:** 1 for True Match (from Ground Truth), 0 for everything else in the candidate pool.
*   **Loss Function:** We will use a custom weighted objective in LightGBM, or simply set `scale_pos_weight` lower to make the model inherently conservative (precision-biased).
*   **Action:** 
    *   Confidence > 90%: **MATCH**
    *   Confidence < 30%: **REJECT**
    *   Between 30% - 90%: **Pass to Stage 2**

---

### Stage 2: The Deep Semantic Inspector (Cross-Encoder)
Stage 1 will struggle with semantic differences that string-matching libraries can't understand (e.g., "SBI ATM" vs "State Bank of India Kiosk"). 

**1. The Architecture:**
*   We will fine-tune a pre-trained **Cross-Encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2` or `DeBERTa-v3-small`).
*   Unlike Bi-Encoders used in blocking, Cross-Encoders pass *both* strings through the transformer self-attention layers simultaneously, allowing the model to compare words directly contextually.

**2. The Input Format:**
We format the data into a single string for the model:
`[CLS] Name: {Name1} | Addr: {Addr1} [SEP] Name: {Name2} | Addr: {Addr2} [SEP]`

**3. Action:**
*   Confidence > 85%: **MATCH**
*   Confidence < 50%: **REJECT**
*   Between 50% - 85%: **Pass to Stage 3 (The hardest ~1-3% of candidates)**

---

### Stage 3: The Supreme Judge (Local LLM < 8B)
Only the most mind-bending, ambiguous pairs reach this stage. We will use a fast 8B model (e.g., `Llama-3-8B-Instruct` or `Qwen2.5-7B-Instruct`) running locally.

**1. The Prompt Engineering (Batch Prompting - 10 pairs/iteration):**
We give the LLM specific rules about the business logic of the dataset, directly addressing the noise patterns mentioned in the challenge README. To save compute and time, we process candidates in batches of 10.

```text
You are an expert entity resolution system. Determine if Business A and Business B are the exact same real-world entity. 
Familiarize yourself with these rules:
1. Missing components (like missing PIN codes or "Private" vs "Pvt") do NOT mean they are different businesses.
2. Two businesses sharing a street address but having totally different names (e.g., "McDonalds" vs "Starbucks") are DIFFERENT.
3. Transliteration variants (e.g., Indian or French names spelled slightly differently) are the SAME.

Evaluate the following list of 10 business pairs. 
Respond ONLY with a valid JSON array of strings, where each string is strictly "YES" or "NO", corresponding to the exact order of the pairs. Do not explain.

[
  {
    "pair_id": 1,
    "Business_A": {"Name": "{name_1_1}", "Address": "{address_1_1}"},
    "Business_B": {"Name": "{name_2_1}", "Address": "{address_2_1}"}
  },
  ... (up to 10 pairs) ...
  {
    "pair_id": 10,
    "Business_A": {"Name": "{name_1_10}", "Address": "{address_1_10}"},
    "Business_B": {"Name": "{name_2_10}", "Address": "{address_2_10}"}
  }
]
```

**2. Action:**
*   The LLM's JSON array of YES/NO is the final word. By evaluating 10 pairs per prompt, we drastically reduce inference time over the few thousand borderline cases.

---

## 🛠️ Execution Plan & Validation

Because F0.5 punishes false matches brutally, Team 2's validation loop is critical.

1.  **The Validation Split:** We will take 20% of the training Source 1 entities and put them aside. 
2.  **Evaluating the Cascade:** 
    *   We run Team 1's blocking script on the validation split.
    *   We train Stage 1 (LightGBM) on the remaining 80%.
    *   We test the cascade. If our F0.5 score is low, we **raise the confidence threshold** of Stage 1 to reject more candidates earlier.
3.  **Handling Singletons:** Team 1's aggressive blocking will generate candidates for entities that shouldn't have matches (singletons). Stage 1 must be heavily trained to confidently reject all candidates for an entity, ensuring we get the `1.0` reward for singletons.
