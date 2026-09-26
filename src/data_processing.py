import pandas as pd
import numpy as np

def load_source_data(s1_path, s2_path, s3_path):
    """
    Loads all source files and combines them into a single dictionary mapping:
    entity_id -> {'name': business_name, 'address': business_address}
    """
    print("Loading source datasets...")
    df1 = pd.read_csv(s1_path, sep='\t', dtype=str)
    df2 = pd.read_csv(s2_path, sep='\t', dtype=str)
    df3 = pd.read_csv(s3_path, sep='\t', dtype=str)
    
    combined = pd.concat([df1, df2, df3], ignore_index=True)
    
    # Fill NaN values with empty string
    combined['business_name'] = combined['business_name'].fillna("")
    combined['business_address'] = combined['business_address'].fillna("")
    if 'country' in combined.columns:
        combined['country'] = combined['country'].fillna("")
    else:
        combined['country'] = ""
    
    # Create lookup dictionary
    entity_dict = combined.set_index('entity_id')[['business_name', 'business_address', 'country']].to_dict('index')
    return entity_dict

def parse_list_column(df, id_col, list_col):
    """
    Parses a dataframe with a column containing comma-separated lists of IDs
    into a flattened dataframe of pairs.
    """
    # Drop rows where the list column is NaN or empty
    df = df.dropna(subset=[list_col]).copy()
    df[list_col] = df[list_col].astype(str)
    df = df[df[list_col].str.strip() != ""]
    
    # Split by comma and explode
    df[list_col] = df[list_col].str.split(',')
    flat_df = df.explode(list_col).rename(columns={id_col: 'id1', list_col: 'id2'})
    flat_df['id2'] = flat_df['id2'].str.strip()
    return flat_df[['id1', 'id2']].reset_index(drop=True)

def prepare_candidate_pairs(candidates_path, entity_dict, ground_truth_path=None):
    """
    Loads the candidates file, flattens it, and attaches the names and addresses.
    If ground_truth_path is provided, attaches a 'target' label (1 or 0).
    """
    print(f"Loading candidates from {candidates_path}...")
    try:
        is_parquet = False
        with open(candidates_path, 'rb') as f:
            if f.read(4) == b'PAR1':
                is_parquet = True
        
        if is_parquet or candidates_path.endswith('.parquet'):
            candidates_raw = pd.read_parquet(candidates_path)
        else:
            candidates_raw = pd.read_csv(candidates_path, sep='\t', dtype=str)
            if 'candidate_entity_ids' not in candidates_raw.columns:
                # Fallback for if Team 1 accidentally saved as CSV
                candidates_raw = pd.read_csv(candidates_path, sep=',', dtype=str)
    except FileNotFoundError:
        print(f"WARNING: Candidates file {candidates_path} not found. Returning empty DataFrame.")
        return pd.DataFrame(columns=['id1', 'id2', 'name1', 'addr1', 'name2', 'addr2'])

    # Check if already flattened (singular column name)
    if 'candidate_entity_id' in candidates_raw.columns and 'candidate_entity_ids' not in candidates_raw.columns:
        print("Candidates are already flattened. Renaming columns...")
        df_pairs = candidates_raw.rename(columns={'source1_entity_id': 'id1', 'candidate_entity_id': 'id2'})
    else:
        # Needs flattening
        print("Flattening comma-separated candidates...")
        col_name = 'candidate_entity_ids' if 'candidate_entity_ids' in candidates_raw.columns else 'candidate_entity_id'
        df_pairs = parse_list_column(candidates_raw, 'source1_entity_id', col_name)
    
    print(f"Flattened into {len(df_pairs)} candidate pairs.")
    
    # Map names and addresses
    print("Mapping names and addresses...")
    df_pairs['name1'] = df_pairs['id1'].map(lambda x: entity_dict.get(x, {}).get('business_name', ''))
    df_pairs['addr1'] = df_pairs['id1'].map(lambda x: entity_dict.get(x, {}).get('business_address', ''))
    df_pairs['country1'] = df_pairs['id1'].map(lambda x: entity_dict.get(x, {}).get('country', ''))
    
    df_pairs['name2'] = df_pairs['id2'].map(lambda x: entity_dict.get(x, {}).get('business_name', ''))
    df_pairs['addr2'] = df_pairs['id2'].map(lambda x: entity_dict.get(x, {}).get('business_address', ''))
    df_pairs['country2'] = df_pairs['id2'].map(lambda x: entity_dict.get(x, {}).get('country', ''))

    if ground_truth_path:
        print(f"Loading ground truth from {ground_truth_path}...")
        gt_raw = pd.read_csv(ground_truth_path, sep='\t', dtype=str)
        # Ground truth has 'source1_entity_id' and 'matched_entity_ids'
        df_gt_pairs = parse_list_column(gt_raw, 'source1_entity_id', 'matched_entity_ids')
        
        # Create a set of true pairs for O(1) lookup
        true_pairs = set(zip(df_gt_pairs['id1'], df_gt_pairs['id2']))
        
        print("Assigning targets based on ground truth...")
        df_pairs['target'] = df_pairs.apply(lambda row: 1 if (row['id1'], row['id2']) in true_pairs else 0, axis=1)
        
    return df_pairs

def format_predictions_for_submission(final_matches_df, all_source1_ids):
    """
    Converts a DataFrame of final [id1, id2] matches back into the challenge output format:
    source1_entity_id \t matched_entity_ids
    Also ensures ALL source1_ids are present, even if they have no matches (singletons).
    """
    # Group matches by id1
    grouped = final_matches_df.groupby('id1')['id2'].apply(lambda x: ','.join(sorted(set(x)))).reset_index()
    grouped.columns = ['source1_entity_id', 'matched_entity_ids']
    
    # Ensure all S1 entities from the original list are included
    all_s1_df = pd.DataFrame({'source1_entity_id': all_source1_ids})
    
    submission_df = pd.merge(all_s1_df, grouped, on='source1_entity_id', how='left')
    submission_df['matched_entity_ids'] = submission_df['matched_entity_ids'].fillna("")
    
    return submission_df
