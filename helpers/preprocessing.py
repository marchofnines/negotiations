import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from IPython.core.display import display, HTML
    
def mystats(df, nulls=False):
    #unique_counts = []
    #for col in df.columns:
    #    unique_values = df[col][~df[col].isna() & (df[col] != '') & (df[col] != ' ')].unique()
    #    unique_counts.append(len(unique_values))
    #summary_data['unique_values_excluding_null_blank_nan'] = unique_counts
    
    stats_df = pd.DataFrame({#'blank_spaces': df.apply(lambda x: x.str.isspace().sum(), axis=0), #(df==' ').sum(),
                             #'empty_strs': (df=='').sum(),
                             'nulls': df.isnull().sum(), 
                             'null_pct': round(100*(df.isnull().sum()) / len(df),2), 
                             'unique_not_null': df.nunique(),
                             #'unique_not_null_blank_empty': unique_counts,
                             #'dups': [df[col].duplicated().sum() for col in df.columns]
                            }).sort_values(by='null_pct', ascending=False)
    print(df.shape)
    if nulls:
        return stats_df
    else: 
        return stats_df.query('nulls > 0')

"""def replace_values_in_query(df, col, condition, set_values):
    indices = df.query(f"{col}{condition}").index
    df.loc[indices, col]=set_values"""

def get_rows_with_value_count_threshold(df, col, threshold):
    # Get the values in 'col' that have a count greater than 'threshold'
    col_value_list = df[col].value_counts().loc[lambda x: x >= threshold].index
    # Return rows where 'col' is in the list of values above the threshold
    return df[df[col].isin(col_value_list)]

def reduce_dimensionality(df, col, min_num_rows_per_dimension, default_value):
    # Find the values that occur at least the minimum number of times
    df_top_values = get_rows_with_value_count_threshold(df, col, min_num_rows_per_dimension)
    
    # Identify the values to replace (those NOT in df_top_values)
    to_replace = df.loc[~df.index.isin(df_top_values.index), col].unique()
    
    # Replace the values
    df.loc[df[col].isin(to_replace), col] = default_value

"""
#def get_rows_with_value_count_threshold(df, col, threshold):
    #col_vcs=df[col].value_counts()
    #col_value_list = col_vcs[col_vcs > threshold].index
    #col_values_mask = df[col].isin(col_value_list)
    #return df.loc[col_values_mask]

def reduce_dimensionality(df, col, min_num_rows_per_dimension, default_value, view_top_dim=10):
    # Find the values that occur less than the minimum number of times
    value_counts = df[col].value_counts()
    to_replace = value_counts[value_counts < min_num_rows_per_dimension].index

    # Create a mask for rows that have those values
    mask = df[col].isin(to_replace)
    
    # Replace the values
    df.loc[mask, col] = default_value"""
    
def consolidate_values(df, col, needs_replaced, good_value):
    df.loc[df[col].isin(needs_replaced), col] = good_value #df.col.value_counts().index[0]

def smart_replace(df, col, needs_replaced, good_value, case=False):
    if isinstance(needs_replaced, list):
        for needs_replaced_str in needs_replaced: 
            df.loc[df[col].str.contains(needs_replaced_str, case=case, na=False), col] = good_value #df.col.value_counts().index[0]
    else: 
        df.loc[df[col].str.contains(needs_replaced, case=case, na=False), col] = good_value
        


#find number of zeroes 
#df_edits.eq(0).sum()
#df.isna().sum()
#print(df_edits.level.dtype)
#print(df_edits.level.isna().sum())

#drop dups - currently we have no dups
#duplicate_rows=raw[raw.duplicated()]
#duplicate_rows
#df_edits = raw.drop_duplicates(inplace=False)

def edit_cols(raw):
    #edit field names
    df_edits = raw.rename(columns={"Level": "level",
                            "NSA vs NNSA Claims": "NSA_NNSA", 
                            "SELF vs FULLY Claims": "plan_funding",
                            "Day of Decision Date": "decision_date",
                            "NSA Open Negotiation or Prepayment Negotiation?": "negotiation_type",
                            "Split claim": "split_claim",
                            "In response to:": "in_response_to",
                            "Decision": "decision",
                            "Negotiating Amount": "negotiation_amount",
                            "Group Number": "group_number",
                            "Day of Deadline": "deadline",
                            "TPA Representative": "TPA_rep",
                            "Insurance Name":"carrier",
                            "Billed Amount": "billed_amount",
                            "Max Offer $ Amount": "offer",
                            "Max Counter Offer $ Amount": "counter_offer",
                            "Negotiation ID": "negotiation_id", 
                            "Claim Status": "claim_status",
                            "Claim ID": "claim_ID",
                            "TPA":"TPA",
                            "Facility":"facility",
                            "Day of Patient Date of Birth":"DOB",
                            "Day of D.O.S.": "service_date",
                            "Claim Type": "claim_type",
                            "Max Offer Received Date": "offer_date", 
                            "Max Counter Offer Date": "counter_offer_date"}, inplace=False)

    #select and reorder final columns 
    df_edits = df_edits[['claim_ID', 'claim_type','NSA_NNSA','split_claim', 'negotiation_type','in_response_to', 'claim_status', 
                        'level', 'facility', 'service_date',  
                        'DOB', 'carrier','group_number','plan_funding','TPA', 'TPA_rep', 
                        'billed_amount','negotiation_amount','offer','counter_offer', 'offer_date', 'counter_offer_date', 
                        'deadline', 'decision_date', 'decision'
                        #,'negotiation_id', 'claim_id'
                        ]]
    return df_edits




def consolidate_values_TC(df_edits):
    consolidate_values(df_edits, 'TPA', ['Data iSIGHT', 'DataiSight'], 'Dataisight')
    consolidate_values(df_edits, 'TPA', ['Spring Tide', 'SpringTide'], 'Spring Tide Health')
    consolidate_values(df_edits, 'TPA', ['SANA BENEFITS'], 'Sana Benefits')
    consolidate_values(df_edits, 'TPA', ['Spring Tide'], 'Spring Tide Health')
    consolidate_values(df_edits, 'TPA', ['HRGi', 'HRG'], 'HRGI')
    consolidate_values(df_edits, 'TPA', ['The Phia Group', 'The Phia Group, LLC', 'Phia Group'], 'The Phia Group')
    consolidate_values(df_edits, 'TPA', 
                        ['Friday health Plan', 'Friday Health', 'Friday health Plans', 'Friday Health Plans', 'friday health plans'],
                        'Friday Health Plan' )
    consolidate_values(df_edits, 'TPA', 
                        ['6 degrees health', '6 degrees Health', '6 Degree Health'],
                        '6 Degrees Health' )
    consolidate_values(df_edits, 'TPA', 
                        ['Precision Benefit Services, Inc', 'Precision'],
                        'Precision Benefit Services' )
    consolidate_values(df_edits, 'TPA', 
                        ['The Health Plans'],
                        'The Health Plan' )
    consolidate_values(df_edits, 'TPA', 
                        ['Baylor Scott & White Health Plan', 'Baylor scott & white', 'Baylor Scott & White', 'BSW'],
                        'Baylor Scott and White' )
    consolidate_values(df_edits, 'TPA', 
                        ['GC', 'CGS'],
                        'GCS' )
    consolidate_values(df_edits, 'TPA', 
                        ['Geha'],
                        'GEHA' )
    consolidate_values(df_edits, 'TPA', 
                        ['aetna', 'Aetna/GCS', 'Aetna GCS'], 
                        'Aetna' )
    #pp.consolidate_values(df_edits, 'TPA', 
    #                     ['BCBSTX', 'BCBS Texas'], 
    #                     'BCBS of Texas' )
    consolidate_values(df_edits, 'TPA', 
                        ['bcbs', 'BCBS portal', 'BCBS PORTAL', 'Blue Cross Blue Shield', 'BCBSTX', 'BCBS Texas', 'BCBS of Texas'], 
                        'BCBS' )
    consolidate_values(df_edits, 'in_response_to', ['Verbal request'], 'Verbal Request')
    consolidate_values(df_edits, 'in_response_to', 
                        ['Corrected claim', 'Corrected Claim sent', 'NSA corrected claim'], 
                        'Corrected Claim' )
    consolidate_values(df_edits, 'in_response_to', 
                        ['NNSA Negotiation Letter'], 
                        'Negotiation Letter' )
    consolidate_values(df_edits, 'in_response_to', 
                        ['Money Appeal'], 
                        'Current Money Appeal' )
    
    df_edits['carrier'] = df_edits['carrier'].str.replace(r'^ZZZ', '', regex=True)
    df_edits['carrier'] = df_edits['carrier'].str.replace(r'^zzz', '', regex=True)
    df_edits['carrier'] = df_edits['carrier'].str.replace(r'^x_', '', regex=True)

    consolidate_values(df_edits, 'carrier', 
                        ['AETNA', 'AETNA HEALTH PLANS'], 
                        'Aetna' )
    consolidate_values(df_edits, 'carrier', 
                        ['CIGNA', 'CIGNA HEALTH PLANS'], 
                        'Cigna' )
    consolidate_values(df_edits, 'carrier', 
                        ['MERITAINAETNA'], 
                        'Meritain Aetna' )
    consolidate_values(df_edits, 'carrier', 
                        ['UNITED HEALTH CARE', 'UNITED HEALTHCARE', 'United Healthcare'], 
                        'United Health Care' )
    consolidate_values(df_edits, 'carrier', 
                        ['Baylor Scott and White Health Plan', 'SCOTT & WHITE HEALTH PLANS', 'Baylor Scott and White Health'], 
                        'Baylor Scott and White' )
    consolidate_values(df_edits, 'carrier', 
                        ['BLUE CROSS AND BLUE SHIELD OF TEXAS', 'BCBS of Texas', 'Anthem BlueCross BlueShield', 'BCBS'], 
                        'Blue Cross Blue Shield' )
    consolidate_values(df_edits, 'carrier', 
                        ['90 DAY BENEFITS HEALTHSMART'], 
                        '90 day benefits HealthSmart' )
    consolidate_values(df_edits, 'carrier', 
                        ['90 DEGREE BENEFITS'], 
                        '90 Degree Benefits' )
    consolidate_values(df_edits, 'carrier', 
                        ['ALLSAVERS'], 
                        'United Health Care All Savers' )
    consolidate_values(df_edits, 'carrier', 
                        ['FRIDAY HEALTH PLAN'], 
                        'Friday Health Plans' )
    consolidate_values(df_edits, 'carrier', 
                        ['GALLAGHER BASSETT'], 
                        'Gallagher Bassett' )


def compare_acceptance_rates(df, col, val1, val2, descr1=None, descr2=None):
    if descr1 is None: 
        descr1=val1
    if descr2 is None: 
        descr2=val2
        
    decision_map={'Rejected': 0, 'Accepted':1}
    #compute group means
    group1_AR=df.query(f"{col}=='{val1}'")['decision'].map(decision_map).mean()
    group2_AR=df.query(f"{col}=='{val2}'")['decision'].map(decision_map).mean()
    
    if group1_AR > group2_AR:
        print(f"\n{descr1} claims are {group1_AR/group2_AR:.2f} more likely to be accepted than {descr2} claims")
    else:
        print(f"\n{descr2} claims are {group2_AR/group1_AR:.2f} more likely to be accepted than {descr1} claims")
 



def acompare_acceptance_rates(df1, df2, df1_descr, df2_descr):
    """
    This function takes two subsets of our dataframe and computes the means of 'Y' for each group which is the
    same thing as the acceptance rate. 
    
    It then outputs: 
       - The description of the first group along with its acceptance rate
       - The description of the second group along with its acceptance rate
       - It then computes the ratio of the acceptance rates between both groups (numerator is the dominant group)

    Parameters: 
     - df1: first subset of our dataframe (group1)
     - df2: second subset of our dataframe (group2)
     - df1_descr: description of first subset
     - df2_descr: description of second subset
     
     Returns: None
    """
    decision_map={'Rejected': 0, 'Accepted':1}
    #compute group means
    group1_AR=df1['decision'].map(decision_map).mean()
    group2_AR=df1['decision'].map(decision_map).mean()
    
    #print acceptance rate for each group and for the ratio of the largest to the smallest
    #print(f"\nThe acceptance rate for {df1_descr} is {100*group1_AR:.2f}%")
    #print(f"The acceptance rate for {df2_descr} is {100*group2_AR:.2f}%")
    if group1_AR > group2_AR:
        print(f"\n{df1_descr} claims are {group1_AR/group2_AR:.2f} more likely to be accepted than {df2_descr}")
    else:
        print(f"\n{df2_descr} claims are {group2_AR/group1_AR:.2f} more likely to be accepted than {df1_descr}")
 


def calculate_iqr_bounds(df):
    """
    Calculate the lower and upper bounds for outliers for all numerical columns in a DataFrame.
    
    Parameters:
    - df: DataFrame containing the data
    
    Returns:
    - A DataFrame with columns as index and lower and upper bounds as columns
    """
    # Initialize an empty DataFrame to store the results
    result = pd.DataFrame(columns=["lower_bound", "upper_bound", "num_within_bounds", "num_below_lower", "num_above_upper"])
    
    # Loop through each numerical column in the DataFrame
    for column in df.select_dtypes(include=[np.number]).columns:
        # Calculate Q1, Q3, and IQR
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        # Calculate the lower and upper bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        num_below_lower = df[df[column] < lower_bound].shape[0]   
        num_above_upper = df[df[column] > upper_bound].shape[0] 
        num_within_bounds = df[(df[column] >= lower_bound)&(df[column] <= upper_bound)].shape[0] 
        
        # Add the results to the DataFrame
        result.loc[column] = [lower_bound, upper_bound, num_within_bounds, num_below_lower, num_above_upper]
        
    return result

def remove_all_outliers_based_on_IQRs(df, factor=1.5):
    """
    Remove outliers from a DataFrame based on IQR, only considering numeric columns.
    
    Parameters:
        df (DataFrame): The original DataFrame.
        factor (float): Factor to multiply the IQR range to set bounds.
        
    Returns:
        DataFrame: A new DataFrame with outliers removed.
    """
    # Select only numeric columns
    df_numeric = df.select_dtypes(include=['number'])
    
    # Calculate Q1, Q3 and IQR for each numeric column
    Q1 = df_numeric.quantile(0.25)
    Q3 = df_numeric.quantile(0.75)
    IQR = Q3 - Q1
    
    # Define bounds for the outliers
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    # Identify the outliers
    not_outliers = ((df_numeric >= lower_bound) & (df_numeric <= upper_bound)).all(axis=1)
    
    # Remove outliers from the DataFrame based on numeric columns
    df_cleaned = df[not_outliers]
    print(f'Returned DataFrame has: {df_cleaned.shape[0]} samples remainining')
    return df_cleaned

# Create a sampl

def remove_col_outliers_based_on_IQRs(df, column_name):
    """
    Removes outliers from a specific column in a dataframe using the IQR method.
    
    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - column_name (str): The column from which outliers should be removed.
    
    Returns:
    - pd.DataFrame: A dataframe with outliers removed from the specified column.
    """
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    print(f'lower_bound: {lower_bound} and upper_bound: {upper_bound}')
    df_cleaned = df.query(f"{lower_bound} <= {column_name} <= {upper_bound}") 
    #df_cleaned = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]
    print(f'Removed {len(df)- len(df_cleaned)} rows')
    
    return df_cleaned

import pandas as pd
from scipy import stats

def remove_outliers_based_on_z_scores(df, z_threshold=3):
    """
    Remove outliers from a DataFrame based on Z-score, only considering numeric columns.
    
    Parameters:
        df (DataFrame): The original DataFrame.
        z_threshold (float): Z-score threshold for outlier detection.
        
    Returns:
        DataFrame: A new DataFrame with outliers removed.
    """
    
    # Select only numeric columns
    df_numeric = df.select_dtypes(include=['number'])
    
    # Calculate Z-scores for numeric columns
    z_scores = pd.DataFrame(stats.zscore(df_numeric, nan_policy='omit'), columns=df_numeric.columns)
    
    # Get boolean DataFrame where True indicates the presence of an outlier in numeric columns
    outliers = (z_scores.abs() > z_threshold)
    
    # Remove outliers from the DataFrame based on numeric columns
    df_cleaned = df[~outliers.any(axis=1)]
    
    return df_cleaned


def consolidate_values(df, col, list_of_bad_values, good_value):
    """
    Consolidate multiple values in a DataFrame column into a single value.
    
    Parameters:
    - df: pandas DataFrame, the dataset containing the column to be modified
    - col: str, the name of the column to be modified
    - list_of_bad_values: list, the values in the column to be replaced
    - good_value: the value to replace the 'bad' values with
    
    Returns:
    - None: The function modifies the DataFrame in place.
    """
    # Use DataFrame.loc to find rows where the column's value is in list_of_bad_values
    # and replace them with good_value
    df.loc[df[col].isin(list_of_bad_values), col] = good_value #df.col.value_counts().index[0]


def evaluate_scaler_imputers(scalers, models, X,y, scoring, imputers=None, cv=5):
    """
    
    """
    pipelines = []
    cv_scores = []
    if imputers:
        for scaler in scalers:
            for imputer in imputers:
                for model in models:
                    pipe = Pipeline([
                        ('scaler', scaler),
                        ('imputer', imputer),
                        ('model', model)
                    ])
                    #pipelines.append(pipe)
                    scaler_name=pipe.named_steps['scaler'].__class__.__name__
                    imputer_name=pipe.named_steps['imputer'].__class__.__name__
                    model_name=pipe.named_steps['model'].__class__.__name__
                    
                    score = cross_val_score(pipe, X, y, scoring=scoring, n_jobs=-1, cv=cv)
                    cv_scores.append({'scaler': scaler_name, 'imputer': imputer_name, 'model': model_name,
                                    'score_mean': score.mean(), 'score_std': score.std()})
        display(HTML(f'<h4>CV Results with imputation</h4>'))
    else: 
        for scaler in scalers:
            for model in models:
                pipe = Pipeline([
                    ('scaler', scaler),
                    #('imputer', imputer),
                    ('model', model)
                ])
                #pipelines.append(pipe)
                scaler_name=pipe.named_steps['scaler'].__class__.__name__
                #imputer_name=pipe.named_steps['imputer'].__class__.__name__
                model_name=pipe.named_steps['model'].__class__.__name__
                
                score = cross_val_score(pipe, X, y, scoring=scoring, n_jobs=-1, cv=cv)
                cv_scores.append({'scaler': scaler_name, 'imputer': 'N/A', 'model': model_name,
                                'score_mean': score.mean(), 'score_std': score.std()})
        display(HTML(f'<h4>Baseline CV Score with nulls dropped</h4>'))

    
    df_cv_scores = pd.DataFrame(cv_scores).sort_values(by=['score_mean','score_std'],
                                                  ascending=[False, True]).reset_index(drop=True)
    display(df_cv_scores)
    