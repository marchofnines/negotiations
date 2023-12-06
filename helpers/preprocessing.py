"""
Author: Basil Haddad
Date: 11.01.2023

Description:
    Helper functions for preprocessing and conducting Exploratory Data Analysis (EDA) on datasets. 
"""

from importlib import reload
from helpers.my_imports import *

def edit_cols(raw):
    """
    Rename and reorder columns of the data. 
    This function is  specific to the Negotiations Dataset and is not a general use function.  

    Takes the raw DataFrame and performs three main operations:
    1. Renames columns to more concise or readable names.
    2. Drops Negotation ID since it is a unique identifier that does not add preditive value
    3. Selects and reorders a subset of these columns for the final DataFrame.

    Parameters:
    raw (DataFrame): The raw DataFrame with original column names.

    Returns:
    DataFrame: A DataFrame with renamed and reordered columns.

    Note:
    The function does not modify the original DataFrame in place.
    """
    # Renaming columns for clarity and consistency
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
                            "Claim ID": "claim_id",
                            "TPA":"TPA",
                            "Facility":"facility",
                            "Day of Patient Date of Birth":"DOB",
                            "Day of D.O.S.": "service_date",
                            "Claim Type": "claim_type",
                            "Max Offer Received Date": "offer_date", 
                            "Max Counter Offer Date": "counter_offer_date"}, inplace=False)

    # Selecting and reordering columns 
    df_edits = df_edits[['claim_id', 'claim_type','NSA_NNSA','split_claim', 'negotiation_type', 
                         'in_response_to', 'claim_status', 
                        'level', 'facility', 'service_date',  
                        'DOB', 'carrier','group_number','plan_funding','TPA', 'TPA_rep', 
                        'billed_amount','negotiation_amount','offer','counter_offer', 'offer_date', 'counter_offer_date', 
                        'deadline', 'decision_date', 'decision'
                        #,'negotiation_id'
                        ]]
    return df_edits


def reorder_cols_in(df, in_str, after=True): 
    """
    Reorders the columns of a DataFrame based on substring matches, moving them either to the beginning or end.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame whose columns need to be reordered.
    - in_str (str, list): The substring(s) to match against column names.
    - after (bool): Whether to move the matched columns after the others. If False, moves them before.
    
    Returns:
    - pd.DataFrame: A DataFrame with reordered columns.
    """
    
    #check if in_str belong to column name (handle both lists and individual arguments)
    if isinstance(in_str, list):
        grouped_cols = []
        for s in in_str:
            grouped_cols += [col for col in df.columns if s in col]
    else:
        grouped_cols = [col for col in df.columns if in_str in col]
    #group remaining columns
    remaining_cols = [col for col in df.columns if col not in grouped_cols]
    
    #order column groups: if after, place grouped cols after remaining cols and vice versa
    if after:
        new_col_order = remaining_cols + grouped_cols
    else: 
        new_col_order = grouped_cols + remaining_cols
    
    return df[new_col_order]

def smart_replace(df, col, needs_replaced, new_value, case=False):
    """
    Searches for strings or lists of strings  contained in a specified 
    column of a DataFrame and replaces those values with a new one

    Parameters:
    df (DataFrame): The DataFrame in which replacements are to be made.
    col (str): The name of the column to perform replacements in.
    needs_replaced (str or list): The pattern(s) to search for in the column. 
                                  Can be a single string or a list of strings.
    new_value: The value to replace the matching entries with.
    case (bool): Whether to consider case sensitivity in pattern matching. 
                 Defaults to False (case insensitive).

    Returns:
    None: The function directly modifies the DataFrame and does not return anything.

    Note:
    The function modifies the DataFrame in place.
    """
    if isinstance(needs_replaced, list):
        for needs_replaced_str in needs_replaced: 
            df.loc[df[col].str.contains(needs_replaced_str, case=case, na=False), col] = new_value 
    else: 
        df.loc[df[col].str.contains(needs_replaced, case=case, na=False), col] = new_value


def mystats(df, nulls_only=False):    
    """
    Generate statistics about missing values and unique values in a DataFrame.

    This function calculates the number of missing values, the percentage of missing 
    values, and the number of unique non-null values for each column in the DataFrame. 
    The results are sorted in descending order by the percentage of missing values.

    Parameters:
    df (DataFrame): The DataFrame to analyze.
    nulls_only (bool): Flag to determine the output.
                       If True, returns stats for all columns.
                       If False, returns stats only for columns with missing values.
                       Defaults to False.

    Returns:
    DataFrame: A DataFrame containing the calculated statistics.

    Notes:
    - The function prints the shape of the input DataFrame.
    """
    # Creating a DataFrame to hold statistics
    stats_df = pd.DataFrame({
                             'nulls': df.isnull().sum(), #count of nulls 
                             'null_pct': round(100*(df.isnull().sum()) / len(df),2), #percentage of nulls
                             'dimensions': df.nunique(), #count of unique non-null values 
                            }).sort_values(by='null_pct', ascending=False)
    #print shape of dataframe 
    print(df.shape)
    # Return the statistics DataFrame if nulls_only is True
    if not nulls_only:
        return stats_df
    else: 
        return stats_df.query('nulls > 0')

def compare_acceptance_rates(df, col, val1, val2, descr1=None, descr2=None):
    """
    Compares the acceptance rates for two groups within a dataframe.

    This function calculates and compares the acceptance rates of two groups 
    in a given dataframe. The groups are identified by the values in val1 and val2. 
    Acceptance rates are calculated as the mean of mapped decision values 
    ('Accepted' to 1, 'Rejected' to 0).

    Parameters:
    - df (DataFrame): The dataframe containing the relevant data.
    - col (str): The name of the column used to identify groups.
    - val1 (str): The value in the column that identifies the first group.
    - val2 (str): The value in the column that identifies the second group.
    - descr1 (str, optional): Description for the first group. Defaults to val1.
    - descr2 (str, optional): Description for the second group. Defaults to val2.

    Prints:
    - A statement comparing the acceptance rates of the two groups.
    """
    # Set descriptions to values if no descriptions are provided
    if descr1 is None: 
        descr1=val1
    if descr2 is None: 
        descr2=val2
     # Map decisions to numeric values for easier computation
    decision_map={'Rejected': 0, 'Accepted':1}
    
    # Compute the mean acceptance rate for each group
    group1_AR=df.query(f"{col}=='{val1}'")['decision'].map(decision_map).mean()
    group2_AR=df.query(f"{col}=='{val2}'")['decision'].map(decision_map).mean()
    
    # Compare the acceptance rates and print the result as the ration of majority group over minority group
    if group1_AR > group2_AR:
        print(f"\n{descr1} claims are {group1_AR/group2_AR:.2f} more likely to be accepted than {descr2} claims")
    else:
        print(f"\n{descr2} claims are {group2_AR/group1_AR:.2f} more likely to be accepted than {descr1} claims")


def get_rows_with_value_count_threshold(df, col, threshold):
    """
    Returns rows from DataFrame where column values exceed a specified count threshold.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to be filtered.

    col : str
        Column name to count unique values.

    threshold : int
        Minimum count for values to be included.

    Returns:
    --------
    pandas.DataFrame
        Filtered DataFrame with rows where column values meet or exceed the threshold.
    """

    # Identify values in 'col' with occurrence above 'threshold'
    col_value_list = df[col].value_counts().loc[lambda x: x >= threshold].index

    # Filter and return DataFrame rows matching the criteria
    return df[df[col].isin(col_value_list)]


def logit1(x, epsilon=2e-3):
    """
    Apply the logit transformation to the data with an adjustment to handle extreme values.

    The function adjusts the input data to avoid extreme values near 0 and 1 using a clipping threshold.

    Parameters:
    x (array-like): The input data.
    epsilon (float, optional): The threshold used for clipping. Defaults to 2e-3.

    Returns:
    array: Transformed data.
    """
    # Avoid extreme values based on value of epsilon passed in
    x_adj = np.clip(x, epsilon, 1 - epsilon)
    return np.log(x_adj / (1 - x_adj))


def logit2(x, epsilon=1e-2):
    """
    Defined this second logit function so it can be passed as an argument into a another (visualization) function
    Apply the logit transformation to the data with an adjustment to handle extreme values.

    This version uses a different clipping threshold compared to logit1 function.

    Parameters:
    x (array-like): The input data.
    epsilon (float, optional): The threshold used for clipping. Defaults to 1e-2.

    Returns:
    array: Transformed data.
    """
    # Avoid extreme values based on value of epsilon passed in
    x_adj = np.clip(x, epsilon, 1 - epsilon)
    return np.log(x_adj / (1 - x_adj))


def yeo_johnson(x):
    """
    Apply the Yeo-Johnson transformation to the data.

    This transformation is used to make data more normally distributed. 
    It's an extension of the Box-Cox transformation that can handle zero and negative values.

    Parameters:
    x (array-like): The input data.

    Returns:
    array: Transformed data.
    """
    # Reshaping the input data to a 2D array as PowerTransformer expects 2D inputs.
    # The '-1' in reshape(-1, 1) infers the length of the array
    reshaped = np.array(x).reshape(-1, 1)
    pt = PowerTransformer(method='yeo-johnson', standardize=True)
    # Fit and transform the data
    return pt.fit_transform(reshaped)


def quantile_transform(x):
    """
    Apply a quantile transformation to the data.

    This transformation converts the variable to follow a specified distribution, 
    such as the normal distribution, thereby making it more suitable for linear models.

    Parameters:
    x (array-like): The input data.

    Returns:
    array: Transformed data.
    """
    # Reshaping the input data to a 2D array as QuantileTransformer expects 2D inputs.
    # The '-1' in reshape(-1, 1) infers the length of the array
    reshaped = np.array(x).reshape(-1, 1)
    qt = QuantileTransformer(output_distribution='normal', n_quantiles=1000, random_state=42)
    # Fit and transform the data
    return qt.fit_transform(reshaped)

def arcsin_sqrt_transform(data):
    """
    Apply the arcsine square root transformation to the data.

    This transformation is suitable for data where values are proportions that lie between 0 and 1. 
    It's commonly used in statistics to stabilize variances of proportions.

    Parameters:
    data (array-like): An array-like object containing proportion data.

    Returns:
    array: Transformed data.
    """
    # Ensures that all values in the data are within the range [0, 1] which is the acceptable range for arcsin
    data = np.clip(data, 0, 1)
    return np.arcsin(np.sqrt(data))

def get_kurtosis(df, col, function):
    """
    Calculate the kurtosis of a specified column in a DataFrame after applying a transformation function.

    Kurtosis is a measure of the "tailedness" of the probability distribution of a real-valued random variable.

    Parameters:
    df (DataFrame): The input DataFrame.
    col (str): The column name in the DataFrame for which to calculate the kurtosis.
    function (callable): The transformation function to apply to the column.

    Returns:
    float: The kurtosis value.
    """
    import scipy.stats as stats
    dfk = df.copy(deep=True)
    return stats.kurtosis(function(df[col]), fisher=True)


