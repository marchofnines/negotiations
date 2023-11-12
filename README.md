
| Column Name        | nulls | null_pct | unique_not_null |
|--------------------|-------|----------|-----------------|
| plan_funding       | 4142  | 45.07    | 2               |
| decision_date      | 1170  | 12.73    | 214             |
| group_number       | 30    | 0.33     | 1886            |
| negotiation_type   | 29    | 0.32     | 3               |
| TPA_rep            | 26    | 0.28     | 648             |
| negotiation_amount | 8     | 0.09     | 4173            |



#### Executive Summary
- Outliers are important for this dataset but extreme/unreasonable outliers will still be removed 

### Claim Type
- Background information: 
    - Each patient visit yields two claims: 
        - HCFA which covers the professional services
        - UB which covers the facility charges
    - HCFA and UB claims pertaining to one same visit are handled compeletely separately and are not even handled at the same time.  
- Observations:
    - I had assumed there would be an equal number of HCFA and UB claims.  However per the visualization this was not the case. In speaking with the team, it turns out that for one patient visit sometimes one claim is accepted outright while the other is sent to negotiations.  This further confirms that even HCFA/UB claims pertaining to the same visit are treated completely separately
    - UB claims are 1.27 more likely to be accepted than HCFA claims

### NSA/NNSA
- Null handling: 
    - There are only 26 rows with missing values.  Attempting to impute values may bring in bias especially given the imbalance of the classes.  Attempting to set missing values to unknown will likely increase model complexity.  Since this is such a small number of rows, we will drop these rows
- Observations:
    - Vast majority of claims are NSA
    - Acceptance Rate is 3.26 times higher for NNSA than it is for NSA


### Split Claim
- Null handling: N/A
- Observations:
    - Majority of claims are Non-Split claims
    - Split claims are 1.21 more likely to be accepted than Non-Split claims

### Negotiation Type
- Null handling: 
    - We plan to impute the 15.99% nulls to see if it helps our model 
- Observations:
    - Majority of claims are negotiated through open negotiation
    - NNSA Negotiation claims are 6.72 more likely to be accepted than Open Negotiation claims
    - Prepayment Negotiation claims are 3.97 more likely to be accepted than Open Negotiation claims

### In Response To
- Data Cleaning/Dimensionality Reduction:
    - Consolidate values related to corrected claims
    - Consolidate values with less than 5 occurences as 'Unknown'
- Observations: 
    - Vast majority of claims are Insurance Initated or initiated by Negotation Letter 
    - Insurance Initiated claims are 4.64 more likely to be accepted than Open Negotiation Notice claims
    - Insurance Initiated claims are 3.35 more likely to be accepted than Negotiation Letter claims

### Claim Status
In discussing this feature with the team, I learned that most statuses are set after the decision is made. This feature is providing us with information that the billing team does not have at the time of the decision. Indeed one of the statuses is Negotiation Acceted! In essence, I consider this to be a form of data leakage and will drop this feature

### Level
- Feature Engineering Observation:
    - This will be an ordinal feature for us
- Other Observations:
    - The majority of our claims are Level 4
    - Level 2 claims are ~1.6 times more likely to be accepted than Level 4/Level 5 claims 

### Facility
- Observations:
   - The acceptance rate varies considerably between facilities which is not immediately intuitive
   - In talking to the team, insurance considers many factors including zip code before extending an offer
   - Garland claims are 1.95 more likely to be accepted than Weatherford claims
   - Matlock facility has low number of claims because it was shut down
   - Benbrook facility has really low numbers but that is because we just recently took over the billing for this facility

### service, decision_date, offer_date, counter_offer_date, deadline
- Feature Engineering:
    - In discussing the team, the importance of the date fields below (if any) is the relative number of days between those dates and the negotiation deadline. Therefore we will convert these columns to equate to the relative number of days prior to the deadline
- Data cleaning:
    - Invalid values for any of these features were  converted to nulls to be imputed later on.  Invalid values include:
        - Negative values
        - decision_days > 50
        - offer_days > 600
- Observations:
    - All date fields have very high variance with outliers. If needed we will revisit the outliers
    - Most deadlines are set to 150 days after the date of service
    - Most decisions are taken within 3 days of the deadline
    - Most offers and counter-offers are extended within a week of the deadline

### Date of Birth
- Feature Engineering:
    - The idea in tracking Date of Birth is that as patients age, they tend to purchase better insurance plans which would potentially result in better offers.  Therefore we will convert this field to year of birth

### Carrier
- Feature Engineering: 
    - We consolidated carrier values that belong together including:
        -typos and variations on the same carriers
        - As explained by the IT team, Carriers starting with ZZZ are not invalid.  They are just duplicates of the value that comes after the ZZZ.  These were entered this way because UB claims cannot reuse the list of carriers from HCFA claims
        - Carrier with less than 4 occurences were consolidated into an 'Unknown' category
- Observations:
    - A large portion of our patients have Cigna and United Health Care insurance even though claims from those carriers tend to have the lowest acceptance rates
    - The highest acceptance rates are for claims from the following carriers:  Baylor Scott and White, Blue Cross, Other and Aetna 
    - Blue Cross Blue Shield claims are 13.30 more likely to be accepted than United Health Care claims
    - Blue Cross Blue Shield claims are 6.40 more likely to be accepted than Cigna claims

### group_number
- Feature Engineering: 
    - There are 1886 dimensions for Group Number
    - Group Numbber represents the agreement an insurance carrier has with an employer. This is why this is a high dimenstionality feature. Group Numbers with less than 5 occurences were consolidated into an 'Unknown' category
- Data cleaning:
    - Null values will also be converted to 'Unknown'
- Observations: 
    - The average acceptance rate against all uncommon group numbers that have less than 40 claims is 15.4%
        - These have been grouped together under group_number = 'Unknown'
    - 884800 claims are 2.18 more likely to be accepted than Unknown claims

### Plan Funding
- Data cleaning: 
    - Plan Funding has a very high percentage of nulls but per the billing team and based on the visualizations, it is an important column. Therefore we will replace nulls with 'Unknown'
- Observations:
    - Majority of plans are self funded
    - FULLY claims are 2.60 more likely to be accepted than SELF claims

### TPA
- Feature Engineering/Dimensionality Reduction:
    - Conslidate variations of the same TPA name
    - Consolidate values that have less than 8 occurences to 'Unknown' 
- Observations: 
    - Most claims are processed through Multiplan and Naviguard
    - Zelis claims are 286.69 more likely to be accepted than Naviguard claims
    - Zelis claims are 8.41 more likely to be accepted than Multiplan claims

### TPA_rep 
- Dimensionality Reduction:
    - Initially we had close to 600 TPA reps 
    - Consolidate values that have less than 5 occurences to 'Unknown' to end up with 134 dimensions
- Data cleaning: N/A
- Observations: 
    - A couple of the TPA reps such as Courtney Kiernan and Debra Caprioti have acceptance rates above 40% whereas others such as Gale Carriedo or Christopher Talley have near 0% acceptance rates despite high volumes

### billed_amount, negotiation_amount, offer, counter_offer
- Data cleaning:
    - After stripping the $ and ',', we found we have some nulls in the negotiation_amount, offer and counter_offer columns. Those will be imputed later on
    - We replaced extreme outliers for counter offer amounts > 150k with null values to be imputed later on
- Observations:
    - Even though we have billed amounts going up to 140_000, most accepted claims have billed and negotiated amounts between 0 and ~20_000 while most offers come in below 8_000
    - There is high correlation between the amount fields
    - There is high correlatoin between the 'days' fields 
    - The highest correlations are between: 
        - negotiation_amount and billed amount: 0.89
        - offer and counter_offer: 0.54
        - negotiation_amount and offer: 0.47
        - billed_amount and the offer: 0.42
        - offer_days and counter_offer_days: 0.82
        - offer_days and decision_days: 0.65
        - counter_offer_days and decision_days: 0.62
    - Ratios: 
        - Ratio of Negotiated Amount to Billed

#### table about means and std dev

#### Permutation Importance Results
f1_weighted
    offer         0.090 +/- 0.004
    carrier       0.023 +/- 0.003
    negotiation_amount      0.015 +/- 0.002
    TPA           0.015 +/- 0.002
    negotiation_type      0.014 +/- 0.002
    counter_offer      0.009 +/- 0.002
    in_response_to      0.009 +/- 0.002
    group_number      0.008 +/- 0.003
    plan_funding      0.007 +/- 0.002
    offer_days      0.005 +/- 0.002
    NSA_NNSA      0.005 +/- 0.002
    level         0.004 +/- 0.001
precision_weighted
    offer         0.098 +/- 0.005
    carrier       0.022 +/- 0.003
    TPA           0.016 +/- 0.003
    negotiation_amount      0.015 +/- 0.003
    negotiation_type      0.015 +/- 0.002
    counter_offer      0.009 +/- 0.002
    in_response_to      0.009 +/- 0.003
    group_number      0.008 +/- 0.003
    plan_funding      0.007 +/- 0.002
    offer_days      0.006 +/- 0.002
    NSA_NNSA      0.005 +/- 0.002
    level         0.004 +/- 0.002
recall_weighted
    offer         0.080 +/- 0.004
    negotiation_amount      0.016 +/- 0.002
    carrier       0.016 +/- 0.003
    TPA           0.013 +/- 0.002
    negotiation_type      0.010 +/- 0.002
    counter_offer      0.010 +/- 0.002
    group_number      0.007 +/- 0.002
    in_response_to      0.006 +/- 0.002
    plan_funding      0.006 +/- 0.002
    offer_days      0.004 +/- 0.002
    NSA_NNSA      0.004 +/- 0.002
    level         0.004 +/- 0.001

#### Imputation:
- decision_days, offer_days, counter_offer_days, service_days, neg_amount, counter_offer, offer: KNNImputer
- Categorical:
   - negotiation_type:  Simple Imputer
   - plan_funding: N/A because was 53% null
   - in_response_to, TPA, TPA rep, carrier: N/A because we combined other existing values
   - claim_type, NSA/NNSA: Nothing to impute

#### New imputation
-'claim_type': N/A no nulls
- 'carrier': N/A no nulls
- 'negotiation_type': We plan to impute the 15.99% 
- 'facility': N/A no nulls
- TPA: N/A 
- level: N/A
- offer, counter_offer, negotiation_amount:  KNNImputer
- service days, counter_offer_days, offer_days: Mean
- YOB: N/A




Jordan questions
- split claim acceptance rate does not match what I know about split claims 

- removing values prior to 2023 (process and reimbursement% changes)
- list of fields
- remove outliers (consider starting with ratios)
- //duplicates
- go through charts for each field (Try to summarize highlights before meeting)
- Data leakage concern:  When i split the data, should I try to keep pairs of claims together OR create two datasets and two models 
- What are some of the reasons counter offer have such a high variance
- additional viz?


Savio questions
- Interpreting coefficients! 
- Interpreting PRC AUC and F1 Weighted 
- Interpreting counterfactuals 
- framing the problem using LinearRegression TODO! 
- looking at confidence intervals !! TODO!
- SMOTE, Impute, Build Pipes
- When i split the data, try to keep pairs of claims together 
    -  OR create two datasets and two models 
- how to impute negotiation_amount given a 0.89 correlation with billed_amount and 0.47 with offer
    - also offer_days and counter_offer_days given 0.76 correlation
    - also show number of nulls and ask how to impute 
- negative handling of number of days 
- outliers for number of days


Despite this, the team usually tries to get insurance to accept the same offer to negotiation amount on both claims. 

    - NSA claims can be brought to an Independent Dispute Resolution (a hearing/court) with CMS
    - NNSA claims get disputed at the State level

| Column Name        | nulls | null_pct | unique_not_null |
|--------------------|-------|----------|-----------------|
| plan_funding       | 4142  | 45.07    | 2               |
| decision_date      | 1170  | 12.73    | 214             |
| group_number       | 30    | 0.33     | 1886            |
| negotiation_type   | 29    | 0.32     | 3               |
| TPA_rep            | 26    | 0.28     | 648             |
| negotiation_amount | 8     | 0.09     | 4173            |

