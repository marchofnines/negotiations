## Emergency Medicine Negotiations: Will the Provider Accept the Insurance Company Offer? 
[Link to er_negotiations.ipynb Jupyter Notebook](https://github.com/marchofnines/er_negotiations/blob/main/er_negotiations.ipynb)


### Context
When an Out of Network Health Care Provider bills insurance, Insurance companies and Third Party Administrators (TPAs) do not automatically reimburse providers.  Instead, they typically offer a reduced reimbursement amount.  Many of these claims end up in negotiations between Insurance/TPAs and providers.

### Business Objective
To save time and resources for the Billing Department by automating the majority of Negotation Decisions and simplying the Decision Workflow.  

### Business Question
Given an Insurance Claim, should TotalCare Accept or Reject the Offer extended by the the TPA or Insurance Carrier? And with what level of confidence can these decisions be made?  

### Dataset
- #### negotiation_id: identifies a unique negotation. One claim can have more than one negotiation 
- #### claim_id: Unique identifier representing a claim arising from a patient visit
- #### claim_type
    - Each patient visit yields two claims: 
        - HCFA claims cover the professional services
        - UB claims cover the facility charges
        - HCFA and UB claims pertaining to one same visit are handled compeletely separately and are not even handled at the same time.  

- #### NSA_NNSA: Is this a ‘No Surprise Act’ claim or not?
    - Under NSA, patients are only responsible for their in-network costs even if they go to an out of network provider.  This type of claim is bound by more restrictions than claims not under NSA.

- #### split_claim:  Has the claim been split into smaller claims? 
    - Insurance handles these claims as multiple smaller claims

- #### negotiation_type
    - Prepayment negotiations: Negotiation takes place before insurance processes the claim.
    - Open Negotiation: Only applies to NSA.  Takes place after insurance processes the claim
    - NNSA Negotiation: NNSA negotiation that takes place after insurance processes the claim 

- #### in_response_to:  What initiated the claim? 
    - Insurance Initiated: 
        - Insurance Initiated         
    - Provider Initiated: 
        - Open Negotiation Notice 
        - Negotiation Letter (NNSA)
        - Verbal Request           
        - Reconsideration Letter   
        - Corrected Claim          

- #### claim_status:  Claim status at the end of negotiation

- #### level: Indicates complexity of medical decision-making and amount of time spent with the patient
    - Level 2:  Low severity and low complexity
    - Level 3: Moderate severity and complexity
    - Level 4: High complexity but not severe enough to require immediate life saving intervention
    - Level 5: Severe or life-threatening compexity

- #### facility: Location where the patient was seen
    - This column was included because offer amounts may be based on zip code

- #### service_date: Date patient was seen 

- #### decision_date: Date provider accepted or rejected the claim

- #### offer_date: Date Insurance or TPA provider extended the final offer

- #### counter_offer_Date:  Date of final counter offer

- #### deadline: Last day to complete the negotiation

- #### date of birth: Patient date of birth
    - The idea in tracking Date of Birth is that as patients age, they tend to purchase better insurance plans which would potentially result in better offers

- #### carrier: Insurance carrier

- #### group number:  Represents group policy which determines benefits and coverage and reimbursement rates

- #### plan_funding: Is plan Self or Fully funded? 
    - Self Funded: Employer is responsible for covering employee healthcare claims typically via a TPA.  Not subject to State Insurance regulation and therefore these plans tend to offer more flexibility
    - Fully Funded: Employer pays fixed premium to insurance carriers and carriers assume the risk and financial responsibility for covering the claims.  

- #### TPA: Third Party Administrator processing the claims and conducting negotiations on carrier's behalf

- #### TPA_rep:  Name of Representative at TPA who is managing the negotiation

- #### billed_amount:  Amount billed by provider

- #### negotiation_amount:  Billed amount minus Disallowed Amounts
    - Insurance companies take the billed amount and deduct disallowed amounts which include co-insurance, deductible amounts as well as amounts for services not covered by the insurance plan 

- #### offer_amount:  Maximum offer extended by insurance/TPA after multiple rounds of negotiation

- #### counter_offer_amount:  Final counter offer made by provider in final round of negotiation??

- #### decision: This is the Target and indicates whether the Provider Accepted (Positive Class) or Rejected the offer

### Model Evaluation: 
- We will use the F1 scoring metric as our primary performance metrics because we want to strike a balance between minimizing False Positives and minimizing False Negatives: 
    - We do not want to accept offers that our team would normally be rejecting.  This means that we want to minimize False Positives i.e. maximize Precision/PPV which is True Positives divided by the Predicted Positives TP/(TP+FP).  Precision is intuitively the ability of the classifier not to label as positive a sample that is negative.
    - At the same time we do not want to reject offers that our team would normally be accepting.  This means that we want to minimze False Negatives i.e. maximize Recall/Sensitivity/TPR which is the proportion of true positives divided by the total number of truly postive classifications (TP/(TP+FN). Recall is intuitively the ability of the classifier to find all the positive samples.
- And since we have an Imbalanced Dataset where only ~12% of the claims are accepted, we will use the weighted version of F1. F1 weighted score weighs the importance of each class according to its presence in the dataset. This avoids giving undue influence to the performance on the Rejected class just because it is more common. Specifically in our case, if our model has a good performance on Rejections, it could mask poor performance on Acceptances in an unweighted F1 score. The weighted F1 score gives more importance to the performance on Acceptances, proportional to their presence in the dataset.
- Additionally we will use the Precision-Recall Curve to find the best trade-off between Precision and Recall and then select the optimal probability threshold based on that 
- While we do want to balance Precision and Recall, False Negatives are slightly worst for us than False Positives.  Therefore we have a slight preference for Precision over Recall

### Results: 
- Ability to predict decisions along with a level of confidence (Not available with all models):
- Summarize the most important factors that result in an accepted claim
- Provide recommendations and next steps 
- Provide counterfactuals 


### Findings for Data Cleaning and Understanding
- #### claim_type
    - I had assumed there would be an equal number of HCFA and UB claims.  However per the visualization this was not the case. In speaking with the team, it turns out that some of the claims are automatically accepted, while others go to negotiation. This further cements the concept that even HCFA/UB claims pertaining to the same visit are treated completely separately
    - UB claims are 1.27 more likely to be accepted than HCFA claims

- #### NSA_NNSA: 
    - Null handling: 
        - There are only 26 rows with missing values.  Attempting to impute values may bring in bias especially given the imbalance of the classes.  Attempting to set missing values to unknown will likely increase model complexity.  Since this is such a small number of rows, we will drop these rows
    - Observations:
        - Vast majority of claims are NSA
        - Acceptance Rate is 3.26 times higher for NNSA than it is for NSA

- #### split_claim:  Has the claim been split into smaller claims? 
    - Majority of claims are Non-Split claims
    - Split claims are 1.21 more likely to be accepted than Non-Split claims

- #### negotiation_type
    - Null handling: 
        - There are ~16% nulls in this column.  However, per the team any claims that have a null value in this column that are also NNSA claims should be labeled as 'NNSA Negotiation'
        - Drop the remaining 29 nulls.  Attempting to impute these values may bring in bias especially given the imbalance of the classes. Attempting to set missing values to unknown will likely increase model complexity. Since this is such a small number of rows, we will drop these rows

    - Observations:
        - Majority of claims are negotiated through open negotiation
        - NNSA Negotiation claims are 6.72 more likely to be accepted than Open Negotiation claims
       - Prepayment Negotiation claims are 3.97 more likely to be accepted than Open Negotiation claims

- #### In Response To
    - Data Cleaning/Dimensionality Reduction:
        - Consolidated values with typos and similar values 
        - Consolidated values with less than 32 occurences as 'Unknown'
    - Observations: 
        - Vast majority of claims are Insurance Initated or initiated by Negotation Letter 
        - Insurance Initiated claims are 4.66 more likely to be accepted than Open Negotiation Notice claims
        - Insurance Initiated claims are 3.36 more likely to be accepted than Negotiation Letter claims

- #### Claim Status
    - In discussing this feature with the team, I learned that most statuses are set after the decision is made.  This feature is providing us with information that the billing team does not have at the time of the decision. Indeed one of the statuses is Negotiation Acceted!  In essence, this is causing data leakage so we will drop this feature

- ### Level
    - The majority of our claims are Level 4
    - Level 2 claims are ~1.6 times more likely to be accepted than Level 4/Level 5 claims 

- ### Facility
   - The acceptance rate varies considerably between facilities which is not immediately intuitive
   - In talking to the team, insurance considers many factors including zip code before extending an offer
   - Garland claims are 1.93 more likely to be accepted than Weatherford claims
   - Matlock facility has low number of claims because it was shut down
   - Benbrook facility has really low numbers but that is because we just recently took over the billing for this facility

- ### service, decision_date, offer_date, counter_offer_date, deadline
    - Feature Engineering:
        - In discussing the team, the importance of the date fields below (if any) is the relative number of days between those dates and the negotiation deadline. Therefore we will convert these columns to equate to the relative number of days prior to the deadline
    - Data cleaning:
        - Invalid values for any of these features were  converted to nulls to be imputed later on.  Invalid values include:
            - Negative values
            - decision_days > 50
            - offer_days and counter_offer_days > 200
    - Observations:
        - All date fields have very high variance with outliers. If needed we will revisit the outliers
        - Most deadlines are set to 150 days after the date of service
        - Most decisions are taken within 3 days of the deadline
        - Most offers and counter-offers are extended within a week of the deadline

- ### Date of Birth
    - The idea in tracking Date of Birth is that as patients age, they tend to purchase better insurance plans which would potentially result in better offers.  Therefore we converted this field to year of birth

- ### Carrier
    - Feature Engineering: 
        - We consolidated carrier values that belong together including:
            -typos and variations on the same carriers
            - As explained by the IT team, Carriers starting with ZZZ are not invalid.  They are just duplicates of the value that comes after the ZZZ.  These were entered this way because UB claims cannot reuse the list of carriers from HCFA claims
            - Carrier with less than 17 occurences were consolidated into an 'Unknown' category
    - Observations:
        - A large portion of our patients have Cigna and United Health Care insurance even though claims from those carriers tend to have the lowest acceptance rates
        - The highest acceptance rates are for claims from the following carriers:  Baylor Scott and White, Blue Cross, Other and Aetna 
        - Blue Cross Blue Shield claims are 13.30 more likely to be accepted than United Health Care claims
        - Blue Cross Blue Shield claims are 6.40 more likely to be accepted than Cigna claims

- ### group_number
    - Feature Engineering/Data Cleaning:
        - The 30 Null values will  be converted to 'Unknown'
        - There are 1893 dimensions for Group Number. Group Numbers with less than 20 occurences were consolidated into an 'Unknown' category
    - Observations: 
        - The average acceptance rate against all uncommon group numbers that have less than 40 claims is 15.4%
            - These have been grouped together under group_number = 'Unknown'
        - 884800 claims are 2.64 more likely to be accepted than Unknown group numbers

- ### Plan Funding
    - Data cleaning: 
        - Plan Funding has a very high percentage of nulls but per the billing team and based on the visualizations, it is an important column. Therefore we will replace nulls with 'Unknown'
    - Observations:
        - Majority of plans are self funded
        - FULLY claims are 2.61 more likely to be accepted than SELF claims

- ### TPA
    - Feature Engineering/Dimensionality Reduction:
        - Conslidate variations of the same TPA name
        - Consolidate values that have less than 10 occurences to 'Unknown' 
    - Observations: 
        - Most claims are processed through Multiplan and Naviguard
        - Zelis claims are 286.69 more likely to be accepted than Naviguard claims
        - Zelis claims are 8.41 more likely to be accepted than Multiplan claims

- ### TPA_rep 
    - Dimensionality Reduction:
        - Initially we had close to 600 TPA reps 
        - Consolidate values that have less than 15 occurences to 'Unknown' 
    - Observations: 
        - A couple of the TPA reps such as Courtney Kiernan and Debra Caprioti have acceptance rates above 40% whereas others such as Gale Carriedo or Christopher Talley have near 0% acceptance rates despite high volumes

- ### billed_amount, negotiation_amount, offer, counter_offer
    - Data cleaning:
        - We replaced extreme outliers for counter offer amounts > 150k with null values to be imputed later on
        - We dropped rows with invalid aomunts such as:
            - offer > negotiation_amount
            - offer > counter_offer
            - counter_offer > negotiation_amount

    - Observations:
        - Even though we have billed amounts going up to 140_000, most accepted claims have billed and negotiated amounts between 0 and ~20_000 while most offers come in below 8_000
        - There is high correlation between the amount fields
        - There is high correlatoin between the 'days' fields 
        - The highest correlations are between: 
            - negotiation_amount and counter_offer: 0.94
            - negotiation_amount and billed_amount: 0.90
            - offer and counter_offer: 0.55
            - negotiation_amount and offer: 0.47
            - billed_amount and the offer: 0.42
            - offer_days and counter_offer_days: 0.82
            - offer_days and decision_days: 0.65
            - counter_offer_days and decision_days: 0.63
        - The table below displays information about the amounts as well as the ratios we calculated: 

|       | billed_amount | negotiation_amount |   offer | counter_offer | neg_to_billed | offer_to_billed | offer_to_neg | counter_offer_to_offer |   
|-------|---------------|--------------------|---------|---------------|---------------|-----------------|--------------|------------------------|
| count | 6705.00       | 6705.00            | 6705.00 | 6705.00       | 6705.00       | 6705.00         | 6705.00      | 6705.00                | 
| mean  | 13790.90      | 12359.41           | 1684.33 | 9015.30       | 0.92          | 0.13            | 0.14         | 16.85                  |
| std   | 10649.50      | 9743.23            | 2522.86 | 7480.28       | 0.17          | 0.15            | 0.16         | 30.94                  | 
| min   | 175.00        | 175.00             | 3.00    | 60.00         | 0.01          | 0.00            | 0.00         | 1.00                   | 
| 25%   | 6878.00       | 6103.00            | 287.00  | 4260.00       | 0.96          | 0.03            | 0.03         | 4.15                   | 
| 50%   | 10774.00      | 9218.00            | 836.00  | 6638.00       | 1.00          | 0.07            | 0.08         | 7.95                   | 
| 75%   | 16471.00      | 15248.00           | 1862.00 | 11118.00      | 1.00          | 0.15            | 0.17         | 21.82                  | 
| max   | 143038.00     | 143038.00          | 36190.00| 100127.00     | 1.00          | 0.95            | 0.95         | 1097.80                | 


#### Permutation Importance Results
f1_weighted
    offer         0.087 +/- 0.005
    carrier       0.021 +/- 0.003
    counter_offer      0.017 +/- 0.003
    negotiation_type      0.010 +/- 0.003
    TPA           0.009 +/- 0.003
    negotiation_amount      0.008 +/- 0.003
    plan_funding      0.006 +/- 0.002
    counter_offer_days      0.006 +/- 0.002
    claim_type      0.006 +/- 0.003

#### KNN
f1_weighted
    offer         0.019 +/- 0.002
    carrier       0.009 +/- 0.002
    plan_funding      0.008 +/- 0.003
    TPA           0.005 +/- 0.002
    in_response_to      0.003 +/- 0.001

#### RandomForest
 offer         0.059 +/- 0.005
    TPA           0.013 +/- 0.003
    plan_funding      0.011 +/- 0.003
    carrier       0.008 +/- 0.003
    level         0.004 +/- 0.001
    NSA_NNSA      0.003 +/- 0.001
    offer_days      0.003 +/- 0.001

#### SVM 
f1_weighted
    offer         0.111 +/- 0.005
    negotiation_amount      0.010 +/- 0.002
    counter_offer      0.009 +/- 0.003
    counter_offer_days      0.003 +/- 0.002
    TPA           0.003 +/- 0.001
    level         0.003 +/- 0.001


# HalvingRandom
n_candidates = 'exhaust'
factor=3
resource = n_samples 
max_resources = auto
min_resources = smallest #because exhaust is not available unless i pick a number of n_candidates
scoring:  No dicts. Just string.  
refit: bool: no string.  cause yeah there's just one scoring metric 
rs, n_jobs, verbose

### Factors to try
#### Dimensionality
#### SimpleImputerStrategy
#### KNNImputer
#### Keep more features, reduce features
#### lgrsaga, lgrelasticnet

## Sets
### Set 1:
- Numerical Imputer: SimpleImputer/Mean
- PolynomialFeatures
- Oversampler
- Undersampler
- StandardScaler
- Selector: SelectFromModel/Logistic 

### Set 2: 
   - Numerical Imputer: SimpleImputer/most-frequent
   - PolynomialFeatures
   - Oversampler
   - Undersampler
   - StandardScaler
   - Selector: SelectFromModel/Logistic 

### Set 3: 
   - Numerical Imputer: SimpleImputer/most-frequent
   - PolynomialFeatures
   - Oversampler
   - Undersampler
   - RobustScaler
   - Selector: SelectFromModel/Logistic 

### Set 4: 
   - Numerical Imputer: SimpleImputer/most-frequent
   - PolynomialFeatures
   - Oversampler
   - Undersampler
   - MaxAbsScaler
   - Selector: SelectFromModel/Logistic 

### Scaler selected: RobustScaler
### Set 5: 
   - Numerical Imputer: KNNImputer
   - PolynomialFeatures!!!!!!!!!???????
   - Oversampler
   - Undersampler
   - RobustScaler
   - Selector: SelectFromModel/Logistic 

### Set 6: 
   - Numerical Imputer: IterativeImputer
   - PolynomialFeatures
   - Oversampler
   - Undersampler
   - RobustScaler
   - Selector: SelectFromModel/Logistic 





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



"""{
        'est_tuple': ('dtree', DecisionTreeClassifier(random_state=rs)),
        'est_params': {
            'dtree__criterion': ['gini', 'entropy', 'log_loss'],
            'dtree__splitter': ['best', 'random'],
            'dtree__max_depth': randint(3, 31),
            'dtree__max_features': randint(2,5), # num features to consider when looking for best split
            'dtree__min_samples_split': uniform(0.1, 0.7),# overfit at lower values 
            'dtree__min_samples_leaf': uniform(0.1, 0.7), #min samples required to be a leaf node | overfit at lower values |     
            'dtree__max_leaf_nodes': randint(2,100), 
            'dtree__ccp_alpha': uniform(0, 0.05),
            'dtree__min_impurity_decrease': loguniform(1e-3, 1e-1), #uniform(0, 0.2),  
            'dtree__class_weight': class_weights,
        }
    },
    {
        'est_tuple': ('rf', RandomForestClassifier(random_state=rs, n_jobs=3)),
        'est_params': {
            'rf__n_estimators': randint(100,1000),
            'rf__criterion': ['gini', 'entropy', 'log_loss'],
            'rf__max_depth': [None] + list(range(3, 31)),  #randint(2, 25),
            #'rf__max_features': randint(2,5), # num features to consider when looking for best split
            'rf__max_features': [None, 'sqrt', 'log2'],
            'rf__min_samples_split': uniform(0.01, 0.2), # overfit at lower values 
            'rf__min_samples_leaf': uniform(0.01, 0.2), #min samples required to be a leaf node | overfit at lower values |     
            'rf__min_weight_fraction_leaf': uniform(0, 0.5),
            'rf__max_leaf_nodes': [None] + list(range(7, 900)),  #randint(2,70), 
            'rf__ccp_alpha':  loguniform(1e-4, 1e-1), #uniform(0, 0.07),
            'rf__min_impurity_decrease': uniform(0, 0.2), #stats.loguniform(1e-3, 2e-1), 
            'rf__class_weight': [None, 'balanced', 'balanced_subsample'] + class_weights,
            'rf__bootstrap': [True],
            'rf__oob_score': [True, False],
            'rf__max_samples': uniform(0,0.5),
        }
    },"""

     {
        'est_tuple': ('lgr_saga_elastic', LogisticRegression(verbose=0, random_state=rs, n_jobs=3)), 
        'est_params': {
            'lgr_saga_elastic__penalty': ['elasticnet'],
            'lgr_saga_elastic__C':  loguniform(1e-1, 1e2), #larger C stronger regularization, smaller C increases penalty --> simpler model, smaller coefficients
            'lgr_saga_elastic__solver': [ 'saga'], 
            'lgr_saga_elastic__max_iter': [7000],
            'lgr_saga_elastic__class_weight': class_weights,
            'lgr_saga_elastic__fit_intercept': [True, False],
            'lgr_saga_elastic__multi_class': ['auto'],  # ovr, multinomial
            'lgr_saga_elastic__l1_ratio': [ 0.3, 0.4, 0.5, 0.6]
        }
    },