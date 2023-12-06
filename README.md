## Emergency Medicine Negotiations: Will the Provider Accept the Insurance Company Offer? 

### Files Included
#### Jupyter Notebooks
- [1. Exploratory Data Analysis](https://github.com/marchofnines/negotiations/blob/main/1_EDA_capstone.ipynb)
- [2. Feature Engineering](https://github.com/marchofnines/negotiations/blob/main/2_FE_capstone.ipynb)
- [3. Cross-Validation](https://github.com/marchofnines/negotiations/blob/main/3_CV_capstone.ipynb)
- [4. Cross-Validation with Hyperparameter Tuning](https://github.com/marchofnines/negotiations/blob/main/4_HT_capstone.ipynb)
- [5. Model Evaluation](https://github.com/marchofnines/negotiations/blob/main/5_ME_capstone.ipynb)

#### data folder
- negotiations.csv (dataset)

#### helpers module
- __init__.py - Package initialization
- my_imports.py - import libraries
- helpers/plot.py - helper functions for plotting
- helpers/preprocessing.py - helper functions for preprocessing/EDA
- helpers/reload.py - helper function for reloading .py files when modified
- helpers/tools.py - helper functions for modeling 
- helpers/transformers.py - Classes of numerical transformers

#### models folder
- cross_val folder
    - models and results from feature engineering and cross-validation
- hyperparam_tuning folder
    - models and results from hyperparameter tuning
- README.md
#### saved_dfs folder
- saved_dfs/preprocessed_negotiations_df.csv - cleaned dataframe
- saved_dfs/sfm_scores.csv - scores of scores vs. n features selected (used in Feature Engineering)



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

- #### counter_offer_amount:  Final counter offer made by provider in final round of negotiation

- #### decision: This is the Target and indicates whether the Provider Accepted (Positive Class) or Rejected the offer

### Model Evaluation: 
- We will use the F1 scoring metric as our primary performance metrics because we want to strike a balance between minimizing False Positives and minimizing False Negatives: 
    - We do not want to accept offers that our team would normally be rejecting.  This means that we want to minimize False Positives i.e. maximize Precision/PPV which is True Positives divided by the Predicted Positives TP/(TP+FP).  Precision is intuitively the ability of the classifier not to label as positive a sample that is negative.
    - At the same time we do not want to reject offers that our team would normally be accepting.  This means that we want to minimze False Negatives i.e. maximize Recall/Sensitivity/TPR which is the proportion of true positives divided by the total number of truly postive classifications (TP/(TP+FN). Recall is intuitively the ability of the classifier to find all the positive samples.
- And since we have an Imbalanced Dataset where only ~12% of the claims are accepted, we will use the weighted version of F1. F1 weighted score weighs the importance of each class according to its presence in the dataset. This avoids giving undue influence to the performance on the Rejected class just because it is more common. Specifically in our case, if our model has a good performance on Rejections, it could mask poor performance on Acceptances in an unweighted F1 score. The weighted F1 score gives more importance to the performance on Acceptances, proportional to their presence in the dataset.
- Additionally we will use the Precision-Recall Curve to find the best trade-off between Precision and Recall 
- While we do want to balance Precision and Recall, False Negatives are slightly worst for us than False Positives.  Therefore we have a slight preference for Precision over Recall

### Results: 
- Ability to predict decisions along with a level of confidence (Not available with all models)
- Identify the most important factors that result in an accepted claim (Not available with all models)

### Summary of EDA Findings
- Removed rows containing invalid data as well as a small percentage of rows containing nulls 
- Removed claim_id, negotiation_id and claim_status columns
- Plan to combine rare categories and cross-validate using different size groupings
- Converted date fields into numerical fields that show the difference in number of days relative to the deadline
- Included ratios of the amount fields so that we can test to see whether ratios perform better than absolute numbers 
- Explored various transformations of numerical features which will have to be tested out during Cross-Validation: 
    - Log Transforms and Quantile Transforms for the day and amount features
    - Logit and Quantile Transforms for the ratios

### Summary of Feature Engineering Findings
- Dummy Classifier score: 0.84
- We will be replacing the amount fields billed_amount, negotiation_amount, offer, counter_offer, with 3 ratio columns as we saw that this reduces multicollinearity and improves the performance across most model types
- Permutation Importance showed that offer_to_neg and offer_to_counter_offer are the most important features across most models
- Using simple models without fully built transformers, and without hyper parameter tuning we see slightly overfit models with F1 weighted scores ranging from ~0.93 to ~0.94
- Our Cross-Validation against different values of n features showed that RandomForest and GradientBoosting looked the most promising, especially between n=10 to n=25 features, but continued to do well up until ~n=41
- Regularization stabilizes around 1/C=0.1 (i.e. C=10)

### Summary of Cross-Validation Findings
- In addition to not having the benefit of Hyperparameter Tuning in this section, most of the tests resulted in overfit models so we  take these results with a grain of salt and leave options open for our transformer/pipeline design during Hyperparameter Tuning
- We do plan on exploring adding Oversampling/Undersampling to our Pipeline, as well as a Transformer to combine rare values for Categorical columns but since these functions depend heavilty on their parameters we will explore them when we do Hyperparameter Tuning
- We plan to start the next section (HyperParameter Tuning) with Yeo-Johnson Transformation and TargetEncoder for all our tests but we will still try multiple scalers.  
- RFE and SequentialFeatureSelector run much slower than SelectFromModel and with default settings ended up giving us similar results.  Since Execution Time will become increasingly important for HyperParameter Tuning, we will stick with SelectFromModel
- We did not see an especially high Precision Results for the RidgeClassifier so we may drop it early in the next section
- There were some features that we considered dropping in Feature Engineering, however we will let our SelectFromModel function handle that for us in the next section

### Summary of Hyper Parameter Tuning Findings
- The best non-overfit model for Logistic Regression was from set 7 and had a test f1 weighted score of 0.9502 with mean fit time 8.6s 
- The best non-overfit model for Gradient Boost Classifier was from set 8 and has a test f1 weight score of 0.9515 and a mean fit time of 13.09s. It also had higher Cross-Validation Results than the Logistic Regression Model (0.944 vs 0.937)

### Summary of Model Evaluation Findings
- We showed the coefficients of our best Logistic Regression Model even though it was a close 2nd overall.  This was done to show the relative importance of the features.  Ability to interpret the features was limited due to the fact that numerical values underwent a yeo-johnson transformation
- We also showed the confusion matrix and Precision-Curves for our best Logistic Regression Models and our best overall (GBC) model.  
- For both models, we also showed what happens when we increase the probability threshold to reduce the number of FNs which is something the Billing Department was interested in.