## Emergency Medicine Insurance Claim Negotiations: Will the Provider Accept the Insurance Company Offer? 

 ### Files Included
#### Jupyter Notebooks
- [1. Exploratory Data Analysis](https://github.com/marchofnines/negotiations/blob/part2/1_Exploratory_Data_Analysis.ipynb)
- [2. Feature Engineering](https://github.com/marchofnines/negotiations/blob/part2/2_Feature_Engineering.ipynb)
- [3. Cross-Validation with Hyperparameter Tuning LGR, RF, GBC](https://github.com/marchofnines/negotiations/blob/part2/3_HyperParameter_Tuning_LGR_RF_GBC.ipynb)
- [4. Cross-Validation with Hyperparameter Tuning MLPClassifier](https://github.com/marchofnines/negotiations/blob/part2/4_HyperParameter_Tuning_MLPClassifier.ipynb)
- [5. Cross-Validation with Hyperparameter Tuning KerasClassifier](https://github.com/marchofnines/negotiations/blob/part2/5_HyperParameter_Tuning_KerasClassifier.ipynb)
- [6. Model Evaluation](https://github.com/marchofnines/negotiations/blob/part2/6_Model_Evaluation.ipynb)

#### data folder
- negotiations.csv (dataset)

#### helpers package
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

#### saved_dfs folder
- saved_dfs/preprocessed_negotiations_df.csv - cleaned dataframe
- saved_dfs/sfm_scores.csv - scores of scores vs. n features selected (used in Feature Engineering)


### Business Context
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
#### Choice of Metric
- We will use the F1 scoring metric as our primary performance metrics because we want to strike a balance between minimizing False Positives and minimizing False Negatives: 
    - We do not want to accept offers that our team would normally be rejecting.  This means that we want to minimize False Positives i.e. maximize Precision/PPV which is True Positives divided by the Predicted Positives TP/(TP+FP).  Precision answers the question: Of all the instances predicted as positive, how many were actually positive?.
    - At the same time we do not want to reject offers that our team would normally be accepting.  This means that we want to minimze False Negatives i.e. maximize Recall/Sensitivity/TPR which is the proportion of true positives divided by the total number of truly postive classifications (TP/(TP+FN). Recall answers the question: Of all the actual positive instances, how many did the model successfully identify?"

#### Variations of the Metric
- There are three variations of the F1 score that we considered:
    - Micro-averaged: all samples equally contribute to the final averaged metric
    - Macro-averaged: all classes equally contribute to the final averaged metric
    - Weighted-averaged: each classes’s contribution to the average is weighted by its size
- Since we have an Imbalanced Dataset where only ~12% of the claims are accepted, we will use the weighted version of F1. This avoids giving undue influence to the performance on the Rejected class just because it is more common. Specifically in our case, if our model has a good performance on Rejections, it could mask poor performance on Acceptances in an unweighted F1 score. The weighted F1 score gives more importance to the performance on Acceptances, proportional to their presence in the dataset.

### Results and Interpretation: 
Results and Interpretation will include: 
- Model that predicts decisions along with a level of confidence (confidence thresholds not available with all models)
- Identify the most important factors that result in an accepted claim (Not available with all models)
- Confusion Matrix and Precision Recall Curve
- Lift and Cumulative Gain Curves
- Counter Factuals

### Summary of Feature Engineering Findings (Notebook 2)
- Dummy Classifier f1 weighted score: 0.588
- We confirmed that it would be helpful to replace the amount fields `billed_amount`, `negotiation_amount`, `offer`, `counter_offer`, with 3 ratio columns as we saw that this reduces multicollinearity and improves the performance across most model types
- Permutation Importance showed that offer_to_neg and offer_to_counter_offer are the most important features across most models
- The `level` feature tends to get excluded in feature selection and tests without it are perhaps slightly better but still mixed.  We will let Feature Selection handle its inclusion/exclusion for us when we do Cross-Validation
- Our Cross-Validation against different values of n features showed that RandomForest and GradientBoosting looked the most promising, especially between n=17 to n=48 features, 
- Regularization stabilizes around 1/C=1 (i.e. C=1)
- Using models with default parameters we see mostly overfit models with test F1 weighted scores reaching up to approximately ~0.93

### Summary of Cross-Validation with HyperParameter Tuning Findings for LGR, RF and GBC  (Notebook 3)
- The best non-overfit model for Logistic Regression was from set 7 and had a test f1 weighted score of 0.9502 with mean fit time 8.6s 
- RandomForestClassifier consistently underperformed GradientBoostClassifier, so we dropped it after Set2
- The best non-overfit model for GradientBoostClassifier was from set 8 and has a test f1 weight score of 0.9515 and a mean fit time of 13.09s. It also had higher Cross-Validation Results than the Logistic Regression Model (0.944 vs 0.937)
- We note that not only did GradientBoostClassifier score slightly higher than LogisticRegression, it did so with Polynomial Degree = 1 which equates to a simpler model

### Summary of Findings for Cross-Validation / Hyperparameter Tuning for MLPClassifier (Notebook 4)
- Using MLPClassifier allowed to leverage Neural Networks while using an sklearn classifier
- We had some overfitting in some of our sets which we were able to address by reducing layer dimensions, number of layers, alpha and learning rate among other parameters
- MLPClassifier's best sets were Sets 5 and 7 with a test score of ~0.931.  Set 5 had no polynomialfeatures and used 3 layers (20,15,10) whereas Set 7 used PolynomialFeatures degree 2 and had 2 layers (64,32)
- Our GradientBoostClassifier model from notebook 3 is still the top performer

### Summary of Findings for Cross-Validation / Hyperparameter Tuning for KerasClassifier (Notebook 5)
### Summary of Findings for Cross-Validation with Hyperparameter Tuning
- In sets 3 and 4, we finally were able to find models that are not overfit, even though this was at the cost of a reduction in score.  The best non-overfit f1 weighted score was from set 4 and was 0.894.
- To arrive at this model, we did a compbination of reducing the number of hidden layers and the neurons per layer.  We also increased regularization, dropout rate and batch size while decreaseing the learning rate and the number of epochs and introduced early stopping with a Learning Rate Schedule.
- Next steps would include:
  - Looking into why the F1Score layer was not properly built for sets 3 and 4 only
  - Use class weights parameters to see if this improves performance.  This makes sense since we have an imbalanced dataset.  
- The best model remains our GradientBoostClassifier with a score of 0.9515

### Summary of Model Evaluation Findings (Notebook 6)
#### Overview 
- We performed HyperParameter Tuning on 5 models: `LogisticRegression`, `RandomForest`, `GradientBoostClassifier`, `MLPClassifier` and `KerasClassifier`.  While `LogisticRegression` scored slightly less than `GradientBoostClassifier`,  we still take a look at the `LogisticRegression` so we can inspect the most important coefficients. 
- It bears reiterating that not only did GradientBoostClassifier score slightly higher than LogisticRegression, it did so with Polynomial Degree = 1 which equates to a simpler model

#### Coefficients
- We showed the coefficients of our best Logistic Regression Model even though we had a better GradientBoostClassifier Model.  This was done to show the relative importance of the features. The features were listed from most important to least important.  
- Ability to interpret the features themselves and their effect on the target is limited due to the fact that numerical values underwent a yeo-johnson transformation and also because we have a binary classification problem which means we cannot speak in terms of an increase or decrease in the target like we can with a Regression problem.

#### Precision-Recall Tradeoff 
- We also showed the confusion matrix and Precision-Curves for our best Logistic Regression Models and our best overall (GBC) model.  
- For both models, we invited the reader to set the threshold to various values in order to get a sense of the tradeoff by seeing the effect on the FPs and FNs in the confusion matrix as well as where on the PRC curve the selected threshold lies.  We did this because the Billing Department has a slight leaning towards Precision over Recall.  This allows the team to visualize what the tradeoff looks like at varying thresholds

#### Lift and Cumulative Gains Curve for best Model
- The Lift Curve for our GBC model shows that for the first 20% of our 'best' claims, we will have ~5x more Accepted negotiations than if we selected random claims. 
- However, we do not know what the characteristics of those claims we are calling 'best'are.  In order to explain this, we would need to use clustering methods such as k-means. 
- In the Cumulative Gains Curve, we see we can obtain almost ~99% of the Accepted Negotiations with just 20% of our claims! 

#### Counter Factuals
- Two Counter Factuals for the first sample were explored: 
1. All features being the same, but increasing the offer_to_neg ratio from 0.0546 to 0.634 and increasing the offer_to_counter_offer ratio from 0.0728 to 0.8662 (likely to occur with a change in TPA_rep) would have resulted in changing the outcome from Rejected to Accepted.  
2. All features being the same, but increasing the offer_to_neg ratio from 0.0546 to 0.9456 and increasing the offer_to_counter_offer ratio from 0.0728 to 0.625 (likely to occur with a change in group_number) would have resulted in changing the outcome from Rejected to Accepted.  

#### Summary of Non-Technical Findings (Findings Only)
We have built a Machine Learning model to predict the outcome of negotations. We share our main findings below: 
- Out of 2133 claims that we tested our best model, the model predicted:
    - 2029 negotiation decisions correctly 
    - 55 negotiations as Accepted that were actually Rejected
    - 49 negotiations as Rejected that were actually Accepted
- The most important aspects of the claim that affect predictability (in descending order) are: Offer to Negotgiation Amount ratio, Offer to Counter offer Ratio, Negotiation Amount to Billed Amount, Was the claim initiated in response to a Reconsideration Letter or some other action taken by us?, Was it a Split Claim? Was it a UB claim?  How many days before the deadline did we receive the offer?  There are of course other factors but these are the most important ones. Unfortunately with the exception of who initiated the claim, these factors are not something we control.  
- If we were able to group our claims by Acceptance Rate, we would find that the first 20% of our 'best' claims have ~5x more Accepted negotiations than if we select a claim at random. Unfortunately, we do not yet know what  the characteristics of those claims we are calling 'best'claims.  In order to explain this, we would need to conduct additional research. If we are successful with this research, we can potentially obtain almost ~99% of the Accepted Negotiations with just 20% of our claims! 
    

#### Actionable Items, Recommendations and Next Steps (Non-Technical)
- If/whenever possible, we should initiate claims ourselves rather than let the Insurance companies initiate them because this increases the Acceptance Rate. 
- Perform a study to identify profiles of claims that would result in higher than average acceptance rates
- Build data validations to improve the quality of the data. For example, check if the offer amount is greater than the negotiation amount or billed amount before allowing a record to be saved.
- As an organization, we need to make a business decision on whether we wish to tune our model to increase the confidence level (e.g. 80% confidence) even though this would result in more claims predicted as Rejections that are actually Acceptances
- In order for this project to be useful, we would recommend building an application that would take all the fields in a claim including the offer amount and predict the negotiation decision. 
- Update our data with new claim samples 2 to 3 times a year and retrain and tune the model 
- Based on our work, we can take an already rejected claim and show by how much the ratio fields or number of days features would have needed to change to bring about an Acceptance. This counter-factual information may be useful for us to study going forward, in order to develop a guidance chart for the team