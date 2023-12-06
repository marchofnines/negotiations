"""
Author: Basil Haddad
Date: 11.01.2023

Description:
    Contains Helper functions for Feature Engineering and Cross-Validation
"""

from importlib import reload
from helpers.my_imports import * 
import helpers.preprocessing as pp
from IPython.core.display import HTML
from pandas.io.formats import style


def vif(data, impute_strategy='most_frequent', dropna=False):
    """
        Calculate the Variance Inflation Factor (VIF) for each numeric feature in a DataFrame. As an extra
        feature of this function, if the data contains nulls, nulls can be:
        - imputed using SimpleImputer
        - dropped 
        
        Note that these changes do not affect the original data, rather a deep copy of the data

        Parameters:
        -----------
        data : pandas.DataFrame
            The DataFrame containing the features for which VIFs are to be calculated.

        impute_strategy : str, optional (default='most_frequent')
            The strategy for imputing missing values. Options include 'mean', 'median', 
            'most_frequent', or any other strategy accepted by SimpleImputer. 
            If None, no imputation is performed.

        dropna : bool, optional (default=False)
            If True, rows with any missing values are dropped. If False, imputation is 
            performed according to `impute_strategy`.

        Returns:
        --------
        vif_df : pandas.DataFrame
            A DataFrame with feature names and their corresponding VIF values, sorted in 
            descending order of VIF values.

        Notes:
        ------
        - Only numeric features are considered for VIF calculation.
        - VIF values above 5 or 10 indicate high multicollinearity, depending on the 
        threshold used in the specific domain of study.
        """

    # Initialize dictionary to store VIF values
    vif_dict = {}
    
    #Create copy of data and set exogs to contain only numeric columns
    df = data.copy(deep=True).select_dtypes(include=['number'])
    exogs=df.columns
    
    #Optionally drop nulls or impute them 
    if dropna: 
        df=df.dropna()
    else:
        #Define imputer pipe
        if impute_strategy is not None:
            imputer_pipe = Pipeline([
                ('imputer', SimpleImputer(strategy=impute_strategy)),
                ])
            # Apply the imputation
            df = pd.DataFrame(imputer_pipe.fit_transform(df), columns=exogs)
    
    for exog in exogs:
        not_exog = [i for i in exogs if i !=exog]
        # split the dataset, one independent variable against all others
        X, y = df[not_exog], df[exog]

        # fit the model and obtain R^2
        r_squared = LinearRegression().fit(X,y).score(X,y)

        # compute the VIF
        vif = 1/(1-r_squared)
        vif_dict[exog] = vif

    return pd.DataFrame({"VIF": vif_dict}).sort_values(by='VIF', ascending = False)


def my_permutation_importance(fit_models, X_test, y_test, scoring, n_repeats=10, rs=42): 
    """
    Calculate and display permutation importance for a list of fitted models.

    This function iterates over a list of fitted machine learning models and calculates
    the permutation importance for each model on a given test dataset. The permutation
    importance is calculated using a specified scoring metric and is presented in a sorted
    DataFrame format for each model.

    Parameters:
    fit_models (list): A list of fitted model objects. These can be individual models or
                       pipeline objects from scikit-learn.
    X_test (DataFrame): Test features used to evaluate the model.
    y_test (Series or array): True labels for the test set.
    scoring (str): Scoring metric to evaluate the models (e.g., 'accuracy', 'roc_auc').
    n_repeats (int, optional): Number of times to permute a feature. Defaults to 10.
    rs (int, optional): Random state for reproducibility of the permutation. Defaults to 42.

    Returns:
    None: The function does not return any value. It prints the model name and displays the
          permutation importance in a DataFrame format for each model.
    """
    
    for fit_model in fit_models:
        # Calculate permutation importance for each model
        r = permutation_importance(fit_model, X_test, y_test, n_repeats=10, random_state=42, 
                               scoring = scoring,
                               )
        pi_result = pd.DataFrame({"Variables":X_test.columns,"Score":r.importances_mean}).sort_values(by="Score",ascending = False)
        
        # Sort by score and then transpose the DataFrame
        pi_result = pi_result.set_index("Variables").T

        # Extract the name of the current model for display purposes
        model_name = fit_model.__class__.__name__ if not hasattr(fit_model, 'named_steps') else next(reversed(fit_model.named_steps.items()))[0]
        
        # Display the model name and its permutation importance
        print(f"{model_name}:")
        
        # Display the model name and its permutation importance
        display(pi_result)


def build_pipe_evaluate_bin_clf(models, X, y, test_size=0.25, stratify=True, rs=42, drop_cols=[], transformer=None, scaler=None, selector=None, summary=False):

    #Define dict of arrays to store results
    results = {
    'Model': [],
    'Train Time': [],
    'Inference Time': [],
    'Train F1': [],
    'Test F1': [],
    'Train Precision': [],
    'Test Precision': [],
    'Train Recall': [],
    'Test Recall': [],
    'Train Accuracy': [],
    'Test Accuracy': [],
    }
 
    #If not a dict, make it a dict for uniform processing
    if not isinstance(models, dict):
        models = {
            models.__class__.__name__ : models
        }
           
    if drop_cols !=[]:
        X=X.copy(deep=True).drop(columns=drop_cols)
 
    # Split Data into Train and Test Sets
    if stratify:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=rs)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=None, random_state=rs)
               
    import time
    fit_models=[]
    for model_name, model in models.items():      
        #build steps for pipeline
        steps = []
        if transformer is not None:
            steps.append(('transformer', transformer))
        if scaler is not None:
            steps.append(('scaler', scaler))
        if selector is not None:
            steps.append(('selector', selector))
        
        #Take start timestamp for training
        start_time = time.time()

        #if it's a pipeline, or if it's a standalone model, just fit it
        if isinstance(model, Pipeline) or (transformer is None and scaler is None and selector is None):
            fit_model = model.fit(X_train, y_train)
        #else add the model to the steps and then fit it 
        else:
            steps.append((model_name, model))
            fit_model = Pipeline(steps).fit(X_train, y_train)
            fit_models.append(fit_model)

        #Take end timestamp for training
        train_time = time.time() - start_time

        #Take start timestamp for test inference
        start_time = time.time()
        y_test_pred = fit_model.predict(X_test)
        #Take end timestamp for test inference 
        inference_time = time.time() - start_time

        #Compute various score metrics
        train_accuracy = fit_model.score(X_train, y_train)
        test_accuracy = fit_model.score(X_test, y_test)

        y_train_pred = fit_model.predict(X_train)
        #y_test_preds = model.predict(X_test) (already computed)

        train_precision = precision_score(y_train, y_train_pred, average='weighted')
        train_recall = recall_score(y_train, y_train_pred, average='weighted')
        train_f1 = f1_score(y_train, y_train_pred, average='weighted')    

        test_precision = precision_score(y_test, y_test_pred, average='weighted')
        test_recall = recall_score(y_test, y_test_pred, average='weighted')
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')

        #Append results to arrays 
        results['Model'].append(model_name)
        results['Train Time'].append(train_time)
        results['Inference Time'].append(inference_time)
        results['Train F1'].append(train_f1)
        results['Test F1'].append(test_f1)
        results['Train Precision'].append(train_precision)
        results['Test Precision'].append(test_precision)
        results['Train Recall'].append(train_recall)
        results['Test Recall'].append(test_recall)
        results['Train Accuracy'].append(train_accuracy)
        results['Test Accuracy'].append(test_accuracy)

    #Create results dataframe using the arrays of metrics
    results_df = pd.DataFrame(results)
    if summary:
        display(results_df)
    return results_df, fit_models


def custom_get_feature_names(fit_model):
    """
    Extract feature names after they have been through the transformer and selector.
    See Notes for limitations.
    
    Parameters:
    -----------
    fit_model : scikit-learn Pipeline object
        The fitted pipeline model from which to extract feature names.

    Returns:
    --------
    selected_features : numpy.ndarray
        Array of selected feature names after applying feature selection.

    discarded_features : numpy.ndarray
        Array of feature names that were discarded during feature selection.

    Notes:
    ------
    This function only works for transformers with the attributes and steps used in 
    the Hyperparameter Tuning section of this project.  Namely: 
    - The transformer is expected to have PolynomialFeatures, OneHotEncoder, OrdinalEncoder,
    one other encoder and a selector
    - The pipeline is expected to have named steps 'transformer' and 'selector'.
    - The 'transformer' is expected to have named transformers 'num', 'cat', 'ohe', 'ord'
    """

    transf = fit_model.named_steps['transformer']
    num_transformer = transf.named_transformers_['num']
    orig_num_features = num_transformer.feature_names_in_
    #Obtain feature names coming out of PolynomialFeatures
    transf_num_feature_names = num_transformer[2].get_feature_names_out(orig_num_features)
    #Obtain feature names coming out of Categorical Encoder (Binary/Target, etc)
    binary_feature_names = transf.named_transformers_['cat'][1].get_feature_names_out()
    #Obtain feature names coming out of OneHotEncoder
    ohe_feature_names = transf.named_transformers_['ohe'].get_feature_names_out()
    #Obtain feature names coming out of OrdinalEncoder
    ord_feature_names = transf.named_transformers_['ord'].get_feature_names_out()
    #Define all features coming out of transformer by combining the features from various parts of the transformer
    all_feature_names= list(transf_num_feature_names)+list(binary_feature_names)+list(ohe_feature_names)+list(ord_feature_names)
    #Define selector mask
    selector_mask = transf = fit_model.named_steps['selector'].get_support()
    #Define features that were selected by selector
    selected_features = np.array(all_feature_names)[selector_mask]
    #Define features discarded by selector
    discarded_features = list(set(all_feature_names) - set(selected_features))
    return selected_features, discarded_features

       

def get_lgr_pipe_coefs(pipe, transf_name='transformer', scaler_name='scaler', selector_name=None):
    ####PROB NEED TO REDO
    """
    Extract logistic regression coefficients from a pipeline for feature interpretation.

    This function retrieves the coefficients and intercept from a logistic regression model within
    a pipeline. It also obtains feature names, means, and standard deviations to assist in 
    interpreting the model. It returns a DataFrame sorted by the exponential of the unscaled coefficients.

    Parameters:
    pipe (Pipeline or ImbPipeline): Fitted scikit-learn pipeline containing a logistic regression model.
    transf_name (str, optional): Name of the transformer step in the pipeline. Defaults to 'transformer'.
    scaler_name (str, optional): Name of the scaler step in the pipeline. Defaults to 'scaler'.
    selector_name (str, optional): Name of the feature selector step in the pipeline. Defaults to None.

    Returns:
    DataFrame: A DataFrame containing coefficients, means, standard deviations, and exponential unscaled coefficients.
    """
    
    #Define coefficients and intercept
    steps_list = list(pipe.named_steps.items())
    model_name, _ = steps_list[-1]
    my_coefs = pipe.named_steps[model_name].coef_[0]
    intercept = pipe.named_steps[model_name].intercept_
    #Features from Transformer
    remaining_features, _ = custom_get_feature_names(pipe)   #pipe.named_steps[transf_name].get_feature_names_out()
    if selector_name:
        remaining_feature_mask = pipe.named_steps['selector'].get_support()
        scaler = pipe.named_steps[scaler_name]
        #median = scaler.center_[remaining_feature_mask]
        std_dev = scaler.scale_[remaining_feature_mask] 
      
    else: 
        # Get the means and standard deviations for all features
        scaler = pipe.named_steps[scaler_name]
        #median = scaler.center_
        std_dev = scaler.scale_
        
    # Create the dataframe with coefficients, means, and std_devs
    interpretation_df = pd.DataFrame({
    'coef': my_coefs,
    #'median': median,
    'std_dev': std_dev, 
    'exp_unscaled_coefs': np.exp(my_coefs/std_dev),
    }, index=remaining_features)
    
    #Sort the dataframe
    return interpretation_df.sort_values(by='exp_unscaled_coefs', ascending=False)


def cv_and_holdout(estimator,X, y, test_size=0.25, stratify=None, random_state=42, search_type='halving_random', param_dict=None,
                  scoring=None, refit=None, refit_scorer=None, holdout_tolerance=0, verbose=0, cv=5, n_iter=10, factor=3, summary=True):
    """
    Conducts cross-validation and holdout validation for a given estimator using specified parameters.

    This function splits the dataset into training and testing sets,  dynamically builds transformer and pipeline to include 
    imputers, polynomial features, numerical transformers, rare category combiners, scalers, encoders, selectors, etc, performs 
    hyperparameter tuning using specified search methods, and evaluates model performance using both cross-validation and holdout 
    validation. It supports GridSearchCV, RandomizedSearchCV,and HalvingRandomSearchCV for hyperparameter tuning. Results are 
    presented in 3 DataFrames as follows: 
    - cvresults_ enhanced with holdout results and sorted by descending order of Cross-Validation rank
    - models sorted from best to worse by descending order of holdout validation rank. It also prioritizes models that are not overfit 
    even if those scores are lower
    - If multiple estimators are evaluated, it creates a third dataframe displaying the best holdout results from each estimator

    Function optionally plots model performance.

    Parameters:
    estimator (estimator object): An estimator object implementing 'fit'.
    X (array-like): Feature dataset.
    y (array-like): Target dataset.
    test_size (float, optional): Proportion of the dataset to include in the test split. Default is 0.25.
    stratify (array-like, optional): Data is split in a stratified fashion using this as the class labels. Default is None.
    random_state (int, optional): Controls the shuffling applied to the data before applying the split. Default is 42.
    search_type (str, optional): Type of search to perform ('grid', 'random', 'halving_random'). Default is 'halving_random'.
    param_dict (dict, optional): Dictionary with parameters names as keys and lists of parameter settings to try as values.
    scoring (str or callable, optional): A str or a scorer callable object/function with signature scorer(estimator, X, y).
    refit (bool or str, optional): Refit an estimator using the best found parameters on the whole dataset. Default is None.
    refit_scorer (callable, optional): A scorer callable object/function for the refit step. Default is None.
    holdout_tolerance (float, optional): Tolerance for overfitting in holdout validation. Default is 0.
    verbose (int, optional): Controls the verbosity: the higher, the more messages. Default is 0.
    cv (int, optional): Specify the number of folds in a (Stratified)KFold. Default is 5.
    n_iter (int, optional): Number of parameter settings that are sampled in RandomizedSearchCV. Default is 10.
    factor (int, optional): Halving factor for n_candidates in HalvingRandomSearchCV. Default is 3.
    summary (bool, optional): Whether to display summary plots and tables. Default is True.

    Returns:
    tuple: A tuple containing the results DataFrame and the best holdout estimator.
    """
    pd.set_option('display.max_columns', None)
   
    #Validate refit metric was entered

    if isinstance(scoring, dict): #assume refit scorer is one of the item in the dict
        if refit in scoring:  
            refit_scorer = scoring[refit]
        else: 
            raise ValueError(f"The refit metric {refit} was not found in the scoring_metrics dictionary.")
    else: 
        refit = scoring
        refit_scorer = refit_scorer
            
    # Step 1: Split Data into Train and Test Sets and Run GridSearchCV or RandomizedSearchCV
    if stratify is None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=None, random_state=random_state)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

    if 'grid' in search_type.lower():
        search = GridSearchCV(estimator, param_dict, scoring=scoring, refit=refit, cv=cv, n_jobs=-1)
    elif 'halving' in search_type.lower():
        search = HalvingRandomSearchCV(estimator,param_dict,factor=factor,scoring=scoring,refit=True,random_state=random_state,
                                       verbose=verbose, cv=cv, n_jobs=-1)
    elif 'random' in search_type.lower():
        search = RandomizedSearchCV(estimator, param_dict, n_iter=n_iter, scoring=scoring, refit=refit, random_state=random_state, verbose=verbose, cv=cv, n_jobs=-1)
    search.fit(X_train, y_train)
    
    # Step 2: Build Custom Results DataFrame based on cv_results_
    cv_results = pd.DataFrame(search.cv_results_)
    #Unclutter results
    cv_results = cv_results.drop(columns=cv_results.columns[cv_results.columns.str.startswith('split')])
    cv_results = cv_results.drop([col for col in cv_results.columns if 'std_' in col], axis=1)
    cv_results = cv_results.drop(columns='mean_score_time')
    cv_results = cv_results.drop(columns='params')
    
    # Step 3: Holdout Validation using refit score metric
    holdout_train_scores = []
    holdout_test_scores = []
    overfit_flags = []
    best_holdout_estimator = None
    best_holdout_score = 0
    for candidate_params in search.cv_results_['params']:
        current_estimator = clone(estimator)
        current_estimator.set_params(**candidate_params)
        current_estimator.fit(X_train, y_train)
        
        holdout_train_score = refit_scorer(current_estimator, X_train, y_train)
        holdout_test_score = refit_scorer(current_estimator, X_test, y_test)
        
        if holdout_train_score < (1+holdout_tolerance)*holdout_test_score: 
            if holdout_test_score > best_holdout_score:
                best_holdout_score = holdout_test_score
                best_holdout_estimator = copy.deepcopy(current_estimator)
        
        holdout_train_scores.append(holdout_train_score)
        holdout_test_scores.append(holdout_test_score)
        
        overfit_flags.append(1 if holdout_train_score > (1+holdout_tolerance)*holdout_test_score else 0)
 
    # Step 4: Augment cv_results with holdout testing results 
    #Define column names
    cv_rank_refit_col=f"cv_rank_{refit}"
    cv_test_score_refit_col=f"cv_{refit}" #mean_test!
    ho_rank_refit_col=f"holdout_rank_{refit}"
    ho_train_score_refit_col=f"train_{refit}"
    ho_test_score_refit_col=f"test_{refit}"
    #rename cv columns so we can distinguish these from the holdout columns
    cv_results.rename(columns={
        f'rank_test_{refit}': cv_rank_refit_col,
        f'mean_test_{refit}': cv_test_score_refit_col
    }, inplace=True)
    #Update dataframe with new columns
    cv_results[ho_train_score_refit_col] = holdout_train_scores
    cv_results[ho_test_score_refit_col] = holdout_test_scores
    cv_results['is_overfit'] = overfit_flags
    cv_results['is_overfit']= cv_results['is_overfit'].map({0: 'No', 1: 'Yes'}) 
    #Take snapshot to be used in step 5
    ho_results = cv_results.copy()
    #Sort results by descending CV Rank and reorder columns for visibility
    cv_results = cv_results.sort_values(by=[cv_rank_refit_col, ho_test_score_refit_col] , ascending=[True, False])  
    #Show most important columns first
    cv_results= pp.reorder_cols_in(cv_results, [cv_rank_refit_col, cv_test_score_refit_col, 'is_overfit', ho_train_score_refit_col, ho_test_score_refit_col, 
                                               'mean', 'param'] , after=False) 
    
    
    #Step 5: Define dataframe to display holdout test results ordered by overfit status and descending holdout test scores
    ho_results.sort_values(by=['is_overfit', ho_test_score_refit_col, cv_rank_refit_col], ascending=[True, False, True], inplace=True)
    ho_results.reset_index(drop=True, inplace=True)
    # Create the new holdout rank column based on the new index
    ho_results[ho_rank_refit_col] = ho_results.index + 1  
      
    #Show most important columns first
    ho_results= pp.reorder_cols_in(ho_results, [cv_rank_refit_col, cv_test_score_refit_col, 'is_overfit', ho_train_score_refit_col, ho_test_score_refit_col,  
                                              'mean', 'param'] , after=False) #'mean_fit_time', 'mean_score_time'

    #Step 6: Display Best CV Model Details and Best Holdout Model Details
    if summary:
        if hasattr(estimator, 'steps'):
            display(HTML(f'<h3>Results for {estimator.steps[-1][0]}: </h3>'))
        display(HTML(f'<h5>Models ranked by descending {cv_rank_refit_col}</h5>'))
        display(cv_results.iloc[:4,:].style.hide(axis='index'))
        display(HTML(f'<h5>Models ranked by overfit status and descending holdout {ho_test_score_refit_col}</h5>'))
        display(HTML(ho_results.iloc[:4,:].to_html(index=False)))

    # Step 7: Plot Holdout Validation Model Scores and show best non-overfit (or least overfit if threshold > 0) if available
    if summary:
        sns.set_style('darkgrid')
        common_fontsize=23
        linewidth=1.8
        markers='o',
        s=80
        plt.clf()
        plt.figure(figsize=(23 , 6))
        
        sns.scatterplot(x=cv_rank_refit_col, y=ho_train_score_refit_col, label=f'Holdout Train {refit} Score', markers=markers,  s=s, data=ho_results)
        sns.scatterplot(x=cv_rank_refit_col, y=ho_test_score_refit_col, label=f'Holdout Test {refit} Score', markers=markers,  s=s, data=ho_results)
        
        #Show best non-overfit (or least overfit if threshold > 0) if available
        best_model_rank_score_list = None
        filtered_ho_results = ho_results.query(f"{ho_rank_refit_col}==1 and is_overfit=='No'")
        if not filtered_ho_results.empty:
            best_model_rank_score_list = filtered_ho_results[[cv_rank_refit_col, ho_test_score_refit_col]].iloc[0].to_list()
            plt.axvline(x=best_model_rank_score_list[0], color='r', linestyle='--', label=f"Best Non-Overfit Model {refit} Test Score: {best_model_rank_score_list[1]:.3f}")
        else:
            print("No non-overfit models were found. Consider re-running the function with a houldout_threshold > 0")
        
        plt.xticks(fontsize=common_fontsize)
        plt.yticks(fontsize=common_fontsize)
        plt.title(f"{refit} Holdout Train and Test Scores", weight='bold', fontsize=common_fontsize+2)
        plt.xlabel(cv_rank_refit_col, fontsize=common_fontsize) 
        plt.ylabel("Score", fontsize=common_fontsize)
        plt.grid(True, which='both', linestyle='--', linewidth=0.6)
        plt.legend(fontsize=common_fontsize-3)
        
        plt.tight_layout()
        plt.show()
    return ho_results, best_holdout_estimator


def run_pipelines(pipe_param_pairs, X, y, test_size = 0.25, stratify=None, random_state=42, search_type='random', 
                   scoring=None, refit=None, holdout_tolerance=0, verbose=0, cv=5, n_iter=10, factor=3, summary=True):
    """
    Run multiple pipelines with different hyperparameter settings into cv_and_holdout function and collect the results.
    
    Parameters:
        pipe_param_pairs: Pairs of pipelines and parameter dics
        X: Feature matrix
        y: Target vector
        test_size: test_size for train_test_split
        stratify: stratify for train_test_split
        random_state: random_state for train_test_split
        search_type: use grid for GridSearchCV and random for RandomizedSearchCV
        param_dict: Dict of parameters to be used in grid/randomized search
        scoring: Dict of Scoring Metrics to be used in grid/randomized search
        refit: refit scoring metric (required) for grid/randomized search as well as holdout validation
        holdout_tolerance: is overfitting within given tolerance?
        verbose: verbose parameter for grid/randomized search
        cv: cv parameter for grid/randomized search
        n_iter: number of iterations for grid/randomized search
        summary: Show summary of results which include: 
                 - Models ranked by descending CV rank
                 - Models ranked by overfit status and descending holdout test scores
                 - Plot of all the models + vertical line at best non-overfit model
                 
        Returns:
        - Dataframe containing the performance metrics for the best model from each pipe/param pair
        - An array of the best models 
    """
    if search_type=='halving_random' and isinstance(scoring,str):
        refit=scoring
    output, models= [], []
    for pipe, params in pipe_param_pairs:
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=None, random_state=random_state)
        result, model =cv_and_holdout(estimator=pipe,
                                X=X,
                                y=y,
                                stratify=stratify,
                                param_dict=params,
                                scoring=scoring,
                                refit= refit,  
                                random_state=random_state,
                                search_type= search_type,  
                                n_iter=n_iter,
                                holdout_tolerance=holdout_tolerance,
                                cv=cv,
                                verbose=verbose, 
                                summary=summary,
                                factor=factor)
        
       #Create summary dict of best model from each pipe/param pair
        output_row = {'model': pipe.steps[-1][0], 
                      f'train {refit} score': result.loc[0, f'train_{refit}'],
                      f'test {refit} score': result.loc[0, f'test_{refit}'], 
                      'mean fit time': result.loc[0, 'mean_fit_time']}
        # Append columns that start with 'mean_test' to output_row
        mean_test_cols = [col for col in result.columns if col.startswith('mean_test')]
        for col in mean_test_cols:
            output_row[col] = result.loc[0, col]
        # Append to output and models arrays 
        output.append(output_row)
        models.append(model)
    #Display summary DataFrame containing best models from each pipe/param pair    
    output_df = pd.DataFrame(output)    
    display(HTML(f'<h3>Best Models From Each Grid/Random Search: </h3>'))
    display(output_df.style.hide(axis='index'))

    return output_df, models

def build_and_run_pipes (df,target,scoring_metrics, refit, search_type, estimator_dicts,
                   num_imputer=None, 
                   num_imputer_params=None, 
                   num_transformer=None, 
                   num_transformer_params=None,
                   poly=None, 
                   poly_params=None, 
                   num_cols=None,
                   
                   cat_imputer=None, 
                   cat_imputer_params=None, 
                   cat_combiner=None,
                   cat_combiner_params=None,
                   cat_encoder=None, 
                   cat_encoder_params=None,
                   cat_cols=None,  
                   onehotencoder=None,
                   ohe_params=None,
                   ohe_cols=None, 
                   ord_cols=None, 
                   
                   oversampler=None, over_params=None,
                   undersampler=None, under_params=None, 
                   
                   scaler=None, scaler_params=None,
                   selector=None, selector_params=None,
                 
                   set_name=None,
                   cv=5, n_iter=10, summary=True, verbose=0,
                   test_size=0.25, stratify=None,rs=42,factor=3):
    """
    Constructs, pipelines that include preprocessing, scaling, rare category compbiners, oversampling/undersampling, 
    feature selection.  Pipeline can be rebuilt for various estimators for cross-validation with hyperparameter tuning 
    as well as holdout evaluation. 

    Parameters:
    df (DataFrame): The input data frame containing features and the target variable.
    target (str): The name of the target variable column.
    scoring_metrics (str or list): Scoring metric(s) for model evaluation.
    refit (str or bool): Metric to refit models during hyperparameter tuning.
    search_type (str): Type of hyperparameter search ('GridSearchCV', 'RandomizedSearchCV', etc.).
    estimator_dicts (list of dicts): List containing estimator tuples and corresponding parameter grids.
    [Additional parameters for preprocessing, such as imputers, transformers, scalers, encoders, samplers, etc.]
    set_name (str, optional): Name for saving the models and results.
    cv (int): Number of cross-validation folds.
    n_iter (int): Number of iterations for randomized search.
    summary (bool): Whether to display a summary of results.
    verbose (int): Level of verbosity.
    test_size (float): Proportion of the dataset to include in the test split.
    stratify (str or None): Column to use for stratifying the split.
    rs (int): Random state for reproducibility.
    factor (int): Factor used in Halving search methods.

    Returns:
    tuple: A tuple containing a DataFrame summarizing the results and a list of the best fitted models. The DataFrame includes performance metrics and parameters of evaluated models, while the list contains models fitted with the best-found parameters.
    """
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    #Copy dataframe 
    df_copy = df.copy(deep=True)
    
    #For Catboosting, we convert numbers to string and encode target to numerical
    if isinstance(cat_encoder[1], CatBoostEncoder):
        df_copy[df_copy.select_dtypes(exclude=['number']).columns] = df_copy.select_dtypes(exclude=['number']).astype('str')
        y = df_copy[target].map({'Accepted':1, 'Rejected':0})
    else:
        y = df_copy[target]

    #X will be based on columns passed in.  This allows us to test with some features missing without 
    #actually modifying original dataset
    X = df_copy[ohe_cols+ord_cols+cat_cols+num_cols]

    """ 
    General design of transformern shown below 
    transformer=ColumnTransformer(transformers=[
        ('num', Pipeline(steps=num_pp_steps), num_cols) if num_pp_steps is not None else None,
        ('cat', Pipeline(steps=cat_pp_steps), cat_impute_cols) if cat_pp_steps is not None else None,
        ('ohe', OneHotEncoder(), ohe_cols),
        ('ord', OrdinalEncoder(), ord_cols),
        ('bin', BinaryEncoder(), bin_cols)
    ])"""
    #Build numerical preprocessor to include imputer, numerical transformer and polynomial features
    num_pp_steps=[]
    if num_imputer is not None or num_transformer is not None or poly is not None:
        if num_imputer is not None:
            num_pp_steps.append(num_imputer)
        if num_transformer is not None: 
            num_pp_steps.append(num_transformer)
        if poly is not None: 
            num_pp_steps.append(poly)
    
    #Build categorical preprocessor to include imputer, rare category combiner and encoder
    cat_pp_steps=[]
    if cat_imputer is not None: 
        cat_pp_steps.append(cat_imputer)
    if cat_combiner is not None:
        cat_pp_steps.append(cat_combiner)
    if cat_encoder is not None:
        cat_pp_steps.append(cat_encoder)
    
    #Build transformer steps to include numerical and categorical preprocessors as well as onehot and ordinal encoders
    transformer_list=[]
    if num_pp_steps != []: 
        transformer_list.append( ('num', Pipeline(steps=num_pp_steps), num_cols) )
    if cat_pp_steps !=[]:
        transformer_list.append( ('cat', Pipeline(steps=cat_pp_steps), cat_cols) )
   
    if onehotencoder is not None: 
        transformer_list.append((onehotencoder[0], onehotencoder[1], ohe_cols)) 
        
    if ord_cols is not []: 
        transformer_list.append(('ord', OrdinalEncoder(), ord_cols))
    #Create ColumnTransformer using previously created steps 
    transformer=None
    if transformer_list!=[]:
        transformer=ColumnTransformer(transformers=transformer_list)
    
    #Build pipeline steps to include transformer, oversampler, undersampler, scaler and selector
    #The fixed_pipe_steps do not include an estimator.  This is because we want to be able to pass in 
    #multiple estimators
    fixed_pipe_steps=[]
    if transformer is not None:
        fixed_pipe_steps.append(('transformer', transformer))

    if oversampler is not None:
        fixed_pipe_steps.append(oversampler)
        
    if undersampler is not None:
        fixed_pipe_steps.append(undersampler)
    
    if scaler is not None: 
        fixed_pipe_steps.append(scaler)    
    
    if selector is not None: 
        fixed_pipe_steps.append(selector)

    #Create prefix based on estimator name to be added to list of parameters (e.g. lgr__C)
    pipe_param_pairs=[]
    for est_dict in estimator_dicts: 
        est_tuple, est_params=est_dict['est_tuple'], est_dict['est_params']    
        prefix = est_tuple[0] + '__'

        # Update parameter names with prefix
        est_params = {prefix + key: value for key, value in est_params.items()}

        #add estimator to pipeline steps 
        pipe_steps=list(fixed_pipe_steps)    
        pipe_steps.append( (est_tuple) )
        #create pipeline 
        pipe = ImbPipeline(steps=pipe_steps) if oversampler is not None or undersampler is not None else Pipeline(steps=pipe_steps)
        #generate list of parameters 
        pp_params={}
        if num_imputer_params is not None: 
            pp_params=pp_params | num_imputer_params
        if num_transformer_params is not None: 
            pp_params=pp_params | num_transformer_params
        if poly_params is not None:
            pp_params=pp_params | poly_params
        if ohe_params is not None: 
            pp_params=pp_params | ohe_params
        if cat_imputer_params is not None: 
            pp_params=pp_params | cat_imputer_params
        if cat_combiner_params is not None: 
            pp_params=pp_params | cat_combiner_params
        if cat_encoder_params is not None: 
            pp_params=pp_params | cat_encoder_params
        if over_params is not None:
            pp_params=pp_params | over_params
        if under_params is not None:
            pp_params=pp_params | under_params
        if scaler_params is not None: 
            pp_params=pp_params | scaler_params 
        if selector_params is not None:
            pp_params=pp_params | selector_params
        
        pipe_param_pairs.append( (pipe, pp_params | est_params) )
    #if multiple estimators are passed, this function will call grid/random search for each
    #and provide a summary dataframe at the end with the best models from each estimator
    if len(pipe_param_pairs) > 1: 
        results_df, models = run_pipelines(pipe_param_pairs = pipe_param_pairs,
                X=X,
                y=y,
                test_size=test_size,
                stratify=stratify,
                random_state=rs,
                search_type=search_type,
                scoring=scoring_metrics,
                refit=refit,
                holdout_tolerance=0,
                verbose=verbose,
                cv=cv,
                n_iter=n_iter, 
                summary=summary, 
                factor=factor)  
    else: 
        #perfrom grid/random search cross-validation as well as holdout validation
        #list best models by descending order holdout rank and plot scores 
        results_df, models =cv_and_holdout(estimator=pipe,
                                X=X,
                                y=y,
                                stratify=stratify,
                                param_dict= pipe_param_pairs[0][1],
                                scoring=scoring_metrics,
                                refit= refit,  
                                random_state=rs,
                                search_type= search_type,  
                                n_iter=n_iter,
                                holdout_tolerance=0,
                                cv=cv,
                                verbose=verbose, 
                                summary=summary,
                                factor=factor)
        
    path = 'models/hyperparam_tuning'
    if set_name:
        #Save Models
        dump(models, f'{path}/{set_name}.joblib') 

        #save preprocessed dataframe
        results_df.to_csv(f'{path}/{set_name}_results.csv', index=False)


def my_cross_val(df,target,scoring, 
                   models,
                   num_imputer=None, 
                   num_transformer=None, 
                   poly=None, 
                   num_cols=None,
                   
                   cat_imputer=None, 
                   cat_combiner=None,
                   cat_encoder=None, 
                   cat_cols=None,  
                   
                   onehotencoder=None, 
                   ohe_cols=None, 
                   ord_cols=None, 
                   
                   oversampler=None, 
                   undersampler=None, 
  
                   scaler=None, 
                   selector=None, 
                   
                   set_name=None,
                   sort_metric=None, cv=5, verbose=0,
                   test_size=0.25, stratify=False,rs=42):
    
    """
    Performs cross-validation on a set of models with specified preprocessing steps and hyperparameters.

    Constructs, pipelines that include preprocessing, scaling, rare category compbiners, oversampling/undersampling, 
    feature selection.  Pipeline can be rebuilt for various estimators for cross-validation without any hyper parameter tuning.

    Parameters:
    df (DataFrame): The input data frame containing features and the target variable.
    target (str): The name of the target variable column.
    scoring (str or list): Scoring metric(s) for model evaluation.
    models (dict): A dictionary of models to be evaluated.
    [num_imputer, num_transformer, poly, num_cols, etc.]: Optional parameters for numerical preprocessing.
    [cat_imputer, cat_combiner, cat_encoder, cat_cols, etc.]: Optional parameters for categorical preprocessing.
    [onehotencoder, ohe_cols, ord_cols]: Optional parameters for encoding.
    [oversampler, undersampler]: Optional parameters for handling imbalanced data.
    scaler (object, optional): An instance of a scaler.
    selector (object, optional): Feature selection mechanism.
    set_name (str, optional): Name for saving the models and results.
    sort_metric (str, optional): Metric to sort the results by.
    cv (int): Number of cross-validation folds.
    verbose (int): Verbosity level.
    test_size (float): Proportion of the dataset to include in the test split.
    stratify (bool): Whether to stratify the split based on the target variable.
    rs (int): Random state for reproducibility.

    Returns:
    list: A list of fitted models after cross-validation.
    """
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    #Copy dataframe 
    df_copy = df.copy(deep=True)
    #For Catboosting, we convert numbers to string and encode target to numerical
    if isinstance(cat_encoder[1], CatBoostEncoder):
        df_copy[df_copy.select_dtypes(exclude=['number']).columns] = df_copy.select_dtypes(exclude=['number']).astype('str')
        y = df_copy[target].map({'Accepted':1, 'Rejected':0})
    else:
        y = df_copy[target]
    #select subset of features based on passed columns.  This allows us to not to have to constantly modify our input data
    X = df_copy[ohe_cols+ord_cols+cat_cols+num_cols]
    
    # Split Data into Train and Test Sets
    if stratify:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=rs)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=None, random_state=rs)
   
    #Build numerical preprocessor to include imputer, numerical transformer and polynomial features
    num_pp_steps=[]
    if num_imputer is not None or num_transformer is not None or poly is not None:
        if num_imputer is not None:
            num_pp_steps.append(num_imputer)
        if num_transformer is not None: 
            num_pp_steps.append(num_transformer)
        if poly is not None: 
            num_pp_steps.append(poly)
    #Build categorical preprocessor to include imputer, rare category combiner and encoder
    cat_pp_steps=[]
    if cat_imputer is not None: 
        cat_pp_steps.append(cat_imputer)
    if cat_combiner is not None:
        cat_pp_steps.append(cat_combiner)
    if cat_encoder is not None:
        cat_pp_steps.append(cat_encoder)
    #Build transformer steps to include numerical and categorical preprocessors as well as onehot and ordinal encoders
    transformer_list=[]
    if num_pp_steps != []: 
        transformer_list.append( ('num', Pipeline(steps=num_pp_steps), num_cols) )
    if cat_pp_steps !=[]:
        transformer_list.append( ('cat', Pipeline(steps=cat_pp_steps), cat_cols) )
    
    if onehotencoder is not None: 
        transformer_list.append((onehotencoder[0], onehotencoder[1], ohe_cols)) 
        
    if ord_cols is not []: 
        transformer_list.append(('ord', OrdinalEncoder(), ord_cols))
    #Build ColumnTransformer with previously created transformer steps 
    transformer=None
    if transformer_list!=[]:
        transformer=ColumnTransformer(transformers=transformer_list)
    
    #Build pipeline steps to include transformer, oversampler, undersampler, scaler and selector
    #The fixed_pipe_steps do not include an estimator.  This is because we want to be able to pass in 
    #multiple estimators
    fixed_pipe_steps=[]
    if transformer is not None:
        fixed_pipe_steps.append(('transformer', transformer))

    if oversampler is not None:
        fixed_pipe_steps.append(oversampler)
        
    if undersampler is not None:
        fixed_pipe_steps.append(undersampler)
    
    if scaler is not None: 
        fixed_pipe_steps.append(scaler)    
    
    if selector is not None: 
        fixed_pipe_steps.append(selector)
        
    #Completes the pipeline for each estimator passed 
    model_names,fit_models =[],[]
    results_df = pd.DataFrame()
    for model_name, model in models.items():  
        model_names.append(model_name)    
        
        #build steps for pipeline
        pipe_steps = fixed_pipe_steps.copy()
        pipe_steps.append( (model_name, model) )
        pipe = ImbPipeline(steps=pipe_steps) if oversampler is not None or undersampler is not None else Pipeline(steps=pipe_steps)
        #fit pipe
        fit_model = pipe.fit(X_train, y_train)
        fit_models.append(fit_model)
        #Build summary dataframe with results 
        result = pd.DataFrame(pd.DataFrame(cross_validate(fit_model, X_test, y_test, return_train_score=True, scoring=scoring, cv=cv, n_jobs=-1)).mean()).T
        result['model']= model_name
        results_df=pd.concat([results_df, result], axis=0)
    #reorder columns so we can see most important columns first
    results_df = pp.reorder_cols_in(results_df, 'model', after=False)
    #sort results by primary metric
    if sort_metric:
        results_df=results_df.sort_values(by=f'test_{sort_metric}', ascending=False)
    display(results_df.style.hide(axis='index'))
    
    if set_name:
        #Save Models
        dump(fit_models, f'models/cross_val/{set_name}.joblib') 

        #save preprocessed dataframe
        results_df.to_csv(f'models/cross_val/{set_name}_results.csv', index=False)
    return fit_models


