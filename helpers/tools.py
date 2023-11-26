import sys
sys.path.append('/Users/basilhaddad/jupyter/capstone/')
from importlib import reload
from helpers.my_imports import * 
from helpers import preprocessing as pp
from IPython.core.display import HTML

import warnings


def vif(data, impute_strategy='most_frequent', dropna=False):
  vif_dict = {}
  df = data.copy(deep=True).select_dtypes(include=['number'])
  exogs=df.columns
  
  
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
    for fit_model in fit_models:
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


def my_permutation_importance2(fit_models, X_test, y_test, scoring, n_repeats=10, rs=42):
    """
    Calculates and displays the permutation importance of features for a list of fitted models.

    Parameters:
    fit_models (list): A list of fitted machine learning models.
    X_test (DataFrame): The test dataset used for evaluating the models.
    y_test (Series or array-like): The target values for the test dataset.
    scoring (list): A list of scoring metrics to be used for evaluating feature importance.
    rs (int, optional): The random state seed used in permutation importance calculation. Defaults to 42.

    This function calculates the permutation importance of each feature in the test dataset for each 
    model in `fit_models` using the specified `scoring` metrics. It then filters out rows that only contain
    zero means and stds.  The results are displayed for each model with the features sorted by their importance in descending order.
    """
    # Store the feature names from the test dataset
    feature_names = X_test.columns
    
    # Display Heading
    display(HTML(f'<h3>Permuation Results: </h3>'))

    # Iterate over each model in the list of fitted models
    for fit_model in fit_models:
        # Calculate permutation importance for the current model
        # using the provided test dataset and scoring metrics
        r_multi = permutation_importance(fit_model, X_test, y_test, n_repeats=n_repeats, random_state=rs, scoring=scoring)

        # Initialize a dictionary to hold the mean and standard deviation of feature importances
        feature_importance_dict = {}

        # Process the permutation importance results for each scoring metric
        for metric in r_multi:
            r = r_multi[metric]

            # Lists to store the mean and standard deviation of importances for each feature
            importances_mean = []
            importances_std = []
            
            # Sort features by their mean importance in descending order and iterate over them
            for i in r.importances_mean.argsort()[::-1]:
                # Check if the feature's importance is statistically significant
                if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
                    importances_mean.append(r.importances_mean[i])
                    importances_std.append(r.importances_std[i])
                else:
                    # If importance is not significant, append 0 or NaN
                    importances_mean.append(0)
                    importances_std.append(0)
            
            # Add the computed mean and standard deviation to the dictionary for the current metric
            feature_importance_dict[f'{metric}_mean'] = importances_mean
            feature_importance_dict[f'{metric}_std'] = importances_std

        # Extract the name of the current model for display purposes
        if hasattr(fit_model, 'named_steps'):
            model_name, _ = next(reversed(fit_model.named_steps.items()))

        # Convert the importance data dictionary into a DataFrame for easier visualization
        feature_importance_df = pd.DataFrame(feature_importance_dict, index=feature_names)

        # Filter the DataFrame to include rows containing non-zero values
        filtered_df = feature_importance_df[(feature_importance_df > 0).any(axis=1)]
        
        #Display the results for the current model
        if hasattr(fit_model, 'named_steps'):
            display(HTML(f'<h4>{model_name} </h4>'))
        display(filtered_df)  # Display the filtered DataFrame


def build_pipe_evaluate_bin_clf(models, X, y, test_size=0.25, stratify=True, rs=42, drop_cols=[], transformer=None, scaler=None, selector=None, summary=False):
    """
    Evaluate one or more machine learning models on given data.
    
    Parameters:
    - models: dict or single  model. If dict, keys are model names and values are model instances.
    - X_train, y_train, X_test, y_test: Training and test datasets.
    - transformer: Data transformer (optional).
    - scaler: Data scaler (optional).
    - selector: Feature selector (optional).
    
    Returns:
    - results_df: DataFrame containing performance metrics.
    """
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
    #'Train ROC AUC': [],
    #'Test ROC AUC': [],
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
        """results['Train ROC AUC'].append(train_roc_auc)
        results['Test ROC AUC'].append(test_roc_auc)"""

    #Create results dataframe using the arrays of metrics
    results_df = pd.DataFrame(results)
    if summary:
        display(results_df)
    return results_df, fit_models


def build_transf_evaluate_bin_clf(models, X, y, test_size=0.25, stratify=False, rs=42,
                                  ohe_cols=[], binary_cols=[], ordinal_cols=[], numerical_cols=[],
                                  scaler=StandardScaler(), selector=None):
    
    X=X.copy(deep=True)[ohe_cols+binary_cols+ordinal_cols+numerical_cols]
    
    # Split Data into Train and Test Sets
    if stratify:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=rs)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=None, random_state=rs)
    
    simple_numeric_pp = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ])
    
    transformer = ColumnTransformer(
    transformers=[
        ('num_pp', simple_numeric_pp, numerical_cols),
        ('ohe', OneHotEncoder(drop='if_binary', sparse_output=False, handle_unknown='ignore'), ohe_cols),
        ('binary', BinaryEncoder(), binary_cols),
        ('ordinal', OrdinalEncoder(), ordinal_cols)
    ], remainder='passthrough')
    return build_pipe_evaluate_bin_clf(models, X_train, y_train, X_test,y_test, 
                                       transformer=transformer, 
                                       scaler=scaler, 
                                       selector=selector)


def cv_and_holdout(estimator,X, y, test_size=0.25, stratify=None, random_state=42, search_type='halving_random', param_dict=None,
                  scoring=None, refit=None, refit_scorer=None, holdout_tolerance=0, verbose=0, cv=5, n_iter=10, factor=3, summary=True):
    pd.set_option('display.max_columns', None)
    """
    Perform cross-validation and holdout validation on a given estimator.
    
    Parameters:
        estimator: scikit-learn estimator object
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
        ho_results: DataFrame containing holdout results
        best_holdout_estimator: Best estimator based on holdout validation
    """
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
        estimator.set_params(**candidate_params)
        estimator.fit(X_train, y_train)
        
        holdout_train_score = refit_scorer(estimator, X_train, y_train)
        holdout_test_score = refit_scorer(estimator, X_test, y_test)
        
        if holdout_train_score > (1+holdout_tolerance)*holdout_test_score: 
            if holdout_test_score > best_holdout_score:
                best_holdout_score = holdout_test_score
                best_holdout_estimator = estimator
        
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
        display(cv_results.iloc[:4,:].style.hide_index())
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
    display(output_df.style.hide_index())

    return output_df, models

def build_and_run_pipes (df,target,scoring_metrics, search_type,
                   num_imputer, 
                   num_imputer_params, 
                   num_transformer, 
                   num_transformer_params,
                   poly, 
                   poly_params, 
                   num_cols,
                   
                   cat_imputer, 
                   cat_imputer_params, 
                   cat_combiner,
                   cat_combiner_params,
                   cat_encoder, 
                   cat_encoder_params,
                   cat_cols,  
                   ohe_drop,
                   ohe_cols, 
                   ord_cols, 
                   
                   oversampler, 
                   over_params,
                   undersampler, under_params, 
                   
                   scaler, 
                   selector, 
                   selector_params,
                   
                   estimator_dicts,
                   
                   refit,
                   cv=5, n_iter=10, summary=True, verbose=1,
                   test_size=0.25, stratify=None,rs=42,factor=3):
    X = df[ohe_cols+ord_cols+cat_cols+num_cols]
    y = df[target]

    num_pp_steps=[]
    if num_imputer is not None or num_transformer is not None or poly is not None:
        if num_imputer is not None:
            num_pp_steps.append(num_imputer)
        if num_transformer is not None: 
            num_pp_steps.append(num_transformer)
        if poly is not None: 
            num_pp_steps.append(poly)
            
    cat_pp_steps=[]
    if cat_imputer is not None: 
        cat_pp_steps.append(cat_imputer)
    if cat_combiner is not None:
        cat_pp_steps.append(cat_combiner)
    if cat_encoder is not None:
        cat_pp_steps.append(cat_encoder)
        
    transformer_list=[]
    if num_pp_steps != []: 
        transformer_list.append( ('num', Pipeline(steps=num_pp_steps), num_cols) )
    if cat_pp_steps !=[]:
        transformer_list.append( ('cat', Pipeline(steps=cat_pp_steps), cat_cols) )
    
    if ohe_cols is not []: 
        transformer_list.append(('ohe', OneHotEncoder(drop=ohe_drop, sparse_output=True), ohe_cols)) 
        
    if ord_cols is not []: 
        transformer_list.append(('ord', OrdinalEncoder(), ord_cols))
    
    transformer=None
    if transformer_list!=[]:
        transformer=ColumnTransformer(transformers=transformer_list)
    """
    transformer=ColumnTransformer(transformers=[
        ('num', Pipeline(steps=num_pp_steps), num_cols) if num_pp_steps is not None else None,
        ('cat', Pipeline(steps=cat_pp_steps), cat_impute_cols) if cat_pp_steps is not None else None,
        ('ohe', OneHotEncoder(), ohe_cols),
        ('ord', OrdinalEncoder(), ord_cols),
        ('bin', BinaryEncoder(), bin_cols)
    ])"""
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
    
    pipe_param_pairs=[]
    for est_dict in estimator_dicts: 
        est_tuple, est_params=est_dict['est_tuple'], est_dict['est_params']    
        pipe_steps=list(fixed_pipe_steps)    
        pipe_steps.append( (est_tuple) )
    
        pipe = ImbPipeline(steps=pipe_steps) if oversampler is not None or undersampler is not None else Pipeline(steps=pipe_steps)

        pp_params={}
        if num_imputer_params is not None: 
            pp_params=pp_params | num_imputer_params
        if num_transformer_params is not None: 
            pp_params=pp_params | num_transformer_params
        if poly_params is not None:
            pp_params=pp_params | poly_params
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
        if selector_params is not None:
            pp_params=pp_params | selector_params
        
        pipe_param_pairs.append( (pipe, pp_params | est_params) )
  
    results, models = run_pipelines(pipe_param_pairs = pipe_param_pairs,
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
    
    return results, models          

def quick_cross_val_score(fit_models, X, y, scoring, excluded_cols=[], cv=5):
    if excluded_cols !=[]:
        X=X.copy(deep=True).drop(columns=excluded_cols)
    model_names, scores=[],[]
    for model in fit_models:
        model_names.append(model.__class__.__name__)
        scores.append(cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1).mean())  
    return pd.DataFrame({'Model': model_names, 'Score': scores})
        

def get_lgr_pipe_coefs(pipe, transf_name='transformer', scaler_name='scaler', selector_name=None):
    #Define coefficients and intercept
    steps_list = list(pipe.named_steps.items())
    model_name, _ = steps_list[-1]
    my_coefs = pipe.named_steps[model_name].coef_[0]
    intercept = pipe.named_steps[model_name].intercept_
    #Features from Transformer
    my_features = pipe.named_steps[transf_name].get_feature_names_out()
    
    if selector_name:
        #Create mask for remaining features after selectfrommodel 
        remaining_feature_mask = pipe.named_steps.selector.get_support()
        #Set remaining features
        remaining_features = np.array(my_features)[remaining_feature_mask]
        
        # Get the means and standard deviations
        scaler = pipe.named_steps[scaler_name]
        means = scaler.mean_[remaining_feature_mask]
        std_devs = scaler.scale_[remaining_feature_mask]  # standard deviation
        
    else: 
        # If there is no selector, all features remain
        remaining_features = my_features
        # Get the means and standard deviations for all features
        scaler = pipe.named_steps[scaler_name]
        means = scaler.mean_
        std_devs = scaler.scale_
        
    # Create the dataframe with coefficients, means, and std_devs
    interpretation_df = pd.DataFrame({
    'coefs': my_coefs,
    'means': means,
    'std_devs': std_devs, 
    'exp_unscaled_coefs': np.exp(my_coefs/std_devs),
    }, index=remaining_features)
    
    #Sort the dataframe
    return interpretation_df.sort_values(by='exp_unscaled_coefs', ascending=False)