import sys
sys.path.append('/Users/basilhaddad/jupyter/capstone/')
from importlib import reload
from helpers.my_imports import * 
from IPython.core.display import HTML

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
    #order column groups
    
    #if after, place grouped cols after remaining cols and vice versa
    if after:
        new_col_order = remaining_cols + grouped_cols
    else: 
        new_col_order = grouped_cols + remaining_cols
    
    return df[new_col_order]




def build_pipe_evaluate_bin_clf(models, X_train, y_train, X_test,y_test, transformer=None, scaler=None, selector=None):
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
   
        """# Encoding labels to numeric values, 1 for Positive Class and 0 for Negative class
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)

        # Train the model with the training data
        model.fit(X_train, y_train_encoded)

        # Predict probabilities for the test data
        # The predicted probabilities are for class 1, hence the use of [:, 1]
        y_train_probas = model.predict_proba(X_train)[:, 1]
        y_test_probas = model.predict_proba(X_test)[:, 1]

        # Now we can calculate the ROC AUC score using the true binary labels and the predicted probabilities
        train_roc_auc = roc_auc_score(y_train_encoded, y_train_probas)
        test_roc_auc = roc_auc_score(y_test_encoded, y_test_probas)"""

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
    return results_df, fit_models

def build_transf_evaluate_bin_clf(models, X, y, test_size=0.25, stratify=None, rs=42,
                                  ohe_cols=[], binary_cols=[], ordinal_cols=[], numerical_cols=[],scaler=StandardScaler(), selector=None):
    X=X[ohe_cols+binary_cols+ordinal_cols+numerical_cols]
    X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=stratify, test_size=test_size, random_state=rs)
    
    transformer = ColumnTransformer(
    transformers=[
        ('ohe', OneHotEncoder(drop='if_binary', sparse_output=False, handle_unknown='ignore'), ohe_cols),
        ('binary', BinaryEncoder(), binary_cols),
        ('ordinal', OrdinalEncoder(), ordinal_cols)
    ], remainder='passthrough')
    return build_pipe_evaluate_bin_clf(models, X_train, y_train, X_test,y_test, 
                                       transformer=transformer, 
                                       scaler=scaler, 
                                       selector=selector)

"""def build_transf_evaluate_bin_clf(df_orig, target, models, dropna=False, test_size=0.25, stratify=None, rs=42,
                                  ohe_cols=[], binary_cols=[], ordinal_cols=[], numerical_cols=[], selector=None):
    df = df_orig.copy(deep=True)
    df.dropna(inplace=dropna)     
    X,y=df.drop(columns=target), df[target]
    X=X[ohe_cols+binary_cols+ordinal_cols+numerical_cols]
    X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=stratify, test_size=0.25, random_state=42)
    
    transformer = ColumnTransformer(
    transformers=[
        ('ohe', OneHotEncoder(drop='if_binary', sparse_output=False, handle_unknown='ignore'), ohe_cols),
        ('binary', BinaryEncoder(), binary_cols),
        ('ordinal', OrdinalEncoder(), ordinal_cols)
    ], remainder='passthrough')
    return build_pipe_evaluate_bin_clf(models, X_train, y_train, X_test,y_test, 
                                       transformer=transformer, 
                                       scaler=StandardScaler(), 
                                       selector=selector)"""



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


def select_all_col_names_except(df, exclude_list):
    """
    Select all column names from a DataFrame except those specified in an exclusion list.
    
    Parameters:
    - df: pandas DataFrame
    - exclude_list: list of column names to exclude
    
    Returns:
    - List of column names to keep
    """
    # List of all columns
    all_columns = df.columns.tolist()
    # Columns to exclude
    exclude_columns = exclude_list
    # Columns to keep
    return list(set(all_columns) - set(exclude_columns))



"""def reinit_data(df_orig, target, dropna=False, test_size=0.25, stratify=None, rs=42): 
    df = df_orig.copy(deep=True)
    df.dropna(inplace=dropna)     
    X,y=df.drop(columns=target), df[target]
    X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=stratify, test_size=test_size, random_state=rs)"""

def cv_and_holdout(estimator,X, y, test_size=0.25, stratify=None, random_state=42, search_type='halving_random', param_dict=None,
                  scoring=None, refit=None, holdout_tolerance=0, verbose=0, cv=5, n_iter=10, factor=3, summary=True):
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
    if refit in scoring and isinstance(scoring, dict): #assume refit scorer is one of the item in the dict
        refit_scorer = scoring[refit]
    elif search_type=='halving_random' and isinstance(scoring,str): #can probably make this more dynamic at another time
        refit=scoring
        if scoring=='f1_weighted': 
            refit_scorer = make_scorer(f1_score, average='weighted', zero_division='warn')
        elif scoring=='precision_weighted':
            refit_scorer = make_scorer(precision_score, average='weighted', zero_division='warn')
    else:
        raise ValueError(f"The refit metric {refit} was not found in the scoring_metrics dictionary.")

    # Step 1: Split Data into Train and Test Sets and Run GridSearchCV or RandomizedSearchCV
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=None, random_state=random_state)

    if search_type == 'grid':
        search = GridSearchCV(estimator, param_dict, scoring=scoring, refit=refit, cv=cv, n_jobs=3)
    elif search_type == 'random':
        search = RandomizedSearchCV(estimator, param_dict, n_iter=n_iter, scoring=scoring, refit=refit, random_state=random_state, verbose=verbose, cv=cv, n_jobs=3)
    elif search_type == 'halving_random':
        search = HalvingRandomSearchCV(estimator,param_dict,factor=factor,scoring=scoring,refit=True,random_state=random_state,
                                       verbose=verbose, cv=cv, n_jobs=3)
    
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
    cv_results= reorder_cols_in(cv_results, [cv_rank_refit_col, cv_test_score_refit_col, 'is_overfit', ho_train_score_refit_col, ho_test_score_refit_col, 
                                               'mean', 'param'] , after=False) 
    
    
    #Step 5: Define dataframe to display holdout test results ordered by overfit status and descending holdout test scores
    ho_results.sort_values(by=['is_overfit', ho_test_score_refit_col, cv_rank_refit_col], ascending=[True, False, True], inplace=True)
    ho_results.reset_index(drop=True, inplace=True)
    # Create the new holdout rank column based on the new index
    ho_results[ho_rank_refit_col] = ho_results.index + 1  
      
    #Show most important columns first
    ho_results= reorder_cols_in(ho_results, [cv_rank_refit_col, cv_test_score_refit_col, 'is_overfit', ho_train_score_refit_col, ho_test_score_refit_col,  
                                              'mean', 'param'] , after=False) #'mean_fit_time', 'mean_score_time'

    #Step 6: Display Best CV Model Details and Best Holdout Model Details
    if summary:
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




def run_pipelines(pipe_param_pairs, X, y, test_size = 0.25, stratify=None, random_state=42, search_type='halving_random', 
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
                   poly, 
                   poly_params, 
                   num_cols,
                   
                   cat_imputer, 
                   cat_imputer_params, 
                   cat_impute_cols,  
                   
                   ohe_cols, bin_cols, ord_cols, 
                   
                   oversampler, over_params,
                   undersampler, under_params, 
                   
                   scaler, 
                   selector, selector_params,
                   estimator_dicts,
                   
                   refit,cv=5, n_iter=10, summary=True, verbose=1,
                   test_size=0.25, stratify=None,rs=42,factor=3):
    X = df.copy(deep=True)[ohe_cols+bin_cols+ord_cols+num_cols]
    y = df[target]

    num_pp_steps=[]
    if num_imputer is not None or poly is not None:
        if num_imputer is not None:
            num_pp_steps.append(num_imputer)
        if poly is not None: 
            num_pp_steps.append(poly)
            
    cat_pp_steps=[]
    if cat_imputer is not None: 
        cat_pp_steps.append(cat_imputer)
        
    transformer_list=[]
    if num_pp_steps != []: 
        transformer_list.append( ('num', Pipeline(steps=num_pp_steps), num_cols) )
    if cat_pp_steps !=[]:
        transformer_list.append( ('cat', Pipeline(steps=cat_pp_steps), cat_impute_cols) )
    
    if ohe_cols is not None: 
        transformer_list.append(('ohe', OneHotEncoder(), ohe_cols))

    if bin_cols is not None: 
        transformer_list.append(('bin', BinaryEncoder(), bin_cols))        

    if ord_cols is not None: 
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
        #pp_params = num_imputer_params | cat_imputer_params | poly_params | over_params | under_params | selector_params
        pp_params={}
        if num_imputer_params is not None: 
            pp_params=pp_params | num_imputer_params
        if cat_imputer_params is not None: 
            pp_params=pp_params | cat_imputer_params
        if poly_params is not None:
            pp_params=pp_params | poly_params
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
















def custom_imputer_col(X, col_to_impute, missing_value, pipe_step= ('lgr', LogisticRegression(n_jobs=-1, max_iter=1000, random_state=42))):
    """
    Impute missing values in a specific column
    
    Parameters:
    - X: pandas DataFrame, the dataset containing missing values
    - col_to_impute: str, the column with missing values to be imputed
    - missing_value: the value representing missing data in the column
    - pipe_step: tuple, additional step to add to the pipeline (default is logistic regression)
    
    Returns:
    - X: pandas DataFrame, the dataset with imputed values
    """
    # Separate data into known and unknown target values
    X_target_unknown = X[X[col_to_impute] == missing_value]
    X_target_known = X[X[col_to_impute] != missing_value]
    
    #Drop column to be imputed
    X_unknown = X_target_unknown.drop(columns=[col_to_impute])
    
    # Define Features and targets where target is known. Drop column to be imputed
    X_known = X_target_known.drop(columns=[col_to_impute])
    y_known = X_target_known[col_to_impute]
    
    # Split the known data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_known, y_known, stratify = y_known, random_state=42)
    
    #Define columns for one hot encoding.  Note categorical_cols was already stripped of the col to impute
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
    #numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

    #Define simple transformer 
    transformer = ColumnTransformer(
    transformers=[
        #('poly', PolynomialFeatures(include_bias=False, degree =2), numerical_cols), 
        ('ohe', OneHotEncoder(drop='if_binary', sparse_output=False), categorical_cols),
    ], remainder='passthrough')
    
    #Define simple pipeline 
    pipe = Pipeline([
    ('transformer', transformer), 
    ('scaler', StandardScaler()), 
    #('selector', SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000, random_state=42)) ),
    pipe_step
    ])
    
    #Evaluate Imputer
    pipe.fit(X_train, y_train)
    print("\nImputer Scores: ")
    print(f"Train Score: {pipe.score(X_train, y_train):.5f}")
    print(f"Test Score: {pipe.score(X_test, y_test):.5f}")
    
    print(f"\nValues before Imputing:")
    print(X[col_to_impute].value_counts())
        
    #Impute using entire dataset
    pipe.fit(X_known, y_known)
    imputed_values = pipe.predict(X_unknown)
    
    X_imputed = X.copy()
    X_imputed.loc[ X[X[col_to_impute] == missing_value].index, col_to_impute] = imputed_values
    
    print(f"\nValues after Imputing:")
    print(X_imputed[col_to_impute].value_counts())
    return X_imputed

def custom_imputer(X, col_missing_value_dict, 
                   pipe_step= ('lgr', LogisticRegression(n_jobs=-1, max_iter=1000, random_state=42))):
    """
    Impute missing values in multiple columns 
    
    Parameters:
    - X: pandas DataFrame, the dataset containing missing values
    - col_missing_value_dict: dict, a dictionary where keys are the columns to be imputed and 
                              values are the missing values in those columns
    - pipe_step: tuple, additional step to add to the pipeline for imputation 
                 (default is logistic regression)
    
    Returns:
    - X_imputed: pandas DataFrame, the dataset with imputed values in specified columns
    """
    # Create a copy of the original DataFrame to store the imputed value
    X_imputed = X.copy()
    # Loop through each column and its corresponding missing value and call the custom_imputer_col function
    for col_to_impute, missing_value in col_missing_value_dict.items():
        print("="*20)
        print(f"Column to Impute: {col_to_impute}   Missing Value: {missing_value}")
        X_imputed = custom_imputer_col(X_imputed, col_to_impute, missing_value,pipe_step= pipe_step)
    return X_imputed

#perhaps use y instead of df and target_col
def starter_pipes(df, target_col, regressor, X_train= None, imbalanced=False, range_extend=0.2):
    n = df.shape[0]
    m = df.shape[1]
    
    print(f"Target value counts: {df['target_col'].value_counts(normalize=True)}")
    
    if isinstance(regressor, KNeighborsClassifier):
        if imbalanced: 
            params = {
                'knn__n_neighbors': randint(3, np.sqrt(n)+(n*range_extend)), #Rule of Thumb: Sqrt of N
                'knn__weights':  ['uniform', 'distance'], #use distance for imbalanced classes
                'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'knn__leaf_size': randint(5,100),  #default = 30
                'knn__p': [1,2],
                'knn__metric_params': [{'V': np.cov(X_train)}],
                'knn__metric': ['mahalanobis']   
                }   
        else:      
            params = {
            'knn__n_neighbors': randint(3, np.sqrt(n)+(n*range_extend)), #Rule of Thumb: Sqrt of N
            'knn__weights':  ['uniform', 'distance'], #use distance for imbalanced classes
            'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'knn__leaf_size': randint(5,100),  #default = 30
            'knn__p': [1,2],
            'knn__metric': ['minkowski', 'euclidean', 'manhattan']  
            }    
            