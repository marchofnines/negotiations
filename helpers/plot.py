"""
Author: Basil Haddad
Date: 11.01.2023

Description:
    Helper functions for visualizations
"""

from importlib import reload
from helpers.my_imports import *


def reverse_palette(df, hue, base_palette):
    """
    Generate a reversed color palette mapped to the unique values of a specified column. I created this
    function because the regular palette was making the majority class darker than the minority class
    which made it difficult to see the hue of the data in histograms 

    This function creates a color palette based on the unique values in a specified column 
    ('hue') of a DataFrame. It assigns colors in reverse order of the frequency of the 
    unique values.

    Parameters:
    df (DataFrame): The DataFrame containing the data.
    hue (str): The name of the column for which the palette is to be created.
    base_palette (str or list): The name of the seaborn palette or a list of colors to use as the base.

    Returns:
    dict: A dictionary mapping each unique value in the 'hue' column to a color from the reversed palette.

    Note:
    - The number of colors in the palette is equal to the number of unique values in the 'hue' column.
    """

    # Determine the order of unique values based on their frequency
    class_order = df[hue].value_counts().index
    # Count the number of unique values in the hue column
    n_classes = df[hue].nunique()

    # Generate a color palette and reverse its order
    magma_palette = sns.color_palette(base_palette, n_classes)
    magma_palette.reverse()

    # Map each unique value to a color from the reversed palette
    return {class_order[i]: magma_palette[i] for i in range(n_classes)}

    
def sns_categorical(df,feature,feature_label,hue, legend_title, rate_descr, figx=17, figy=8, common_fontsize=19.5, xtick_divisor=1, rotation=0):
    """
    Create a two-part seaborn plot with a categorical distribution and rate visualization.

    This function generates a two-row plot using seaborn. The first row displays a categorical
    count distribution, and the second row shows a bar plot representing rates (e.g., acceptance rates)
    for different categories. The function is customizable with various plotting parameters.

    Parameters:
    df (DataFrame): The DataFrame containing the data to plot.
    feature (str): The column name in the DataFrame to be used as the x-axis.
    feature_label (str): Label for the x-axis.
    hue (str): Column name to be used for color encoding (hue).
    legend_title (str): Title for the legend.
    rate_descr (str): Description for the rate being visualized (e.g., 'Acceptance').
    figx (int): Width of the figure. Defaults to 17.
    figy (int): Height of the figure. Defaults to 8.
    common_fontsize (float): Font size for text elements. Defaults to 19.5.
    xtick_divisor (int): Divisor for x-tick frequency. Defaults to 1.
    rotation (int): Rotation angle for x-tick labels. Defaults to 0.

    Returns:
    None: The function only generates plots and does not return any value.
    """
   
    ### Define df_viz (deep copy) to facilitiate plotting acceptance rates
    df_viz = df.copy(deep=True)
    # Map 'decision' column to numeric values for rate calculation
    df_viz.decision = df_viz['decision'].map({'Rejected': 0, 'Accepted':1})
   
    # Set up color palette and plotting style
    magma_palette = sns.color_palette('magma', 10)
    sns.set(style="whitegrid", font_scale=1)
    
    # Set figure size and common font size for labels and ticks
    plt.figure(figsize=(figx, figy))  
    common_fontsize =common_fontsize
    
    # Create the figure and the first subplot (categorical count distribution)
    plt.subplot(2,1,1)
    sns.countplot(data=df, x=feature,hue=hue, palette='magma')
    # Set title, labels, and customize x and y ticks
    plt.title(f'{feature_label} Distribution by {legend_title}',fontsize=common_fontsize)
    plt.xlabel(f'',fontsize=common_fontsize)
    plt.xticks(ticks=plt.xticks()[0][::xtick_divisor], labels=plt.xticks()[1][::xtick_divisor], fontsize=common_fontsize, rotation=rotation)
    plt.ylabel('Count',fontsize=common_fontsize)
    plt.yticks(fontsize=common_fontsize)
    plt.legend(title=f'{legend_title}', title_fontsize=common_fontsize, fontsize=common_fontsize) 

    # Create the second subplot (bar plot for rates)
    plt.subplot(2, 1, 2)
    ax2=sns.barplot(data=df_viz, x=feature,y=hue, color=magma_palette[5]) ##color='#1f77b4')
    # Set title, labels, and customize x and y ticks
    plt.title(f'{rate_descr} Rate by {feature_label}',fontsize=common_fontsize)
    plt.xlabel(f'{feature_label}',fontsize=common_fontsize)
    plt.ylabel(f'{rate_descr} Rate',fontsize=common_fontsize)
    plt.xticks(ticks=plt.xticks()[0][::xtick_divisor], labels=plt.xticks()[1][::xtick_divisor], fontsize=common_fontsize, rotation=rotation)
    plt.yticks(fontsize=common_fontsize)

    # Adding percentages on the top of bars for the second plot
    for p in ax2.patches:
        ax2.annotate(f'{100 * p.get_height():.1f}%', 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha='center', va='center', 
                     fontsize=common_fontsize, 
                     color='black', 
                     xytext=(0, 25), 
                     textcoords='offset points')

    # Adjust layout and add a main title
    plt.tight_layout(h_pad=1.09)
    banner_y = 1.05
    plt.suptitle(f"{feature_label} Distribution and {rate_descr} Rates", 
                        fontsize=common_fontsize + 4, weight='bold', 
                        x=0.1, y=banner_y-0.02, ha='left')    
    plt.show()

def sns_boxplot(df, feature, feature_label, hue, xmin=None, xmax=None, figx=17, figy=6, common_fontsize=19.5):
    """
    Create a boxplot using seaborn with various customization options.

    This function generates a boxplot for a specified feature and hue. It allows
    customization of the plot dimensions, font sizes, and x-axis limits. The
    function uses a reversed color palette for visual distinction.

    Parameters:
    df (DataFrame): The pandas DataFrame containing the data.
    feature (str): The name of the column in df to be plotted on the x-axis.
    feature_label (str): The label for the x-axis.
    hue (str): The name of the column in df to determine box colors.
    xmin (float, optional): Minimum limit for the x-axis. Defaults to None.
    xmax (float, optional): Maximum limit for the x-axis. Defaults to None.
    figx (int, optional): Width of the figure. Defaults to 17.
    figy (int, optional): Height of the figure. Defaults to 6.
    common_fontsize (float, optional): Font size for plot labels and ticks. Defaults to 19.5.

    Returns:
    None: The function creates and displays a boxplot but does not return any value.
    """

    # Set the common font size for the plot
    common_fontsize = common_fontsize

    # Initialize the plot with specified dimensions
    plt.figure(figsize=(figx, figy))
    sns.set(style="whitegrid", font_scale=1)

    # Generate a reversed color palette for the hue
    palette = reverse_palette(df, hue, 'magma')

    # Create the boxplot with specified data, feature, and hue
    sns.boxplot(data=df, x=feature, y=hue, palette=palette)

    # Set plot titles and labels with the specified font size
    plt.title(f'{feature_label} (Full Range)', fontsize=common_fontsize)
    plt.xlabel(feature_label, fontsize=common_fontsize)
    plt.ylabel('Count', fontsize=common_fontsize)

    # Set the font size for x and y ticks
    plt.xticks(fontsize=common_fontsize)
    plt.yticks(fontsize=common_fontsize)

    # Set x-axis limits if specified
    plt.xlim(xmin, xmax)

    # Display the plot
    plt.show()

    

def sns_numerical(df, feature, feature_label, hue, legend_title, rate_descr, binwidth=None, xmin=None, xmax=None, ymin=None, ymax=None, figx=17, figy=8, common_fontsize=19.5):
    """
    Create a two-part seaborn plot for numerical data visualization.

    This function generates a two-column plot using seaborn. The first column displays a boxplot,
    and the second column shows a histogram. It allows for customization of plot dimensions,
    font sizes, axis limits, and bin width for the histogram. The function uses a reversed color
    palette for the hue.

    Parameters:
    df (DataFrame): The pandas DataFrame containing the data.
    feature (str): The column name in df to be plotted on the x-axis.
    feature_label (str): Label for the x-axis.
    hue (str): Column name to determine box and bar colors.
    legend_title (str): Title for the legend.
    rate_descr (str): Description for the rate being visualized.
    binwidth (int, optional): Width of the bins for the histogram. Defaults to None.
    xmin (float, optional): Minimum limit for the x-axis. Defaults to None.
    xmax (float, optional): Maximum limit for the x-axis. Defaults to None.
    ymin (float, optional): Minimum limit for the y-axis. Defaults to None.
    ymax (float, optional): Maximum limit for the y-axis. Defaults to None.
    figx (int, optional): Width of the figure. Defaults to 17.
    figy (int, optional): Height of the figure. Defaults to 8.
    common_fontsize (float, optional): Font size for plot labels and ticks. Defaults to 19.5.

    Returns:
    None: The function creates and displays the plots but does not return any value.
    """
   
    # Initialize the figure with specified dimensions
    plt.figure(figsize=(figx, figy))  
    sns.set(style="whitegrid", font_scale=1)
    # Generate a reversed color palette for the hue
    palette = reverse_palette(df, hue, 'magma')
    
    # First subplot: Create a boxplot for numerical distribution
    plt.subplot(1,2,1)
    ax1=sns.boxplot(data=df, x=feature, y=hue, palette=palette)
    # Set plot title, labels, and customize x and y ticks
    plt.title(f'{feature_label} (Full Range)',fontsize=common_fontsize)
    plt.xlabel(f'{feature_label}',fontsize=common_fontsize)
    plt.ylabel('Count',fontsize=common_fontsize)
    plt.xticks(fontsize=common_fontsize)
    plt.yticks(fontsize=common_fontsize)
    ax1.xaxis.get_offset_text().set_size(common_fontsize)
    
    #Second subplot: Create a histogram for numerical distribution
    plt.subplot(1, 2, 2)
    ax2=sns.histplot(data=df, x=feature,hue=hue, binwidth = binwidth,label=legend_title, palette=palette)
    # Set plot title, labels, and customize x and y ticks
    plt.title(f'{feature_label} ({xmin},{xmax})',fontsize=common_fontsize)
    plt.xlabel(f'{feature_label}',fontsize=common_fontsize)
    plt.ylabel('Count',fontsize=common_fontsize)
    plt.xticks(fontsize=common_fontsize)
    plt.yticks(fontsize=common_fontsize)
    # Set axis limits if specified
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    
    # Customize the legend
    legend = ax2.get_legend()
    legend.set_title(legend_title, prop={'size': common_fontsize-1})
    for text in legend.get_texts():
        text.set_fontsize(common_fontsize-1)
    
    # Adjust the overall layout and add a main title for the figure
    plt.tight_layout(h_pad=1.09)
    banner_y = 1.07
    plt.suptitle(f"Distribution of {feature_label} (by {legend_title})", 
                        fontsize=common_fontsize + 4, weight='bold', 
                        x=0.5, y=banner_y-0.02, ha='center')    
    plt.show()



def myhist(df, feature, feature_label, hue, legend_title, legend_labels, binwidth=None, xmin=None, xmax=None,figx=18, figy=8,common_fontsize=15):
    """
    Generate a histogram plot using seaborn with customization options.

    This function creates a histogram for a specified feature and hue. It allows
    customization of the plot dimensions, font sizes, bin width, and axis limits.
    The function uses a reversed color palette based on the class order.

    Parameters:
    df (DataFrame): The pandas DataFrame containing the data.
    feature (str): The column name in df to be plotted on the x-axis.
    feature_label (str): Label for the x-axis.
    hue (str): Column name to determine bar colors in the histogram.
    legend_title (str): Title for the legend.
    legend_labels (list): Labels for the legend, corresponding to hue categories.
    binwidth (int, optional): Width of the bins for the histogram. Defaults to None.
    xmin (float, optional): Minimum limit for the x-axis. Defaults to None.
    xmax (float, optional): Maximum limit for the x-axis. Defaults to None.
    figx (int, optional): Width of the figure. Defaults to 18.
    figy (int, optional): Height of the figure. Defaults to 8.
    common_fontsize (float, optional): Font size for plot labels and ticks. Defaults to 15.

    Returns:
    None: The function creates and displays a histogram but does not return any value.
    """
    # Initialize the figure with specified dimensions
    plt.figure(figsize=(figx,figy))
    sns.set(style="whitegrid", font_scale=1)
   
    # Determine the order of classes based on volume and assign colors accordingly
    class_order = df[hue].value_counts().index
    n_classes = df[hue].nunique()
    
    # Generate a color palette and reverse its order for visual distinction
    magma_palette = sns.color_palette('magma', n_classes)
    magma_palette.reverse()
    
    # Create the custom palette dictionary
    palette = {class_order[i]: magma_palette[i] for i in range(n_classes)}
    
    # Create the plot
    sns.histplot(data=df, x=feature, hue=hue, binwidth=binwidth, legend=True, palette=palette)
    
    # Customize labels and title
    plt.title(f'Distribution of {feature_label} by {legend_title}', fontsize=common_fontsize)
    plt.xlabel(feature_label, fontsize=common_fontsize)
    plt.ylabel('Count', fontsize=common_fontsize)
    plt.xticks(fontsize=common_fontsize)
    plt.yticks(fontsize=common_fontsize)
    plt.xlim(xmin,xmax)  if xmin is not None and xmax is not None else None
    
    # Customize the legend
    legend = plt.legend(title=legend_title, labels=legend_labels)
    legend.get_title().set_fontsize(common_fontsize)  # Legend title fontsize
    for t in legend.texts:
        t.set_fontsize(common_fontsize)  # Legend label fontsize
    plt.tight_layout()  
    plt.show()

def hist_orig_transf(df, feature, transformer, feature_label, hue, legend_title, legend_labels, binwidth_left=None, binwidth_right=None, xmin=None, xmax=None, ymin=None, ymax=None, figx=18, figy=8,common_fontsize=15):
    """
    Generate side-by-side histograms for original and transformed data.

    This function creates a two-column plot using seaborn. The first column displays a histogram
    for the original data, and the second column shows a histogram for the transformed data.
    It allows customization of the plot dimensions, font sizes, bin width, axis limits, and 
    applies a transformation to the feature of interest.

    Parameters:
    df (DataFrame): The pandas DataFrame containing the data.
    feature (str): The column name in df to be plotted and transformed.
    transformer (function): A function to apply transformation to the feature.
    feature_label (str): Label for the x-axis.
    hue (str): Column name to determine bar colors in the histogram.
    legend_title (str): Title for the legend.
    legend_labels (list): Labels for the legend, corresponding to hue categories.
    binwidth_left (int, optional): Width of the bins for the original data histogram. Defaults to None.
    binwidth_right (int, optional): Width of the bins for the transformed data histogram. Defaults to None.
    xmin (float, optional): Minimum limit for the x-axis. Defaults to None.
    xmax (float, optional): Maximum limit for the x-axis. Defaults to None.
    figx (int, optional): Width of the figure. Defaults to 18.
    figy (int, optional): Height of the figure. Defaults to 8.
    common_fontsize (float, optional): Font size for plot labels and ticks. Defaults to 15.

    Returns:
    None: The function creates and displays the histograms but does not return any value.
    """
    # Initialize the figure with specified dimensions
    plt.figure(figsize=(figx,figy))
    sns.set(style="whitegrid", font_scale=1)
    
    # Create a copy of the DataFrame and apply the transformation to the specified feature
    dft=df.copy(deep=True)
    dft[f'transformed_{feature}']= transformer(dft[feature])
   
    # Determine the order of classes based on volume and assign colors accordingly
    class_order = df[hue].value_counts().index
    n_classes = df[hue].nunique()
    
    # Generate a color palette and reverse its order for visual distinction
    magma_palette = sns.color_palette('magma', n_classes)
    magma_palette.reverse()
    
    # Create the custom palette dictionary
    palette = {class_order[i]: magma_palette[i] for i in range(n_classes)}
    
    # First subplot for original data
    plt.subplot(1,2,1)
    ax1=sns.histplot(data=dft, x=feature, hue=hue, binwidth=binwidth_left, legend=True, palette=palette)
    plt.title(f'Distribution of {feature_label} by {legend_title}', fontsize=common_fontsize)
    plt.xlabel(feature_label, fontsize=common_fontsize)
    plt.ylabel('Count', fontsize=common_fontsize)
    plt.xticks(fontsize=common_fontsize)
    plt.yticks(fontsize=common_fontsize)
    plt.xlim(xmin,xmax)  if xmin is not None and xmax is not None else None
    plt.ylim(ymin,ymax)  if ymin is not None and ymax is not None else None
    
    # Second subplot for transformed data
    plt.subplot(1,2,2)
    ax2=sns.histplot(data=dft, x=f'transformed_{feature}', hue=hue, binwidth=binwidth_right, legend=True, palette=palette)
    plt.title(f'Distribution of Transformed {feature_label} by {legend_title}', fontsize=common_fontsize)
    plt.xlabel(feature_label, fontsize=common_fontsize)
    plt.ylabel('Count', fontsize=common_fontsize)
    plt.xticks(fontsize=common_fontsize)
    plt.yticks(fontsize=common_fontsize)
    #plt.xlim(xmin,xmax)  if xmin is not None and xmax is not None else None
    plt.ylim(ymin,ymax)  if ymin is not None and ymax is not None else None
    
    # Customize the legend
    legend1 = ax1.get_legend()
    legend1.set_title(legend_title, prop={'size': common_fontsize-1})
    for text in legend1.get_texts():
        text.set_fontsize(common_fontsize-1)
        
    legend2 = ax2.get_legend()
    legend2.set_title(legend_title, prop={'size': common_fontsize-1})
    for text in legend2.get_texts():
        text.set_fontsize(common_fontsize-1)

    plt.tight_layout(h_pad=1.09)
    banner_y = 1.07
    plt.suptitle(f"Original vs. Transformed Distribution for {feature_label} ", 
                        fontsize=common_fontsize + 4, weight='bold', 
                        x=0.5, y=banner_y-0.02, ha='center')    
    plt.show()

    
def conf_matrix_PRC(model, X_test, y_test, probas_pos_index, class_labels=[], threshold=0.5, common_fontsize=20, figx=20, figy=8, xlegend=0.1, ylegend=0.1):
    """
    Plot the confusion matrix and Precision-Recall Curve (PRC) for a given model.

    This function visualizes the performance of a binary classification model
    by plotting its confusion matrix and Precision-Recall Curve. It also calculates
    and displays the F1 scores for the default (0.5) and optimal thresholds.

    Parameters:
    model (classifier): Trained classifier model (e.g., from scikit-learn).
    X_test (DataFrame or array): Test features.
    y_test (Series or array): True labels for the test set.
    probas_pos_index (int): Index of the positive class in model.classes_.
    threshold (float, optional): Threshold to use for converting probabilities to class predictions. Defaults to 0.5.
    common_fontsize (int, optional): Font size for plot text elements. Defaults to 20.
    figx (int, optional): Width of the figure. Defaults to 20.
    figy (int, optional): Height of the figure. Defaults to 8.
    xlegend (float, optional): X-coordinate for the legend's position. Defaults to 0.1.
    ylegend (float, optional): Y-coordinate for the legend's position. Defaults to 0.1.

    Returns:
    None: The function creates and displays plots but does not return any value.
    """
    # Identify positive and negative labels from the model's classes
    pos_label = model.classes_[probas_pos_index] 
    neg_label = model.classes_[1 - probas_pos_index]
    
    #Calculate probabilities for the positive class
    y_test_probas = model.predict_proba(X_test)[:,probas_pos_index]

    # Predict class labels based on the specified threshold
    y_test_preds = np.where(y_test_probas >= threshold, pos_label, neg_label) 
    
    # Calculating precision, recall, thresholds and F1 score for each threshold
    precision, recall, thresholds = precision_recall_curve(y_test, y_test_probas, pos_label=pos_label)
    
    # Calculate F1 scores for each threshold
    # Ignoring the last value of precision and recall as they correspond to the lowest threshold (i.e., "recall" is 1)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1])
    
    # Finding the threshold for the optimal F1 score
    optimal_threshold_index = np.argmax(f1_scores)
    #optimal_threshold = thresholds[optimal_threshold_index]
    optimal_precision = precision[optimal_threshold_index]
    optimal_recall = recall[optimal_threshold_index]
     
    # F1 score using the default 0.5 threshold
    #default_precision_index = np.argmin(np.abs(thresholds - 0.5))  # closest precision index for threshold of 0.5
    #default_precision_prc = precision[default_precision_index]
    default_precision = precision_score(y_test,  model.predict(X_test), pos_label=pos_label)
    default_f1 = f1_score(y_test,  model.predict(X_test), pos_label=pos_label)
     
    #Plot the confusion matrix and the ROC AUC curve
    conf_matrix = confusion_matrix(y_test, y_test_preds)
        
    #fix , ax = plt.subplots(1, 2, figsize=(figx, figy))
    plt.subplots(1, 2, figsize=(figx, figy))
    sns.set(style="whitegrid", font_scale=1)

    # Subplot 1: Confusion Matrix
    plt.subplot(1, 2, 1)
    if len(class_labels)!=0:
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='RdPu', annot_kws={"size": 20},
                    xticklabels=class_labels, yticklabels=class_labels)
    else: 
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='RdPu', annot_kws={"size": 20},
                     xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title('Confusion Matrix', fontsize=common_fontsize)
    plt.xlabel('Predicted',fontsize=common_fontsize)
    plt.ylabel('Actual',fontsize=common_fontsize)
    plt.xticks(fontsize=common_fontsize)
    plt.yticks(fontsize=common_fontsize)

    # Subplot 2: Precision-Recall Curve
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f'Precision-Recall Curve (AUC:{auc(recall, precision):.2f})', linewidth=2.7)
    plt.scatter(optimal_recall, optimal_precision, color='red', s=50)
    
    # Calculate the baseline (chance level) for the PRC
    baseline = sum(y_test == model.classes_[probas_pos_index]) / len(y_test)
    plt.axhline(y=baseline, color='gray', linestyle='--', linewidth=2, label=f'No Skill Precision: ({baseline:.2f})') #Random Guessing

    # Plotting a horizontal line for the default threshold's precision
    plt.axhline(y=default_precision, color='blue', linestyle='--',
                label=f'Default (p=0.5), F1:{default_f1:.2f} Precision:{default_precision:.2f}')
   
    # Plotting a horizontal line for the optimal threshold's precision
    #plt.axhline(y=optimal_precision, color='red', linestyle='--',
    #            label=f'Best Tradeoff at p={optimal_threshold:.2f} (Precision:{optimal_precision:.3f})')

    plt.title('Precision-Recall Curve', fontsize=common_fontsize) 
    plt.xlabel('Recall (TP/TP+FN)', fontsize=common_fontsize) #Sensitivity
    plt.ylabel('Precision (TP/TP+FP)', fontsize=common_fontsize)  #PPV
    plt.xticks(fontsize=common_fontsize-2)
    plt.yticks(fontsize=common_fontsize-2)

    #Customize Legend and place in lower left of graph
    plt.legend(bbox_to_anchor=(xlegend, ylegend), loc='lower left', fontsize=common_fontsize*0.77)
    plt.show()


def scores_vs_thresholds(model, X_test, y_test, probas_pos_index, common_fontsize=20, figx=20, figy=8):
    """
    Plots precision, recall, and F1 score as functions of the probability threshold for a given classifier and test data.

    Parameters:
    model (classifier): The trained classifier model which implements the `predict_proba` method.
    X_test (array-like): Test feature data.
    y_test (array-like): True labels for test data.
    probas_pos_index (int): Index of the positive class in the `classes_` attribute of the classifier.
    common_fontsize (int, optional): Font size to be used for the plot's title, labels, and ticks. Defaults to 20.
    figx (int, optional): Width of the figure. Defaults to 20.
    figy (int, optional): Height of the figure. Defaults to 8.

    Returns:
    None: This function does not return anything but plots the precision, recall, and F1 score against the probability thresholds.

    This function assumes that the model has been fitted and can provide probability estimates via `predict_proba`. It plots precision, recall, and F1 scores as functions of various thresholds, helping to visualize the trade-offs between precision and recall for different threshold values.
    """
    
    # Identify positive and negative labels from the model's classes
    pos_label = model.classes_[probas_pos_index] 
    neg_label = model.classes_[1 - probas_pos_index]
    
    #Calculate probabilities for the positive class
    y_test_probas = model.predict_proba(X_test)[:,probas_pos_index]

    # Calculating precision, recall, thresholds and F1 score for each threshold
    precision, recall, thresholds = precision_recall_curve(y_test, y_test_probas, pos_label=pos_label)
    
    # Calculate F1 scores for each threshold
    # Ignoring the last value of precision and recall as they correspond to the lowest threshold (i.e., "recall" is 1)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1])
    
    # Create a plot
    plt.figure(figsize=(figx, figy))
    plt.plot(thresholds, precision[:-1], label="Precision", linewidth=2)
    plt.plot(thresholds, recall[:-1], label="Recall", linewidth=2)
    plt.plot(thresholds, f1_scores, label="F1 Score", linewidth=2)

    plt.title("Precision, Recall, and F1 Score vs Thresholds", fontsize=common_fontsize)
    plt.xlabel("Probability Threshold", fontsize=common_fontsize)
    plt.ylabel("Score", fontsize=common_fontsize)
    plt.legend(loc="best", fontsize=common_fontsize)
    plt.xticks(fontsize=common_fontsize - 2)
    plt.yticks(fontsize=common_fontsize - 2)
    plt.grid(True)
    plt.show()
