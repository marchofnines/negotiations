import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.subplots as sp
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
from helpers.my_imports import * 
pio.renderers.default='notebook'


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

    ### Define df_viz to facilitiate plotting acceptance rates
    df_viz = df.copy()
    # Map 'decision' column to numeric values for rate calculation
    df_viz.decision = df_viz.decision.map({'Rejected': 0, 'Accepted':1})

    # Set up color palette and plotting style
    magma_palette = sns.color_palette('magma', 10)
    sns.set(style="whitegrid", font_scale=1)
    
    # Set figure size and common font size for labels and ticks
    plt.figure(figsize=(figx, figy))  
    common_fontsize =common_fontsize
    
    # Create the figure and the first subplot (categorical count distribution)
    plt.subplot(2,1,1)
    sns.countplot(data=df, x=feature,hue=hue, palette='magma')
    plt.title(f'{feature_label} Distribution by {legend_title}',fontsize=common_fontsize)
    plt.xlabel(f'',fontsize=common_fontsize)
    plt.xticks(ticks=plt.xticks()[0][::xtick_divisor], labels=plt.xticks()[1][::xtick_divisor], fontsize=common_fontsize, rotation=rotation)
    plt.ylabel('Count',fontsize=common_fontsize)
    plt.yticks(fontsize=common_fontsize)
    plt.legend(title=f'{legend_title}', title_fontsize=common_fontsize, fontsize=common_fontsize) 

    # Create the second subplot (bar plot for rates)
    plt.subplot(2, 1, 2)
    ax2=sns.barplot(data=df_viz, x=feature,y=hue, color=magma_palette[5]) ##color='#1f77b4')
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

def sns_boxplot(df,feature,feature_label,hue, xmin=None, xmax=None, figx=17, figy=6, common_fontsize=19.5):
    common_fontsize =common_fontsize
    plt.figure(figsize=(figx, figy))  
    sns.set(style="whitegrid", font_scale=1)
    palette = reverse_palette(df, hue, 'magma')
    sns.boxplot(data=df, x=feature, y=hue, palette=palette)
    plt.title(f'{feature_label} (Full Range)',fontsize=common_fontsize)
    plt.xlabel(f'{feature_label}',fontsize=common_fontsize)
    plt.ylabel('Count',fontsize=common_fontsize)
    plt.xticks(fontsize=common_fontsize)
    plt.yticks(fontsize=common_fontsize)
    plt.xlim(xmin,xmax)
    plt.show()
    
    palette = reverse_palette(df, hue, 'magma')
def sns_numerical(df,feature,feature_label,hue, legend_title, rate_descr, binwidth=None, xmin=None, xmax=None, ymin=None, ymax=None, figx=17, figy=8, common_fontsize=19.5):
    plt.figure(figsize=(figx, figy))  
    sns.set(style="whitegrid", font_scale=1)
    
    palette = reverse_palette(df, hue, 'magma')
    
    plt.subplot(1,2,1)
    ax1=sns.boxplot(data=df, x=feature, y=hue, palette=palette)
    plt.title(f'{feature_label} (Full Range)',fontsize=common_fontsize)
    plt.xlabel(f'{feature_label}',fontsize=common_fontsize)
    plt.ylabel('Count',fontsize=common_fontsize)
    plt.xticks(fontsize=common_fontsize)
    plt.yticks(fontsize=common_fontsize)
    ax1.xaxis.get_offset_text().set_size(common_fontsize)
    
    plt.subplot(1, 2, 2)
    ax2=sns.histplot(data=df, x=feature,hue=hue, binwidth = binwidth,label=legend_title, palette=palette)
    plt.title(f'{feature_label} ({xmin},{xmax})',fontsize=common_fontsize)
    plt.xlabel(f'{feature_label}',fontsize=common_fontsize)
    plt.ylabel('Count',fontsize=common_fontsize)
    plt.xticks(fontsize=common_fontsize)
    plt.yticks(fontsize=common_fontsize)
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)

    legend = ax2.get_legend()
    legend.set_title(legend_title, prop={'size': common_fontsize-1})
    for text in legend.get_texts():
        text.set_fontsize(common_fontsize-1)
    
    plt.tight_layout(h_pad=1.09)
    banner_y = 1.07
    plt.suptitle(f"Distribution of {feature_label} (by {legend_title})", 
                        fontsize=common_fontsize + 4, weight='bold', 
                        x=0.5, y=banner_y-0.02, ha='center')    
    plt.show()



def myhist(df, feature, feature_label, hue, legend_title, legend_labels, binwidth=None, xmin=None, xmax=None,figx=18, figy=8,common_fontsize=15):
    plt.figure(figsize=(figx,figy))
    sns.set(style="whitegrid", font_scale=1)
   
    # Determine the order of classes based on volume and assign colors accordingly
    class_order = df[hue].value_counts().index
    n_classes = df[hue].nunique()
    magma_palette = sns.color_palette('magma', n_classes)
    
    # If the majority class should have a lighter color, reverse the palette
    #if n_classes == 2:
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

def hist_orig_transf(df, feature, transformer, feature_label, hue, legend_title, legend_labels, binwidth_left=None, binwidth_right=None, xmin=None, xmax=None,figx=18, figy=8,common_fontsize=15):
    plt.figure(figsize=(figx,figy))
    sns.set(style="whitegrid", font_scale=1)
    dft=df.copy(deep=True)
    dft[f'transformed_{feature}']= transformer(dft[feature])
   
    # Determine the order of classes based on volume and assign colors accordingly
    class_order = df[hue].value_counts().index
    n_classes = df[hue].nunique()
    magma_palette = sns.color_palette('magma', n_classes)
    
    # If the majority class should have a lighter color, reverse the palette
    #if n_classes == 2:
    magma_palette.reverse()
    
    # Create the custom palette dictionary
    palette = {class_order[i]: magma_palette[i] for i in range(n_classes)}
    
    # First subplot for original data
    #palette = reverse_palette(df, hue, 'magma')
    plt.subplot(1,2,1)
    ax1=sns.histplot(data=dft, x=feature, hue=hue, binwidth=binwidth_left, legend=True, palette=palette)
    plt.title(f'Distribution of {feature_label} by {legend_title}', fontsize=common_fontsize)
    plt.xlabel(feature_label, fontsize=common_fontsize)
    plt.ylabel('Count', fontsize=common_fontsize)
    plt.xticks(fontsize=common_fontsize)
    plt.yticks(fontsize=common_fontsize)
    plt.xlim(xmin,xmax)  if xmin is not None and xmax is not None else None
    
    # Second subplot for transformed data
    #palette = reverse_palette(df, hue, 'magma')
    plt.subplot(1,2,2)
    ax2=sns.histplot(data=dft, x=f'transformed_{feature}', hue=hue, binwidth=binwidth_right, legend=True, palette=palette)
    plt.title(f'Distribution of Transformed {feature_label} by {legend_title}', fontsize=common_fontsize)
    plt.xlabel(feature_label, fontsize=common_fontsize)
    plt.ylabel('Count', fontsize=common_fontsize)
    plt.xticks(fontsize=common_fontsize)
    plt.yticks(fontsize=common_fontsize)
    #plt.xlim(xmin,xmax)  if xmin is not None and xmax is not None else None
    
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


def dec_boundary_mesh(estimator, X1, y, feature1, feature2):
    xx = np.linspace(X1.iloc[:, 0].min(), X1.iloc[:, 0].max(), 50)
    yy = np.linspace(X1.iloc[:, 1].min(), X1.iloc[:, 1].max(), 50)
    XX, YY = np.meshgrid(xx, yy)
    grid = np.c_[XX.ravel(), YY.ravel()]
    labels = pd.factorize(estimator.predict(grid))[0]
    plt.contourf(xx, yy, labels.reshape(XX.shape), cmap = 'twilight', alpha = 0.6)
    sns.scatterplot(data = X1, x = feature1, y = feature2, hue = y,  palette = 'flare')
    

def plot_percentage_barplots(df, subp_titles, legend_title, figure_title='', target='y', row_height=200):
    """
    Create percentage bar plots for each feature in a DataFrame against a target column.
    
    Parameters:
    - df: pandas DataFrame, the dataset containing the features and target
    - subp_titles: list of str, titles for the subplots
    - legend_title: str, title for the legend
    - figure_title: str, title for the entire figure (default is an empty string)
    - target: str, the target column name (default is 'y')
    - row_height: int, height of each subplot row (default is 200)
    
    Returns:
    - None: The function shows the plot.
    """
    # Extract feature names, removing the target column
    features = df.columns.tolist()
    features.remove(target)  
    # Determine the number of rows for subplots
    num_rows = len(features)
    colors = {'yes': px.colors.qualitative.Dark24[0], 'no': px.colors.qualitative.Dark24[1]}

    # Create subplots with specified number of rows
    fig = make_subplots(rows=num_rows, cols=1, subplot_titles=subp_titles)
    
    # Iterate through each feature to create subplots
    for idx, feature in enumerate(features):
        # Calculate the percentages
        percentages = pd.crosstab(df[feature], df[target], normalize='index') * 100
        
        # Create bar plot
        for i, outcome in enumerate(df[target].unique()):
            fig.add_trace(
                go.Bar(name=outcome, x=percentages.index, y=percentages[outcome], marker_color=colors[outcome], 
                      text = round(percentages[outcome],0)),
                row=idx + 1, col=1
            )
            if idx > 0:  # Hide legend for all but the first subplot
                fig.data[-1].showlegend = False

        # Update axis titles
        fig['layout'][f'xaxis{idx + 1}'].title.text = subp_titles[idx]
        fig['layout'][f'yaxis{idx + 1}'].title.text = 'Percentage (%)'
        fig['layout'][f'xaxis{idx + 1}'].tickfont.size = 14  
        fig['layout'][f'yaxis{idx + 1}'].tickfont.size = 14  
    
    # Update layout
    fig.update_layout(
        height=num_rows * row_height,
        width=1000,
        title=figure_title,
        legend_title=legend_title,
        legend=dict(x=1, y=1, xanchor='right', yanchor='bottom')
    )
    # Show plot
    fig.show('notebook')

def conf_matrix_ROC(model, X_test, y_test):
    y_test_preds = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_test_preds)
    #Plot the confusion matrix and the ROC AUC curve
    fix , ax = plt.subplots(1, 2, figsize=(20, 8))
    sns.set(style="whitegrid", font_scale=1)
    common_fontsize=20

    plt.subplot(1, 2, 1)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='RdPu', annot_kws={"size": 20},
                xticklabels=['Rejected', 'Accepted'], yticklabels=['Rejected', 'Accepted'])
    plt.title('Confusion Matrix for Chosen Model', fontsize=common_fontsize)
    plt.xlabel('Predicted',fontsize=common_fontsize)
    plt.ylabel('True',fontsize=common_fontsize)
    plt.xticks(fontsize=common_fontsize)
    plt.yticks(fontsize=common_fontsize)

    plt.subplot(1, 2, 2)
    RocCurveDisplay.from_estimator(model, X_test, y_test, pos_label='Accepted', linewidth=2.7, ax=ax[1])
    ax[1].plot(np.array([0, 1]), np.array([0, 1]), linewidth=2.7, label='baseline')
    plt.title('ROC Curve for Chosen Model', fontsize=common_fontsize)
    plt.xlabel('False Positive Rate', fontsize=common_fontsize)
    plt.ylabel('True Positive Rate',fontsize=common_fontsize)
    plt.xticks(fontsize=common_fontsize)
    plt.yticks(fontsize=common_fontsize)
    plt.legend(title='', title_fontsize=common_fontsize, fontsize=common_fontsize) 
    
def conf_matrix_PRC(model, X_test, y_test, pos_index, threshold=0.5, common_fontsize=20, figx=20, figy=8, xlegend=0.1, ylegend=0.1):
    pos_label = model.classes_[pos_index]
    neg_label = model.classes_[1 - pos_index]

    #y_test_preds = model.predict(X_test) used in confusion matrix
    y_test_preds = np.where(model.predict_proba(X_test)[:,pos_index] >= threshold, pos_label, neg_label) 

    y_test_probas = model.predict_proba(X_test)[:,pos_index] #used in PRC
    
    # Calculating precision, recall, thresholds and F1 score for each threshold
    precision, recall, thresholds = precision_recall_curve(y_test, y_test_probas, pos_label=pos_label)
    # Calculate F1 scores for each threshold
    # Ignoring the last value of precision and recall as they correspond to the lowest threshold (i.e., "recall" is 1)

    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1])

    # Finding the threshold for the optimal F1 score
    optimal_threshold_index = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_threshold_index]
    optimal_recall = recall[optimal_threshold_index]
    optimal_f1 = f1_scores[optimal_threshold_index]
    
    # F1 score using the default 0.5 threshold
    default_f1 = f1_score(y_test, model.predict(X_test), pos_label=pos_label, average='binary')
    default_f1w = f1_score(y_test, model.predict(X_test), pos_label=pos_label, average='weighted')
    default_recall_index = np.argmin(np.abs(thresholds - 0.5))  # closest recall index for threshold of 0.5
    default_recall = recall[default_recall_index]
    
    #Plot the confusion matrix and the ROC AUC curve
    conf_matrix = confusion_matrix(y_test, y_test_preds)
    fix , ax = plt.subplots(1, 2, figsize=(figx, figy))
    sns.set(style="whitegrid", font_scale=1)

    plt.subplot(1, 2, 1)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='RdPu', annot_kws={"size": 20},
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title('Confusion Matrix', fontsize=common_fontsize)
    plt.xlabel('Predicted',fontsize=common_fontsize)
    plt.ylabel('Actual',fontsize=common_fontsize)
    plt.xticks(fontsize=common_fontsize)
    plt.yticks(fontsize=common_fontsize)

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f'Precision-Recall Curve (AUC:{auc(recall, precision):.2f})', linewidth=2.7)
    #plt.plot(np.array([0, 1]), np.array([1, 0]), linewidth=2.7, label='baseline')
    # Calculate the baseline (chance level) for the PRC
    baseline = sum(y_test == model.classes_[pos_index]) / len(y_test)
    plt.axhline(y=baseline, color='gray', linestyle='--', linewidth=2, label=f'No Skill Model: ({baseline:.2f})') #Random Guessing

    plt.xlabel('Recall (TP/TP+FN)', fontsize=common_fontsize) #Sensitivity
    plt.ylabel('Precision (TP/TP+FP)', fontsize=common_fontsize)  #PPV
    plt.title('Precision-Recall Curve', fontsize=common_fontsize) 
    # Plotting a vertical line for the default threshold's recall
    plt.axvline(x=default_recall, color='blue', linestyle='--',
                label=f'Default at p=0.5 (F1={default_f1:.2f}, F1W={default_f1w:.2f})')
    plt.text(0.5, plt.ylim()[1], 'x=0.5', ha='center', va='bottom', color='red')


    # Plotting a vertical line for the optimal threshold's recall
    plt.axvline(x=optimal_recall, color='red', linestyle='--',
                label=f'Best Tradeoff for p={optimal_threshold:.2f} (F1:{optimal_f1:.2f})')

    plt.legend(bbox_to_anchor=(xlegend, ylegend), loc='lower left', fontsize=common_fontsize*0.77)
    plt.show()


    # Determine the order of classes based on volume and assign colors accordingly
    ##class_order = df[hue].value_counts().index
    #n_classes = df[hue].nunique()
    #magma_palette = sns.color_palette(base_palette, n_classes)
    #magma_palette.reverse()
    #return {class_order[i]: magma_palette[i] for i in range(n_classes)}