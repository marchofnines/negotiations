"""
Author: Basil Haddad
Date: 11.01.2023

Description:
    Contains Numerical Transformers to be used in Pipelines for Cross-Validation
"""

#import custom modules
from importlib import reload
from helpers.my_imports import *
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class LogTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer class to apply logarithmic transformation to features.

    This transformer applies a logarithmic transformation to numeric data.
    It can be useful for transforming skewed data into a more normal distribution.

    Parameters:
    -----------
    add_constant : float, default=1
        A constant value added to features before applying the logarithmic 
        transformation to avoid issues with zero or negative values.

    Attributes:
    -----------
    add_constant : float
        The constant value that will be added to each feature before the 
        logarithmic transformation.

    Methods:
    --------
    fit(X, y=None)
        Fit method for the transformer. In this case, it does nothing and 
        returns the transformer object itself, as there's no fitting process 
        necessary for a logarithmic transformation.

    transform(X, y=None)
        Applies the logarithmic transformation to the input features.
    """

    def __init__(self, add_constant=1):
        """
        Initialize the LogTransformer with an add_constant value.

        Parameters:
        -----------
        add_constant : float, default=1
            A constant value to add to each feature to avoid log(0) and 
            negative values.
        """
        self.add_constant = add_constant
   
    def fit(self, X, y=None):
        """
        Fit the transformer on the data.

        Since the logarithmic transformation does not require fitting to specific 
        data, this method simply returns the transformer object itself.

        Parameters:
        -----------
        X : DataFrame
            The input data to transform.
        y : array-like, optional (default=None)
            Target values (not used in this transformer).

        Returns:
        --------
        self : object
            Returns the instance itself.
        """
        return self

    def transform(self, X, y=None):
        """
        Apply the logarithmic transformation to the input data.

        The transformation is applied as log(X + add_constant) to each element of X.

        Parameters:
        -----------
        X : DataFrame
            The input data to transform.
        y : array-like, optional (default=None)
            Target values (not used in this transformer).

        Returns:
        --------
        X_transformed : array-like
            The transformed data.
        """
        return np.log(X + self.add_constant)
    
class RareCategoryCombiner(BaseEstimator, TransformerMixin):
    """
    A custom transformer class to combine rare categories in categorical features.

    This transformer identifies and replaces rare categories in categorical 
    features with a common label based on their proportion in the dataset. 
    It's useful in preprocessing steps for machine learning models to handle 
    categories with very few occurrences.

    Parameters:
    -----------
    rare_to_value : float, default=0.01
        The proportion threshold below which a category is considered rare. 
        Categories with a frequency less than or equal to rare_to_value * len(X) 
        in each feature will be replaced.

    rare_label : str, default='Unknown'
        The label to replace rare categories with.

    Attributes:
    -----------
    rare_categories : dict
        A dictionary where keys are feature names and values are lists of 
        rare categories in each feature.
    """

    def __init__(self, rare_to_value=0.01, rare_label='Unknown'):
        """
        Initialize the RareCategoryCombiner with a proportion threshold and a label for rare categories.

        Parameters:
        -----------
        rare_to_value : float, default=0.01
            The proportion threshold for determining if a category is rare.
        
        rare_label : str, default='Unknown'
            The label to assign to rare categories.
        """
        self.rare_to_value = rare_to_value
        self.rare_label = rare_label
    
    def fit(self, X, y=None):
        """
        Fit the transformer by identifying rare categories in each feature.

        This method identifies the categories in each categorical feature 
        that occur less frequently than rare_to_value * len(X) and stores them.

        Parameters:
        -----------
        X : DataFrame
            The input data containing categorical features.

        Returns:
        --------
        self : object
            Returns the instance itself.
        """
        # Calculate the frequency threshold based on rare_to_value
        threshold = self.rare_to_value * len(X)

        # Identify and store rare categories in each feature
        self.rare_categories = X.apply(lambda col: col.value_counts()).apply(
            lambda col: col[col <= threshold].index.tolist()
        )
        return self

    def transform(self, X, y=None):
        """
        Transform the input data by replacing rare categories with the specified label.

        Parameters:
        -----------
        X : DataFrame
            The input data to transform.

        Returns:
        --------
        X_transformed : DataFrame
            The transformed data with rare categories replaced.
        """
        # Replace rare categories in each feature with the specified label
        X = X.apply(lambda col: col.replace({x: self.rare_label for x in self.rare_categories[col.name]}))
        return X


class RareCategoryCombiner2(BaseEstimator, TransformerMixin):
    """
    A custom transformer class to combine rare categories in categorical features.
    This is a variation of RareCategoryCombiner but takes a percentage instead of a number of categories.

    This transformer identifies and replaces rare categories in categorical 
    features with a common label. It can be particularly useful in preprocessing 
    steps for machine learning models to handle categories with very few occurrences.

    Parameters:
    -----------
    threshold : int, default=10
        The frequency threshold below which a category is considered rare. 
        Categories with a count less than or equal to this threshold in 
        each feature will be replaced.

    rare_label : str, default='Unknown'
        The label to replace rare categories with.

    Attributes:
    -----------
    rare_categories : dict
        A dictionary where keys are feature names and values are lists of 
        rare categories in each feature.
    """

    def __init__(self, threshold=10, rare_label='Unknown'):
        """
        Initialize the RareCategoryCombiner with a threshold and a label for rare categories.

        Parameters:
        -----------
        threshold : int, default=10
            The threshold for determining if a category is rare.
        
        rare_label : str, default='Unknown'
            The label to assign to rare categories.
        """
        self.threshold = threshold
        self.rare_label = rare_label
    
    def fit(self, X, y=None):
        """
        Fit the transformer by identifying rare categories in each feature.

        This method identifies the categories in each categorical feature 
        that occur less than or equal to the threshold and stores them.

        Parameters:
        -----------
        X : DataFrame
            The input data containing categorical features.
        y : array-like, optional (default=None)
            Target values (not used in this transformer).

        Returns:
        --------
        self : object
            Returns the instance itself.
        """
        # Identify and store rare categories in each feature
        self.rare_categories = X.apply(lambda col: col.value_counts()).apply(
            lambda col: col[col <= self.threshold].index.tolist()
        )
        return self

    def transform(self, X, y=None):
        """
        Transform the input data by replacing rare categories with the specified label.

        Parameters:
        -----------
        X : DataFrame
            The input data to transform.
        y : array-like, optional (default=None)
            Target values (not used in this transformer).

        Returns:
        --------
        X_transformed : DataFrame
            The transformed data with rare categories replaced.
        """
        # Replace rare categories in each feature with the specified label
        X = X.apply(lambda col: col.replace({x: self.rare_label for x in self.rare_categories[col.name]}))
        return X
