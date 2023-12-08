"""
Author: Basil Haddad
Date: 11.01.2023

Description:
    Import Libraries used for capstone project
"""

#import sys
#sys.path.append('/Users/basilhaddad/jupyter/capstone/')

import time
import math 
import copy
import pprint

import pandas as pd
import numpy as np
from joblib import dump, load
from IPython.display import display
from IPython.core.display import HTML
from pandas.io.formats.style import Styler
from pandas.io.formats import style

import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt
import plotly.express as px
import plotly.subplots as sp
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='notebook'

from scipy.stats import entropy, expon, randint, uniform, loguniform
from scipy.linalg import svd

from sklearn.experimental import enable_halving_search_cv, enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from category_encoders.cat_boost import CatBoostEncoder
from category_encoders import JamesSteinEncoder, BinaryEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, PowerTransformer,  QuantileTransformer
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, OrdinalEncoder, TargetEncoder,  LabelBinarizer
from sklearn.compose import make_column_transformer, make_column_selector, ColumnTransformer, TransformedTargetRegressor
from sklearn.feature_selection import SequentialFeatureSelector, SelectFromModel, RFE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN

from sklearn.linear_model import LinearRegression, Lasso, HuberRegressor, Ridge, RidgeClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.multiclass import OneVsOneClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_text, plot_tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score , cross_validate, train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV

from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, recall_score, precision_score, make_scorer
from sklearn.metrics import precision_recall_curve, auc, roc_curve, RocCurveDisplay, log_loss, roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.inspection import PartialDependenceDisplay, partial_dependence

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE 
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

import tensorflow as tf
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout

from sklearn.base import clone
from sklearn import set_config
import warnings
from sklearn.exceptions import ConvergenceWarning

#General Settings
set_config(display="diagram")
#set_config("figure")
pd.set_option('display.max_columns', None)

#warnings.filterwarnings('ignore')
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)