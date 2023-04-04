# imports
import pandas as pd
import numpy as np
import pickle

# load the model
LassoRegressionModel = pickle.load(
    open("./static/model/LassoLinearModel.pkl", "rb"))

# load encoder and scaler
OneHotEncoder = pickle.load(open("./static/model/DataEncoder.pkl", "rb"))
Scaler = pickle.load(open("./static/model/DataScaler.pkl", "rb"))

# features with low correlations for feature engineering
low_corr = ['BsmtFullBath',
            'BsmtUnfSF',
            'BedroomAbvGr',
            'ScreenPorch',
            'PoolArea',
            'MoSold',
            '3SsnPorch',
            'BsmtFinSF2',
            'BsmtHalfBath',
            'MiscVal',
            'LowQualFinSF',
            'YrSold',
            'EnclosedPorch',
            'KitchenAbvGr']

# features with dependant correlation with the target variable
dependats = ['MSZoning',
             'Street',
             'LotShape',
             'LotConfig',
             'Neighborhood',
             'OverallQual',
             'OverallCond',
             'ExterQual',
             'ExterCond',
             'Foundation',
             'BsmtQual',
             'BsmtCond',
             'BsmtExposure',
             'Heating',
             'CentralAir',
             'KitchenQual',
             'GarageFinish',
             'GarageQual',
             'SaleType',
             'SaleCondition']

# ordinal features
ordinal_features = ['LotShape',  'OverallQual', 'OverallCond', 'ExterQual', 'ExterCond',
                    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'KitchenQual', 'GarageFinish', 'GarageQual']

nominal_features = ['MSZoning',
                    'Street',
                    'LotConfig',
                    'Neighborhood',
                    'Foundation',
                    'Heating',
                    'CentralAir',
                    'SaleType',
                    'SaleCondition']

# sperate ordinal values based on the values and the mapping
ordinal_features_0 = ['OverallQual', 'OverallCond']

ordinal_features_1 = ['ExterQual', 'ExterCond',
                      'BsmtQual', 'BsmtCond', 'KitchenQual', 'GarageQual']

ordinal_features_2 = ['LotShape']

ordinal_features_3 = ['BsmtExposure']

ordinal_features_4 = ['GarageFinish']


# define label encoding functions
def map1(x):
    return x.map({np.nan: 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5})


def map2(x):
    return x.map({np.nan: 0, "IR3": 1, "IR2": 2, "IR1": 3, "Reg": 4})


def map3(x):
    return x.map({np.nan: 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4})


def map4(x):
    return x.map({np.nan: 0, "Unf": 1, "RFn": 2, "Fin": 3})


"""
Main PreProcessing Function
"""
def preprocess(DataJson):
    """
    Apply data preprocessing step to the input data
    """

    # preprocess
    df = pd.read_json(DataJson, lines=True)

    print(df)

    # ["OverallQual", "OverallCond"] represent ordinal categories
    # convert them to object dtype for preprocessing
    df["OverallQual"] = df["OverallQual"].astype("object")
    df["OverallCond"] = df["OverallCond"].astype("object")

    #create data frame for numeric features and df for categorical feature
    test_num = df.select_dtypes(np.number)
    test_cat = df.select_dtypes("object")

    #drop columns
    test_num.drop(["Id", "MSSubClass", "GarageYrBlt"], axis=1, inplace=True)
    test_num.drop(low_corr, axis=1, inplace=True)

    # replace nulls
    test_num.fillna(test_num.median(), inplace=True)

    # get only dependant categorical feature
    test_cat = test_cat[dependats]

    # label encode ordinal feature
    test_cat[ordinal_features_0] = test_cat[ordinal_features_0].astype(int)
    test_cat[ordinal_features_1] = test_cat[ordinal_features_1].apply(map1)
    test_cat[ordinal_features_2] = test_cat[ordinal_features_2].apply(map2)
    test_cat[ordinal_features_3] = test_cat[ordinal_features_3].apply(map3)
    test_cat[ordinal_features_4] = test_cat[ordinal_features_4].apply(map4)

    # null values imputions for nominal features
    test_cat.dropna(inplace=True)


    # one-hot encode nominal feature
    test_cat = OneHotEncoder.transform(
        test_cat[nominal_features]).toarray()
    
    # concat cat_nom, cat_ord and num dfs
    frames = [test_num, test_cat]

    test = pd.concat(frames, axis=1)

    # scale data
    test_preprocessed = Scaler.transform(test)

    # convert to numpy array
    test_preprocessed.fillna(0, inplace=True)

    test_preprocessed = test_preprocessed.to_numpy()

    return test_preprocessed
