import numpy as np
import pandas
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import model_selection
from sklearn import metrics

"""

Simple linear regression using scikit learn.

Data from here - (https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

"""

def basic(file, OneHotEncoder = None, StandardScaler = None):

    file_df = pandas.read_csv(file)

    # drop the Id
    dataset_df = file_df.drop("Id", axis= 1)
    y = None
    if "SalePrice" in dataset_df:
        y = dataset_df["SalePrice"]
        dataset_df = dataset_df.drop("SalePrice", axis=1)

    dataset_df = dataset_df.replace({"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 
                                        50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75", 
                                        80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 
                                        150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"},
                                    "MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun",
                                                7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"}
                                    })

    # seperate into numerical and categorical features
    numerical_features = dataset_df.select_dtypes(exclude="object")
    categorical_features = dataset_df.select_dtypes(include="object")

    numerical_features = numerical_features.fillna(numerical_features.median())
    if StandardScaler == None:
        StandardScaler = preprocessing.StandardScaler()
        numerical_features = StandardScaler.fit(numerical_features).transform(numerical_features)
    else:
        numerical_features = StandardScaler.transform(numerical_features)    
    numerical_features = pandas.DataFrame(numerical_features)

    categorical_features = categorical_features.fillna("DummyValue")

    if OneHotEncoder == None:
        OneHotEncoder = preprocessing.OneHotEncoder(sparse_output=False, handle_unknown='ignore')       
        categorical_features = OneHotEncoder.fit(categorical_features).transform(categorical_features)
    else: 
        categorical_features = OneHotEncoder.transform(categorical_features)
       
    categorical_features = pandas.DataFrame(categorical_features)
    all_features = pandas.concat([numerical_features,categorical_features] , axis=1)

    return (all_features, y, OneHotEncoder, StandardScaler)


def train():

    all_features, y, OneHotEncoder , StandardScaler = basic('train.csv')

    LinearRegression = linear_model.LinearRegression()
    LinearRegression.fit(all_features, y)

    return (OneHotEncoder, StandardScaler, LinearRegression)

def test(OneHotEncoder, StandardScaler , LinearRegression):

    all_features, _ , _, _= basic('test.csv', OneHotEncoder, StandardScaler)


    predictions = LinearRegression.predict(all_features)
    return predictions

OneHotEncoder, StandardScaler, LinearRegression = train()
predictions = test(OneHotEncoder, StandardScaler, LinearRegression)

output_df = pandas.DataFrame(predictions, columns=["SalePrice"])
output_df["Id"] = output_df.index + 1460 + 1
output_df = output_df[["Id", "SalePrice"]]
output_df.to_csv('sub.csv', index=False)




