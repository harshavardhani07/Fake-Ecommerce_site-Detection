import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

data = pd.read_csv("raw_data.csv")
#remove duplicate columns
if 'identifierHash' in data.columns:
    data.drop('identifierHash', axis=1, inplace=True)
if 'countryCode' in data.columns:
    data.drop('countryCode', axis=1, inplace=True)
if 'civilityTitle' in data.columns:
    data.drop('civilityTitle', axis=1, inplace=True)

# Identify numeric columns (int and float types)
numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns

# Initialize encoders and perform hot encoding on Gender column
label_encoder = LabelEncoder()
one_hot_encoder = OneHotEncoder(sparse_output=False)
converted = pd.get_dummies(data['gender'], drop_first=1)
data = pd.concat([data, converted], axis = 1)
data.drop('gender',axis = 1, inplace=True)
data.rename(columns={'M': 'Male'}, inplace=True)
data = data.fillna(0)

# Handle non-numeric columns separately (e.g., keep them as they are, or convert them to category type)
boolean_columns = ['hasAnyApp', 'hasAndroidApp', 'hasIosApp', 'hasProfilePicture', 'Male']
object_columns = data.select_dtypes(include='object').columns
for col in object_columns:
    data[col] = data[col].astype('category')
# Convert boolean columns to integers
for col in boolean_columns:
    data[col] = data[col].astype(int)

# Convert only numeric columns to integer types
data[numeric_columns] = data[numeric_columns].astype(int)

categorical_columns = ['type', 'country', 'language']
# Convert categorical columns using one-hot encoding
encoded_data = one_hot_encoder.fit_transform(data[categorical_columns])

# Create a DataFrame with one-hot encoded columns
encoded_df = pd.DataFrame(encoded_data, columns=one_hot_encoder.get_feature_names_out(categorical_columns))

# Combine the one-hot encoded columns with the original data, excluding the original categorical columns
data = pd.concat([data, encoded_df], axis=1).drop(categorical_columns, axis=1)

def zscore(array):
    thr = 3
    mean = np.mean(array)
    std = np.std(array)
    z_scores = (array - mean) / std
    return np.abs(z_scores) > thr


combined_condition = ~(zscore(data['socialNbFollows']) | zscore(data['socialNbFollowers']) | zscore(data['productsListed']) | zscore(data['productsSold']) | zscore(data['socialProductsLiked']))
data = data[combined_condition]
data.reset_index(drop=True, inplace=True)

def pure_round(num):
    integer = int(num)
    fraction = num - float(integer)
    if fraction >= 0.5:
        integer += 1
    return integer

data = data[data['productsListed'] != 0]
for i in data.index:
    case_no = data.loc[i,'productsSold']
    pass_no = pure_round((case_no * data.loc[i,'productsPassRate']) / 100)
    fail_no = case_no - pass_no
    data.loc[i,'productsPassed'] = pass_no
    data.loc[i,'productsFailed'] = fail_no
    if case_no == 0:
        data.drop(i, axis=0, inplace=True)
 
data.drop(['productsPassRate','productsSold'], axis=1, inplace=True)
dfdict = {}
for j in data.index:
    x = data.loc[j,'productsPassed']
    y = data.loc[j,'productsFailed']
    if x != 0:
        data.loc[j,'Fraud'] = 0
        df = pd.DataFrame(data.loc[j,:]).transpose()
        ldf = pd.concat([df]*int(x), ignore_index=True)
    
    if y != 0:
        data.loc[j,'Fraud'] = 1
        df2 = pd.DataFrame(data.loc[j,:]).transpose()
        ldf2 = pd.concat([df2]*int(y), ignore_index=True)
    
    if x != 0 and y != 0:
        dfdict[j] = pd.concat([ldf, ldf2], ignore_index=True)
    elif x != 0:
        dfdict[j] = ldf
    else:
        dfdict[j] = ldf2

data_new = pd.concat(dfdict.values(), ignore_index=True)
data_new.drop(['productsPassed','productsFailed'], axis=1, inplace=True)
#print(data_new.describe())
#print(data_new.head(20))
#print(data_new.info())


from sklearn.model_selection import train_test_split

# Assuming 'Fraud' is your target variable
X = data_new.drop('Fraud', axis=1)  # Features
y = data_new['Fraud']  # Target variable
data_new['Fraud'] = pd.to_numeric(data_new['Fraud'], errors='coerce')
data_new['Fraud'] = data_new['Fraud'].astype('int64')

#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = pd.to_numeric(y_train, errors='coerce')

# Handle any missing values (NaN) as needed
# For example, you can drop rows with missing values or fill them with a default value:
y_train = y_train.dropna()  # Drop rows with NaN values

# Convert the series to integer type
y_train = y_train.astype(int)


X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_test = X_test.apply(pd.to_numeric, errors='coerce')
y_test = y_test.astype(int)
X_test.fillna(X_test.mean(), inplace=True)