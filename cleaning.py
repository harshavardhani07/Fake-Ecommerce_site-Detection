
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("raw_data.csv")
data.drop(['identifierHash','type','country','language','hasAnyApp','civilityTitle','civilityGenderId','seniorityAsMonths','seniorityAsYears','countryCode','productsWished','productsBought','hasAndroidApp','hasIosApp'],axis=1,inplace=True)

#print(data['gender'].unique())
converted = pd.get_dummies(data['gender'], drop_first=1)
data = pd.concat([data, converted], axis = 1)
data.drop('gender',axis = 1, inplace=True)
data.rename(columns={'M': 'Male'}, inplace=True)
#print(data['Male'].unique())
data = data.fillna(0)
data = data.astype(int)


def zscore(array):
    thr = 3
    mean = np.mean(array)
    std = np.std(array)
    z_scores = (array - mean) / std
    return np.abs(z_scores) > thr


combined_condition = ~(zscore(data['socialNbFollows']) | zscore(data['socialNbFollowers']) | zscore(data['productsListed']) | zscore(data['productsSold']) | zscore(data['socialProductsLiked']))
data = data[combined_condition]
data.reset_index(drop=True, inplace=True)
#print(new_data.head())
#print(data.describe())


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

# Encodedict = {}
# for i in ['Male','hasProfilePicture']:
#     key = '_{}'.format(i)
#     le = LabelEncoder()
#     data[key] = le.fit_transform(list(data[i]))
#     Encodedict[key] = le.classes_

# data.drop(['Male','hasProfilePicture'], axis=1, inplace=True)

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
print(data_new.describe())


from sklearn.model_selection import train_test_split

# Assuming 'Fraud' is your target variable
X = data_new.drop('Fraud', axis=1)  # Features
y = data_new['Fraud']  # Target variable

#plt.figure(figsize=(4, 4))
#sns.countplot(x='productsListed', hue='Fraud', data=data_new, palette = 'inferno')
#sns.boxplot(x='productsListed', y='socialNbFollowers', hue='Fraud', data=data, palette = 'inferno')
#sns.histplot(x = 'seniority', data = data_new, binwidth = 350, hue = 'Fraud', element = 'step')
# Splitting the data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optionally, you can reset index for train and test sets
X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

# pd.plotting.scatter_matrix (data, figsize = [30,30])
# plt.show()
# data.hist(bins=50, figsize=(20,15))
# plt.show()
