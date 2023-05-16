import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import sklearn.model_selection
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, log_loss, classification_report,precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler
import warnings 
warnings.filterwarnings("ignore")

df=pd.read_csv("Social_Network_Ads.csv")
# print (df.head(5))
# print(df.isnull())
# print(df.info())
# print(df.describe()())
# print(df.corr())
# sns.heatmap(df.corr(), annot=True)
fig, ax=plt.subplots(figsize=(16,8))
# sns.heatmap(df.corr(), annot=True)
#plt.show()
sns.set_style('whitegrid')
sns.countplot(x='User ID',data=df,hue='Gender')
# plt.show()
sns.countplot(x='User ID', hue='Purchased',data=df )
# plt.show()
sns.displot(df['Age'].dropna(),kde=False,bins=30)
# plt.show()
df['EstimatedSalary'].plot(kind='hist',bins=50)
# plt.show()

#train test split
X=df[['Age','EstimatedSalary']]
Y=df['Purchased']

X_train, X_test, Y_train, Y_test=train_test_split(X, Y , test_size=0.2, random_state=50)

#normalization
nrm=MinMaxScaler().fit(X_train)
X_train=nrm.transform(X_train)

nrm=MinMaxScaler().fit(X_test)
X_test=nrm.transform(X_test)


model=LogisticRegression().fit(X_train,Y_train)

# print("Model score ", model.score(X_test,Y_test))
#Model score  0.825
Y_pred=model.predict(X_test)
# print(Y_pred)

# cf_matrix=confusion_matrix(Y_test,Y_pred)
# print("The confusion matrix is:\n",cf_matrix)
"""
The confusion matrix is:
 [[52  2]
 [12 14]]
"""
# score1=precision_recall_fscore_support(Y_test,Y_pred, average="micro")
# print("Precision: ",score1[0])
# print("Recall: ",score1[1])
# print("F1-score: ",score1[2])

# cl_rep=classification_report(Y_test,Y_pred)
# print(cl_rep)
"""              precision    recall  f1-score   support

           0       0.81      0.96      0.88        54
           1       0.88      0.54      0.67        26

    accuracy                           0.82        80
   macro avg       0.84      0.75      0.77        80
weighted avg       0.83      0.82      0.81        80

"""
# log_loss1=log_loss(Y_test,Y_pred)
# print("Log loss is",log_loss1)
#Log loss  6.044305859045124


#some plots
# print(df.info())

sns.boxplot(x='EstimatedSalary' ,y='Age',data=df)
# plt.show()
# df.dropna(inplace=True)
# Gender=pd.get_dummies(df['Gender']drop_fist=True)
# df=pd.concat([df.Gender],axis=1)
# print(df.head())

# #for logistic regression
# df.drop('User ID',inplace=True,axis=1)
