import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import sklearn.model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, log_loss, classification_report,precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import warnings 
warnings.filterwarnings("ignore")

df = pd.read_csv("Iris.csv")
#print(df.head())
#print(df.info())
#print(df.describe())
#print(df.isnull().sum())       #no null values
# print(df.dtypes)
df=df.astype({"SepalLengthCm":"object"})	
# print(df.dtypes)		#changing dtype to object to perform outlier removal

#detecting outliers:
def remove_outliers(df,var):
	final_df=[]
	q1=df[var].quantile(0.25)
	q3=df[var].quantile(0.75)
	iqr=q3-q1
	high=q3+(1.5*iqr)
	low=q1-(1.5*iqr)
	for j in df[var]:
		if(j<low or j>high):
			final_df.append(j)
			df.drop(df.loc[df[var]==j].index,inplace=True)
remove_outliers(df,'SepalLengthCm')
remove_outliers(df,'SepalWidthCm')
remove_outliers(df,'PetalLengthCm')
remove_outliers(df,'PetalWidthCm')
df.drop(['Id'],axis=1,inplace=True)
#print(df.head())

#sns.heatmap(df.corr(),annot=True)
#plt.show()
X=df.iloc[:,[0,1,2,3]].values
y=df.iloc[:,4].values

#print(X)

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.fit_transform(x_test)

from sklearn.naive_bayes import BernoulliNB
model=BernoulliNB()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

#print(accuracy_score(y_pred,y_test)) 	#0.8

cf=confusion_matrix(y_test,y_pred)
# print(cf)
"""
[[11  0  0]
 [ 2  4  4]
 [ 0  0  9]]
"""
# print("classification_report :\n",classification_report(y_test,y_pred))
"""
classification_report :
                  precision    recall  f1-score   support

    Iris-setosa       0.85      1.00      0.92        11
Iris-versicolor       1.00      0.40      0.57        10
 Iris-virginica       0.69      1.00      0.82         9

       accuracy                           0.80        30
      macro avg       0.85      0.80      0.77        30
   weighted avg       0.85      0.80      0.77        30

"""