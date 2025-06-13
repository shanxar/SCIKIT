import pandas as pd 

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.naive_bayes import CategoricalNB
 
from sklearn.metrics import accuracy_score

import numpy as np
 
#Step1:Load dataset

df=pd.read_csv("adult.csv")
#Cleaning data
missing_vales=["?"]
df.replace(missing_vales,np.nan,inplace=True)

df.dropna(inplace=True)
df.index=list(range(1,len(df)+1))
print("STEP 1:",df.columns)
print("TYPE:",df["income"].dtype)

#step2: split X,Y
X=df.drop("income",axis=1)
Y=df["income"]

#Step 3: encoding
encoded_columns=[]
for x in X.columns:
    encoded_name=str(x)+"_enc"
    encoded_columns.append(encoded_name)
    le=LabelEncoder()
    X[encoded_name]=le.fit_transform(df[x])

le_Y=LabelEncoder()
Y_encoded=le_Y.fit_transform(Y)

#step 4: split train test split
X_train,X_test,Y_train,Y_test=train_test_split(X[encoded_columns],Y_encoded,test_size=0.3,random_state=20)
print(X_train.shape)
print(X_test.shape)


#step6: training
model=CategoricalNB()
model.fit(X_train,Y_train)

#STep7: predict

y_pred=model.predict(X_test)
y_pred_category=le_Y.inverse_transform(y_pred)

#step8:
accuracy=accuracy_score(y_pred,Y_test)
print(accuracy)
print(*(y_pred_category))