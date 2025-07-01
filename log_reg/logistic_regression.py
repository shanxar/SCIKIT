from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


#step 1 :load data and create DataFrame out of it
dataset=load_breast_cancer()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['target']=dataset.target

#step2: split features and targets
X=df.drop('target',axis=1)
Y=df['target']

#Step 3 : split train test data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.4,random_state=42)

#step 4 :Scale the features
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

#Step 5: training the model
model=LogisticRegression(max_iter=400)
model.fit(X_train_scaled,Y_train)

#Step6 : Prediction

y_pred=model.predict(X_test_scaled)

#STep7: Display result

print(y_pred)
print(Y_test)

#Step 8: Evaluation

accuracy=accuracy_score(Y_test,y_pred)
print(accuracy)
conf_matrix=confusion_matrix(Y_test,y_pred)
print(conf_matrix)
