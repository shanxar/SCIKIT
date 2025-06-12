import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

#Step1: load dataset
df=pd.read_csv("decision_tree_classifier_dataset.csv")



#Step2: Since we have text categrocial data we convert them to numberical by encoding use Label ENcoding
le_region=LabelEncoder()
le_product=LabelEncoder()
le_usage=LabelEncoder()
le_device=LabelEncoder()
le_label=LabelEncoder() #encoder for Customer Typpe

df['region_enc']=le_region.fit_transform(df['Region'])
df['product_enc']=le_product.fit_transform(df['Product Type'])
df['usage_enc']=le_usage.fit_transform(df['Usage'])
df['device_enc']=le_device.fit_transform(df['Device'])
df['label_enc']=le_label.fit_transform(df['Customer Type'])

#Step3: split X,Y
X=df[['region_enc','product_enc','usage_enc','device_enc']]
Y=df['label_enc']

#Step4 : split train and test data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=11)

#Step5: Scaling
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

#Step6: train model

model=DecisionTreeClassifier(criterion="gini",max_depth=4)
model.fit(X_train_scaled,Y_train)

#step 7: predict

y_pred=model.predict(X_test_scaled)

print("Predcited Customer types\n:",le_label.inverse_transform(y_pred))
print("\n\nACtual Customer types\n:",le_label.inverse_transform(Y_test))
#step8 : evaluation

accuracy=accuracy_score(Y_test,y_pred)
conf_matrix=confusion_matrix(Y_test,y_pred)

print(f"Acuuracy score : {accuracy}\nConfusion matrix : {confusion_matrix}")