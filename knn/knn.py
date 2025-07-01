import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

#Step1: Load data and create a dataframe with it
iris=load_iris()
iris_df=pd.DataFrame(iris.data,columns=iris.feature_names)
iris_df['species']=iris.target

#Step2: split X,Y

X=iris_df.drop('species',axis=1)
Y=iris_df['species']

#Step3: train test split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=10)

#Step4: Scale the data
#Dataset shoudlnt be scaled then split, because when scaling training and test data would get peak of each other
scaler = StandardScaler()
X_train_scaled=scaler.fit_transform(X_test)
X_test=scaler.transform(X_test) 
 # fit() will calculate the  scaling, transform() will fit  the data , fit_Tranasform() does same
 #in traing  data we use fit_transform(). in testing we use transform() , because we use fit() of training data in the test data also


#Step5:Train the model
model=KNeighborsClassifier(n_neighbors=5) #n_neighbors takes into account of top n no of closest neighbors
model.fit(X_train_scaled,Y_test)

#Step6: Prediction
y_pred=model.predict(X_test)
print(y_pred)
print(Y_test)

#Step 7: performance evaluation
accuracy=accuracy_score(Y_test,y_pred)
conf_matrix=confusion_matrix(Y_test,y_pred)

print(f"Acuuracy score : {accuracy}\nConfusion matrix : {confusion_matrix}")
