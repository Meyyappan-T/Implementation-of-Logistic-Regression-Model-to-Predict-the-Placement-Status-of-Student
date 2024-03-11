# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Collect student data including grades and placement status.
2. Clean and split the data for training and testing.
3. Train the logistic regression model using the training data.
4. Evaluate the model's performance using testing data.
5. Use the trained model to predict placement status for new students.


## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Meyyappan.T
RegisterNumber:  212223240086
import pandas as pd
data=pd.read_csv("C:/Users/marco/OneDrive/Documents/study materials/Placement_Data.csv")
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data.head()
data1.isnull()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
y=data1["status"]
y
x=data1.iloc[:,:-1]
x
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
*/
```

## Output:
![image](https://github.com/marcoyoi/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128804366/a0d925f4-78cd-482c-856a-f0a83d17fdec)
![image](https://github.com/marcoyoi/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128804366/34f8bca1-83e7-4f18-ac8a-f0d6658e1586)
![image](https://github.com/marcoyoi/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128804366/a3f58cc6-0c8f-46fb-9ca6-90bc06459fd4)
![image](https://github.com/marcoyoi/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128804366/482ed051-3493-495b-a454-1279662a19e9)
![image](https://github.com/marcoyoi/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128804366/bad9f500-5fac-4a75-b33d-1b706d0e1772)
![image](https://github.com/marcoyoi/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128804366/1df60bbe-c76e-4ea2-bf3f-d7a87a10d678)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
