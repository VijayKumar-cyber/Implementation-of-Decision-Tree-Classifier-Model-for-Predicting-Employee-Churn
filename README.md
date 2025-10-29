# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and preprocess the employee dataset (handle missing values, encode categorical variables, split features and target).

2.Split the data into training and testing sets.

3.Train a Decision Tree Classifier on the training data.

4.Predict and evaluate the model on the test data using accuracy or confusion matrix.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: VIJAY KUMAR D
RegisterNumber:25000878  
*/
```

```
import pandas as pd
data=pd.read_csv("Employee.csv")
print("data.head():")
data.head()
```
```
print("data.info():")
data.info()
```
```
print("isnull() and sum():")
data.isnull().sum()
```
```
print("data value counts():")
data["left"].value_counts()
```
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
```
```
print("data.head() for Salary:")
data["salary"]=le.fit_transform(data["salary"])
data.head()
```
```
print("data.head() for Salary:")
data["salary"]=le.fit_transform(data["salary"])
data.head()
```
```
print("x.head():")
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
```
```
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
```
```
print("Accuracy value:")
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
```
print("Data Prediction:")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
```
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plot_tree(dt, feature_names=x.columns, class_names=['salary', 'left'], filled=True)
plt.show()
```



## Output:
<img width="1739" height="103" alt="Screenshot 2025-10-29 103824" src="https://github.com/user-attachments/assets/3a60bc1a-50e9-4b4f-8e2f-dbc662729001" />
<img width="227" height="66" alt="Screenshot 2025-10-29 103805" src="https://github.com/user-attachments/assets/316563d5-6651-4ccc-8437-c9b4c82b8671" />
<img width="839" height="622" alt="Screenshot 2025-10-29 103738" src="https://github.com/user-attachments/assets/5502157a-6f45-482c-b753-7769d2db07b5" />
<img width="1492" height="268" alt="Screenshot 2025-10-29 103720" src="https://github.com/user-attachments/assets/0b064962-db77-491e-9186-6a4b03c89e70" />
<img width="1674" height="282" alt="Screenshot 2025-10-29 103704" src="https://github.com/user-attachments/assets/f9db7672-f056-44e8-ac52-ff9899342648" />
<img width="1721" height="278" alt="Screenshot 2025-10-29 103646" src="https://github.com/user-attachments/assets/a509d9ed-e2c3-410d-a1f2-397d8fb1cfb6" />
<img width="356" height="315" alt="Screenshot 2025-10-29 103625" src="https://github.com/user-attachments/assets/1a441d01-6d09-4245-8199-d6b76fe6b4df" />
<img width="437" height="601" alt="Screenshot 2025-10-29 103614" src="https://github.com/user-attachments/assets/602b7c68-b06a-443c-90fb-47dc03ada3b0" />
<img width="594" height="484" alt="Screenshot 2025-10-29 103603" src="https://github.com/user-attachments/assets/e47271e3-044b-436e-bccb-fc0b2baac5e1" />
<img width="1706" height="395" alt="Screenshot 2025-10-29 103546" src="https://github.com/user-attachments/assets/5a3968f0-b322-49f1-885a-c930c367f781" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
