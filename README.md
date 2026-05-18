# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
 1. Import pandas
 2. Import Decision tree classifier 
 3. Fit the data in the model 
 4. Find the accuracy score

## Program:
```
import pandas as pd

data = pd.read_csv("Employee.csv")

print("data.head():")
print(data.head())

print("data.info():")
print(data.info())

print("isnull() and sum():")
print(data.isnull().sum())

print("data value counts():")
print(data["left"].value_counts())

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

print("data.head() for Salary:")
data["salary"] = le.fit_transform(data["salary"])
print(data.head())

print("x.head():")

x = data[["satisfaction_level",
          "last_evaluation",
          "number_project",
          "average_montly_hours",
          "time_spend_company",
          "Work_accident",
          "promotion_last_5years",
          "salary"]]

print(x.head())

y = data["left"]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=100
)

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion="entropy")

dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)

print("Accuracy value:")

from sklearn import metrics

accuracy = metrics.accuracy_score(y_test, y_pred)

print(accuracy)

print("Data Prediction:")

print(dt.predict([[0.5, 0.8, 9, 260, 6, 0, 1, 2]]))

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

plot_tree(
    dt,
    feature_names=x.columns,
    class_names=['salary', 'left'],
    filled=True
)

plt.show()


Developed by: B.SASIREKHA
RegisterNumber: 212225040388

```

## Output:
<img width="902" height="797" alt="image" src="https://github.com/user-attachments/assets/2fe4f307-e402-4e3d-b6d3-28cfe4909965" />
<img width="882" height="672" alt="image" src="https://github.com/user-attachments/assets/73f1e853-4965-43c7-b738-9f5750594f62" />
<img width="901" height="762" alt="image" src="https://github.com/user-attachments/assets/27752e71-05e2-4965-a218-be361cf96f1b" />
<img width="1365" height="751" alt="image" src="https://github.com/user-attachments/assets/c6718bf6-cb19-456b-8783-1c23b7e7e2a3" />




## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
