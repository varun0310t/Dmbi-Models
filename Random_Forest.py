import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import warnings
warnings.filterwarnings("ignore")

breast = pd.read_csv("breast-cancer.csv")

df = breast.copy()

df.drop(["Unnamed: 32","id"],axis = 1, inplace = True)

X = df.drop("diagnosis", axis = 1)
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, stratify = y, random_state = 42)


y = df.diagnosis
x = df.drop("diagnosis", axis = 1)

x.head()


ax = sns.countplot(x = "diagnosis", data = df)
plt.show()

b,m = df.diagnosis.value_counts()
print("Number of Benign: ", b)
print("Number of Malignant: ", m)

x.describe().T


data_dia = y
data = x

#standardization
data_n2 = (data-data.mean()) / data.std()

 

drop_list1 = ['perimeter_mean','radius_mean','compactness_mean','concave points_mean','radius_se','perimeter_se','radius_worst','perimeter_worst','compactness_worst','concave points_worst','compactness_se','concave points_se','texture_worst','area_worst']
x1 = x.drop(drop_list1, axis = 1 )       
x1.head()



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score


x_train, x_test, y_train, y_test = train_test_split(x1, y, test_size=0.3, random_state=42)


clf_rf = RandomForestClassifier(n_estimators=2, max_depth=2, min_samples_split=25, min_samples_leaf=10, random_state=43)      
clr_rf = clf_rf.fit(x_train,y_train)

ac_score = accuracy_score(y_test,clf_rf.predict(x_test))
print('Accuracy is: ',ac_score)

cnf_m = confusion_matrix(y_test,clf_rf.predict(x_test))

plt.figure(figsize=(3,3))
sns.heatmap(cnf_m, annot=True, annot_kws={"fontsize":20}, fmt='d', cbar=False, cmap='PuBu')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Base Model', color='navy', fontsize=15)
plt.show()

def predict_for_row(row_num):
    # Ensure the row number is within the valid range
    if row_num < 0 or row_num >= len(x1):
        return "Invalid row number"

    # Get the data for the specified row
    row_data = x1.iloc[row_num].values.reshape(1, -1)

    # Predict the diagnosis for the specified row
    prediction = clf_rf.predict(row_data)

    return prediction[0]
40

print(predict_for_row(10))
