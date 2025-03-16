import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
from sklearn.linear_model import LogisticRegression
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

logreg = LogisticRegression().fit(X_train,y_train)
y_pred = logreg.predict(X_test)

print(classification_report(y_test, y_pred))


plt.figure(figsize=(3,3))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, annot_kws={"fontsize":20}, fmt='d', cbar=False, cmap='PuBu')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Base Model', color='navy', fontsize=15)
plt.show()


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

data = pd.concat([y,data_n2.iloc[:,0:10]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')

# plotting the violin plot
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True, inner="quart")
plt.xticks(rotation=90)
plt.show()


# second ten part
data = pd.concat([y,data_n2.iloc[:,10:20]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')

# plotting the violin plot
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True, inner="quart")
plt.xticks(rotation=90)
plt.show()

# last ten part
data = pd.concat([y,data_n2.iloc[:,20:31]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')

#


corr = x.corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
f, ax = plt.subplots(figsize=(20, 15))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, annot=True,fmt='.2f',mask=mask, cmap=cmap, ax=ax)


drop_list1 = ['perimeter_mean','radius_mean','compactness_mean','concave points_mean','radius_se','perimeter_se','radius_worst','perimeter_worst','compactness_worst','concave points_worst','compactness_se','concave points_se','texture_worst','area_worst']
x1 = x.drop(drop_list1, axis = 1 )       
x1.head()


corr = x1.corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
f, ax = plt.subplots(figsize=(12, 6))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, annot=True,fmt='.2f',mask=mask, cmap=cmap, ax=ax)


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score


x_train, x_test, y_train, y_test = train_test_split(x1, y, test_size=0.3, random_state=42)

#n_estimators=10 (default)
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