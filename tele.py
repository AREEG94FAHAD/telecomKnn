import os
import joblib
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('teleCust1000t.csv')
# df.head(3)


# Let’s see how many of each class is in our data set
df[['custcat']].value_counts()

X = df[['region', 'tenure', 'age', 'marital', 'address', 'income',
        'ed', 'employ', 'retire', 'gender', 'reside']] .values
y = df['custcat'].values

# Normalize data
stander = preprocessing.StandardScaler()
stander.fit(X)
X = stander.transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=4)
print('Train set:', X_train.shape,  y_train.shape)
print('Test set:', X_test.shape,  y_test.shape)


# Split the data into 80% for training and 20% for testing.
k = 2
# Train Model and Predict
model = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)

yhat = model.predict(X_test)

print("Train set Accuracy: ", metrics.accuracy_score(
    y_train, model.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

k = 20
results = []
for i in range(1, k+1):
    neigh = KNeighborsClassifier(n_neighbors=i).fit(X_train, y_train)
    yhat = neigh.predict(X_test)
    ac = metrics.accuracy_score(y_test, yhat)
    results.append(ac)

plt.bar(np.arange(1, k+1), [i * 100 for i in results])
plt.xlabel('K value')
plt.ylabel('Accuracy score')

print('The best accuracy achieved where k is equal to ', str(
    results.index(max(results))), "The accuracy is ", str(results[15]*100)+"%")


# Deploying

if not os.path.exists('Model'):
    os.mkdir('Model')
if not os.path.exists('Scaler'):
    os.mkdir('Scaler')

joblib.dump(model, r'Model/model.pickle')
joblib.dump(stander, r'Scaler/scaler.pickle')



# test the model using real data
new_data = pd.DataFrame([{'region': 2, 'tenure': 13, 'age': 45, 'marital': 1, 'address': 9,
                        'income': 64.000, 'ed': 4, 'employ': 5, 'retire': 0.000, 'gender': 0, 'reside': 2}])
new_data = new_data[['region', 'tenure', 'age', 'marital',
                     'address', 'income', 'ed', 'employ', 'retire', 'gender', 'reside']]
print(new_data)


model = joblib.load(r'Model/model.pickle')
scaler = joblib.load(r'Scaler/scaler.pickle')
new_data = scaler.transform(new_data)
model.predict(new_data)
