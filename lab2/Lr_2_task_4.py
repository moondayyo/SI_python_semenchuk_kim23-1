import numpy as np
from sklearn import preprocessing
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Load data
input_file = 'income_data.txt'

# Read the data
X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break

        if '?' in line:
            continue

        data = line[:-1].split(', ')

        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1

        if data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

# Convert to numpy array
X = np.array(X)

# Convert string data to numerical data
label_encoder = []
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

# Cross validation
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Train and evaluate the logistic regression model
lr_model = LogisticRegression(max_iter=10000)
lr_model.fit(X_train, Y_train)
lr_predictions = lr_model.predict(X_test)

print('Logistic Regression Accuracy:', accuracy_score(Y_test, lr_predictions))
print('Logistic Regression Precision:', precision_score(Y_test, lr_predictions, average='weighted'))
print('Logistic Regression Recall:', recall_score(Y_test, lr_predictions, average='weighted'))
print('Lofistic Regression F1 Score:', f1_score(Y_test, lr_predictions, average='weighted'))

# Train and evaluate the linear discriminant analysis model
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X_train, Y_train)
lda_predictions = lda_model.predict(X_test)

print('Linear Discriminant Accuracy:', accuracy_score(Y_test, lda_predictions))
print('Linear Discriminant Precision:', precision_score(Y_test, lda_predictions, average='weighted'))
print('Linear Discriminant Recall:', recall_score(Y_test, lda_predictions, average='weighted'))
print('Linear Discriminant F1 Score:', f1_score(Y_test, lda_predictions, average='weighted'))

# Train and evaluate the k-nearest neighbors model
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, Y_train)
knn_predictions = knn_model.predict(X_test)

print('K-Nearest Neighbors Accuracy:', accuracy_score(Y_test, knn_predictions))
print('K-Nearest Neighbors Precision:', precision_score(Y_test, knn_predictions, average='weighted'))
print('K-Nearest Neighbors Recall:', recall_score(Y_test, knn_predictions, average='weighted'))
print('K-Nearest Neighbors F1 Score:', f1_score(Y_test, knn_predictions, average='weighted'))

# Train and evaluate the decision tree model
cart_model = DecisionTreeClassifier()
cart_model.fit(X_train, Y_train)
cart_predictions = cart_model.predict(X_test)

print('Decision Tree Accuracy:', accuracy_score(Y_test, cart_predictions))
print('Decision Tree Precision:', precision_score(Y_test, cart_predictions, average='weighted'))
print('Decision Tree Recall:', recall_score(Y_test, cart_predictions, average='weighted'))
print('Decision Tree F1 Score:', f1_score(Y_test, cart_predictions, average='weighted'))

# Train and evaluate the naive bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, Y_train)
nb_predictions = nb_model.predict(X_test)

print('Naive Bayes Accuracy:', accuracy_score(Y_test, nb_predictions))
print('Naive Bayes Precision:', precision_score(Y_test, nb_predictions, average='weighted'))
print('Naive Bayes Recall:', recall_score(Y_test, nb_predictions, average='weighted'))
print('Naive Bayes F1 Score:', f1_score(Y_test, nb_predictions, average='weighted'))

# Train and evaluate the support vector machine model
svm_model = SVC()
svm_model.fit(X_train, Y_train)
svm_predictions = svm_model.predict(X_test)

print('Support Vector Machine Accuracy:', accuracy_score(Y_test, svm_predictions))
print('Support Vector Machine Precision:', precision_score(Y_test, svm_predictions, average='weighted'))
print('Support Vector Machine Recall:', recall_score(Y_test, svm_predictions, average='weighted'))
print('Support Vector Machine F1 Score:', f1_score(Y_test, svm_predictions, average='weighted'))