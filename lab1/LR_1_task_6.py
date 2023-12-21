from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
from sklearn.naive_bayes import GaussianNB

# Load data from file
data = np.loadtxt('data_multivar_nb.txt', delimiter=',')

# Split data into features and labels
X = data[:, :-1]
y = data[:, -1]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train support vector machine model
clf_svc = svm.SVC(kernel='linear')
clf_svc.fit(X_train, y_train)

# Predict labels for test set
y_pred_svc = clf_svc.predict(X_test)

# Calculate classification accuracy levels
accuracy_svc = accuracy_score(y_test, y_pred_svc)
recall_svc = recall_score(y_test, y_pred_svc, average='weighted')
precision_svc = precision_score(y_test, y_pred_svc, average='weighted')
f1_svc = 2 * (precision_svc * recall_svc) / (precision_svc + recall_svc)
print(f"SVC classification accuracy: {accuracy_svc}")
print(f"SVC classification recall: {recall_svc}")
print(f"SVC classification precision: {precision_svc}")
print(f"SVC classification f1: {f1_svc}")


clf_bayes = GaussianNB()
clf_bayes.fit(X_train, y_train)

# Predict labels for test set
y_pred_bayes = clf_bayes.predict(X_test)

# Calculate classification accuracy levels
accuracy_bayes = accuracy_score(y_test, y_pred_bayes)
recall_bayes = recall_score(y_test, y_pred_bayes, average='weighted')
precision_bayes = precision_score(y_test, y_pred_bayes, average='weighted')
f1_bayes = 2 * (precision_bayes * recall_bayes) / (precision_bayes + recall_bayes)
print(f"Naive Bayes classification accuracy: {accuracy_bayes}")
print(f"Naive Bayes classification recall: {recall_bayes}")
print(f"Naive Bayes classification precision: {precision_bayes}")
print(f"Naive Bayes classification f1: {f1_bayes}")
