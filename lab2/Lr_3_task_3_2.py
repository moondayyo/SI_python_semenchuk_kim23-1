from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_iris
import numpy as np

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# Shape
print(dataset.shape)

# Head
print(dataset.head(20))

# Descriptions
print(dataset.describe())

# Class distribution
print(dataset.groupby('class').size())

# Box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
pyplot.show()

# Histograms
dataset.hist()
pyplot.show()

# Scatter plot matrix
scatter_matrix(dataset)
pyplot.show()

# Split-out validation dataset
array = dataset.values

# All rows, columns 0-4
X = array[:, 0:4]

# All rows, column 4
y = array[:, 4]

# 80% training, 20% validation
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# Test options and evaluation metric
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# Evaluate each model in turn
results = []
names = []

for name, model in models:
    # 10-fold cross validation
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)

    # Evaluate model on 10 different splits of the dataset
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')

    # Store results
    results.append(cv_results)
    names.append(name)

    # Summarize performance
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Compare algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

# Make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# Load iris dataset
iris_dataset = load_iris()

# Create a NumPy array with the new iris data
new_data = np.array([[5.0, 2.9, 1.0, 0.2]])

# Get the shape of the array
shape = new_data.shape

# Print the shape of the array
print('Array shape:', shape)

# Define KNN model
model = SVC(gamma='auto')

# Fit the model to the training data
model.fit(X_train, Y_train)

# Make predictions on new data
new_prediction = model.predict(new_data)

# Print the prediction
print('Prediction:', new_prediction)