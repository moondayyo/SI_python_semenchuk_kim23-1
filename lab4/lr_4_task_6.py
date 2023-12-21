from sklearn.datasets import _samples_generator
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier

# Generate sample data
X, y = _samples_generator.make_classification(n_samples=150, n_features=25, n_classes=3, n_informative=6, n_redundant=0, random_state=7)

# Select features using the chi-squared test
selector_k_best = SelectKBest(f_regression, k=9)

# Build the machine learning pipeline
classifier = ExtraTreesClassifier(n_estimators=60, max_depth=4)

pipeline_classifier = Pipeline([('selector', selector_k_best), ('erf', classifier)])

# Establish the parameters
pipeline_classifier.set_params(selector__k=7, erf__n_estimators=30)

# Training the classifier
pipeline_classifier.fit(X, y)

# Predict the output
prediction = pipeline_classifier.predict(X)
print("\nPredictions:\n", prediction)

# Print scores
print("\nScores:\n", pipeline_classifier.score(X, y))

# Print the selected features chosen by the selector
features_status = pipeline_classifier.named_steps['selector'].get_support()

# Get selected feature indices
selected_features = [i for i, x in enumerate(features_status) if x]
print("\nSelected features (0-indexed):", ', '.join([str(x) for x in selected_features]))
