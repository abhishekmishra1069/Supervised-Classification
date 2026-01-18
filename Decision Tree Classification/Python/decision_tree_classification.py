# Decision Tree Classification
# This script implements a Decision Tree classifier to predict user purchasing behavior
# based on age and estimated salary using the Social Network Ads dataset

# ============================================================================
# SECTION 1: IMPORT LIBRARIES
# ============================================================================
# Importing numerical computing library
import numpy as np
# Importing plotting library for visualizations
import matplotlib.pyplot as plt
# Importing data manipulation library
import pandas as pd

# ============================================================================
# SECTION 2: LOAD AND PREPARE DATASET
# ============================================================================
# Read the CSV file containing social network ads data
dataset = pd.read_csv('Social_Network_Ads.csv')
# Extract features (X): all columns except the last one
X = dataset.iloc[:, :-1].values
# Extract labels (y): only the last column (target variable - purchased or not)
y = dataset.iloc[:, -1].values

# ============================================================================
# SECTION 3: SPLIT DATA INTO TRAINING AND TEST SETS
# ============================================================================
# Import train_test_split function from sklearn
from sklearn.model_selection import train_test_split
# Split data: 75% training, 25% testing with random_state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# Print training and test data for verification
print(X_train)
print(y_train)
print(X_test)
print(y_test)

# ============================================================================
# SECTION 4: FEATURE SCALING
# ============================================================================
# Import StandardScaler for normalizing features to have mean 0 and std 1
from sklearn.preprocessing import StandardScaler
# Create scaler object
sc = StandardScaler()
# Fit scaler on training data and transform it
X_train = sc.fit_transform(X_train)
# Apply same scaling to test data
X_test = sc.transform(X_test)
# Print scaled data for verification
print(X_train)
print(X_test)

# ============================================================================
# SECTION 5: TRAIN DECISION TREE CLASSIFIER
# ============================================================================
# Import DecisionTreeClassifier from sklearn
from sklearn.tree import DecisionTreeClassifier
# Create classifier object with entropy criterion (information gain) and random state for reproducibility
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
# Train the classifier on scaled training data
classifier.fit(X_train, y_train)

# ============================================================================
# SECTION 6: MAKE PREDICTIONS
# ============================================================================
# Test prediction for a new user: age=30, estimated salary=87000
# The scaler transforms it to the same scale as training data
print(classifier.predict(sc.transform([[30, 87000]])))

# Predict for entire test set
y_pred = classifier.predict(X_test)
# Display predictions vs actual results side by side
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# ============================================================================
# SECTION 7: EVALUATE MODEL PERFORMANCE
# ============================================================================
# Import evaluation metrics from sklearn
from sklearn.metrics import confusion_matrix, accuracy_score
# Create confusion matrix to see true positives, true negatives, false positives, false negatives
cm = confusion_matrix(y_test, y_pred)
print(cm)
# Calculate and print accuracy score (percentage of correct predictions)
accuracy_score(y_test, y_pred)

# ============================================================================
# SECTION 8: VISUALIZE TRAINING SET RESULTS
# ============================================================================
# Import ListedColormap for custom color mapping
from matplotlib.colors import ListedColormap
# Inverse transform training data to original scale for visualization
X_set, y_set = sc.inverse_transform(X_train), y_train
# Create mesh grid for contour plot (defines decision boundary area)
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
# Create contour plot showing decision boundaries
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# Set axis limits based on data range
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
# Plot actual data points, colored by class
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
# Add labels and title
plt.title('Decision Tree Classification (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# ============================================================================
# SECTION 9: VISUALIZE TEST SET RESULTS
# ============================================================================
# Inverse transform test data to original scale
X_set, y_set = sc.inverse_transform(X_test), y_test
# Create mesh grid for test set visualization
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
# Create contour plot for test set
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# Set axis limits
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
# Plot test data points
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
# Add labels and title
plt.title('Decision Tree Classification (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()