from itertools import cycle
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    auc,
    roc_curve
)

import matplotlib.pyplot as plt
import numpy as np

# Define directories of interest
working_dir = Path(__file__).resolve()
DATA_ROOT = working_dir.parents[1] / "data"

# Parameters
train_size = 0.7
test_size = 1.0 - train_size
n_iter = 300
act_fn = "relu"

# Load dataset 
X = np.load(DATA_ROOT / "Droso_breeding_genex.npy", allow_pickle=True).astype(float)
y = np.load(DATA_ROOT / "Droso_breeding_labels.npy", allow_pickle=True).astype(float)
n_classes = len(set(y))

# Binarize the classes
y = label_binarize(y, classes=[0, 1, 2])

# Split into train & test set
print(f"We are working with a dataset of shape {X.shape}. Splitting into train/test set using {train_size:.1f}:{test_size:.1f} ratio.")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=123)

# Apply standardization to training & test data
print("Standardizing dataset...")
scaler = StandardScaler()  
scaler.fit(X_train)  
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)

# Create a Multi-layer Perceptron (MLP) classifier
print("Creating multi-class MLP classifier...")
mlp_clf = MLPClassifier(
    hidden_layer_sizes=(150,100,50),
    max_iter=n_iter,
    activation=act_fn,
    solver="adam"
)
mlp_clf.fit(X, y)

# Use model to make predictions on test set
y_pred = mlp_clf.predict(X_test)
print(f"Model predicted {X_test.shape[0]} labels onto the test dataset.")

# Check how the model performed
y_score = mlp_clf.predict_proba(X_test)
prob_estimates = y_score[::,1]
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy_score:.2f}")
print(f"Probability estimates: {prob_estimates}")
print(classification_report(y_test, y_pred))

# Visualize the ROC curve for each individual class 
print("Plotting the ROC curve...")
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
colors = cycle(["blue", "red", "green"])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], "k--", lw=2)
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic for Multi-Class Data")
plt.legend(loc="lower right")
plt.show()

# Perform hyperparameter tuning
param_grid = {
    "hidden_layer_sizes": [(150,100,50), (120,80,40), (100,50,30)],
    "max_iter": [50, 100, 150],
    "activation": ["tanh", "relu"],
    "solver": ["sgd", "adam"],
    "alpha": [0.0001, 0.05],
    "learning_rate": ["constant","adaptive"],
}
grid = GridSearchCV(mlp_clf, param_grid, n_jobs=-1, cv=5)
grid.fit(X_train, y_train)
grid_predictions = grid.predict(X_test) 
print("Accuracy: {:.2f}".format(accuracy_score(y_test, grid_predictions)))
print("Best parameters:")
print(grid.best_params_)