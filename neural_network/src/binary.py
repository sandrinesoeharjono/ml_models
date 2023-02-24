from structlog.stdlib import get_logger
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    RocCurveDisplay,
    roc_auc_score,
    classification_report
)

logger = get_logger()

# Parameters
train_size = 0.7
test_size = 1.0 - train_size

# Load dataset 
data = load_breast_cancer()
X = data.data
y = data.target
logger.info(data.target_names)

# Split into train & test set
logger.info(f"We are working with a dataset of shape {X.shape}. Splitting into train/test set using {train_size:.1f}:{test_size:.1f} ratio.")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=123)

# Apply standardization to training & test data
logger.info("Standardizing dataset...")
scaler = StandardScaler()  
scaler.fit(X_train)  
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)

# Create a Multi-layer Perceptron (MLP) classifier
logger.info("Creating multi-class MLP classifier...")
mlp_clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
mlp_clf.fit(X, y)

# Try model onto test set
y_pred = mlp_clf.predict(X_test)
logger.info(f"Model predicted {X_test.shape[0]} labels onto the test dataset.")

# Check model coefficients (weight matrices containing parameters)
coefs = [coef.shape for coef in mlp_clf.coefs_]
prob_estimates = mlp_clf.predict_proba(X_test)[::,1]

# Let's check the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, prob_estimates)
logger.info(f"Accuracy = {accuracy}")
logger.info(f"AUC = {auc}")
print(classification_report(y_test, y_pred))

# Visualize ROC curve
logger.info("Plotting the ROC curve...")
RocCurveDisplay.from_predictions(
    y_test,
    prob_estimates,
    name="ROC Curve",
    color="darkorange",
)
plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve of MLP Classifier onto Breast Cancer Data")
plt.legend()
plt.show()
