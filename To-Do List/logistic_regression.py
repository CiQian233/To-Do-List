import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import scipy.io as sio

# Load data
data = sio.loadmat('f:\\Homework\\机械学习\\实验2\\data_digits.mat')
X = data['X']
y = data['y'].ravel()

# Convert y to 0-9 instead of 1-10
y[y == 10] = 0

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Data shape: X={X.shape}, y={y.shape}")
print(f"Train shape: X={X_train.shape}, y={y_train.shape}")
print(f"Test shape: X={X_test.shape}, y={y_test.shape}")

# Part 1: Using sklearn's Logistic Regression
print("\n=== Part 1: Using sklearn's Logistic Regression ===")
clf = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
clf.fit(X_train, y_train)
y_pred_sklearn = clf.predict(X_test)
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
print(f"Accuracy using sklearn: {accuracy_sklearn:.4f}")

# Part 2: Implement our own Logistic Regression with One-vs-All
print("\n=== Part 2: Implementing our own Logistic Regression ===")

class LogisticRegressionOVR:
    def __init__(self, learning_rate=0.01, num_iterations=1000, lambda_reg=0.01):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.lambda_reg = lambda_reg
        self.models = []
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))
    
    def compute_cost(self, X, y, theta):
        m = len(y)
        h = self.sigmoid(X.dot(theta))
        cost = -1/m * (y.T.dot(np.log(h)) + (1 - y).T.dot(np.log(1 - h)))
        reg_cost = (self.lambda_reg/(2*m)) * np.sum(theta[1:]**2)
        return cost + reg_cost
    
    def gradient_descent(self, X, y, theta):
        m = len(y)
        h = self.sigmoid(X.dot(theta))
        gradient = (1/m) * X.T.dot(h - y)
        gradient[1:] += (self.lambda_reg/m) * theta[1:]
        return gradient
    
    def fit(self, X, y):
        # Add bias term
        X_bias = np.insert(X, 0, 1, axis=1)
        num_classes = len(np.unique(y))
        
        for c in range(num_classes):
            # Create binary classification problem for class c
            y_binary = np.where(y == c, 1, 0)
            
            # Initialize parameters
            theta = np.zeros(X_bias.shape[1])
            
            # Gradient descent
            for _ in range(self.num_iterations):
                gradient = self.gradient_descent(X_bias, y_binary, theta)
                theta -= self.learning_rate * gradient
            
            self.models.append(theta)
    
    def predict(self, X):
        X_bias = np.insert(X, 0, 1, axis=1)
        predictions = []
        
        for theta in self.models:
            prob = self.sigmoid(X_bias.dot(theta))
            predictions.append(prob)
        
        # Choose class with highest probability
        return np.argmax(predictions, axis=0)

# Train our model
our_clf = LogisticRegressionOVR(learning_rate=0.01, num_iterations=1000, lambda_reg=0.01)
our_clf.fit(X_train, y_train)
y_pred_our = our_clf.predict(X_test)
accuracy_our = accuracy_score(y_test, y_pred_our)
print(f"Accuracy using our implementation: {accuracy_our:.4f}")

# Plot results
plt.figure(figsize=(12, 6))

# Plot some test samples with predictions
plt.subplot(1, 2, 1)
plt.title('sklearn Logistic Regression')
for i in range(9):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_test[i].reshape(20, 20), cmap='gray')
    plt.title(f'Pred: {y_pred_sklearn[i]}')
    plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Our Logistic Regression')
for i in range(9):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_test[i].reshape(20, 20), cmap='gray')
    plt.title(f'Pred: {y_pred_our[i]}')
    plt.axis('off')

plt.tight_layout()
plt.savefig('logistic_regression_results.png')
plt.show()

print("\n=== Summary ===")
print(f"sklearn accuracy: {accuracy_sklearn:.4f}")
print(f"Our implementation accuracy: {accuracy_our:.4f}")
