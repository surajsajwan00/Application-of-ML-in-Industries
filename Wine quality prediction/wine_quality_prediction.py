
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

## XGBClassifier Model 
class XGBClassifier:
    def __init__(self, learning_rate=0.1, n_estimators=100, max_depth=3):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    # Activation function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        self.classes = np.unique(y)
        for cls in self.classes:
            binary_y = np.where(y == cls, 1, 0)
            tree = self.build_tree(X, binary_y)
            self.trees.append(tree)

    def build_tree(self, X, y, depth=0):
        if depth == self.max_depth or len(np.unique(y)) == 1:
            return self.create_leaf_node(y)

        feature_index, threshold = self.find_best_split(X, y)
        left_indices = X[:, feature_index] <= threshold
        right_indices = ~left_indices

        left_tree = self.build_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self.build_tree(X[right_indices], y[right_indices], depth + 1)

        return {'feature_index': feature_index, 'threshold': threshold,
                'left': left_tree, 'right': right_tree}

    def create_leaf_node(self, y):
        p = np.mean(y)
        return {'leaf': True, 'p': p}

    def find_best_split(self, X, y):
        num_features = X.shape[1]
        best_feature = None
        best_threshold = None
        best_gini = float('inf')

        for feature_index in range(num_features):
            unique_values = np.unique(X[:, feature_index])
            for threshold in unique_values:
                left_indices = X[:, feature_index] <= threshold
                right_indices = ~left_indices
                gini = self.calculate_gini(y[left_indices], y[right_indices])

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    def calculate_gini(self, left_y, right_y):
        total_size = len(left_y) + len(right_y)
        left_size = len(left_y) / total_size
        right_size = len(right_y) / total_size

        gini_left = 1 - np.sum((np.unique(left_y, return_counts=True)[1] / len(left_y))**2)
        gini_right = 1 - np.sum((np.unique(right_y, return_counts=True)[1] / len(right_y))**2)

        gini = left_size * gini_left + right_size * gini_right
        return gini

    def predict(self, X):
        predictions = np.zeros((len(X), len(self.classes)))

        for i, tree in enumerate(self.trees):
            predictions[:, i] = self.predict_tree(X, tree)

        return np.argmax(predictions, axis=1)

    def predict_tree(self, X, tree):
        if 'leaf' in tree and tree['leaf']:
            return np.full(len(X), tree['p'])
        else:
            left_indices = X[:, tree['feature_index']] <= tree['threshold']
            right_indices = ~left_indices

            predictions = np.zeros(len(X))
            predictions[left_indices] = self.predict_tree(X[left_indices], tree['left'])
            predictions[right_indices] = self.predict_tree(X[right_indices], tree['right'])

            return predictions


# Load the dataset
wine_data = pd.read_csv('dataset/winequality.csv')
print(wine_data.head())

wine_data.describe().T

## Data Preprocessing
wine_data.isnull().sum()

df = wine_data.drop(columns=['type'])
winedf = wine_data.drop(columns=['type'])
print(wine_data.head())

for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mean())

df.isnull().sum().sum()

## Data Visualization
# Histogram
df.hist(bins=20, figsize=(10, 10))
plt.show()

# Bar plot
plt.bar(df['quality'], df['alcohol'])
plt.xlabel('quality')
plt.ylabel('alcohol')
plt.show()

# Heatmap
plt.figure(figsize=(12, 12))
sb.heatmap(df.corr() > 0.7, annot=True, cbar=False)
plt.show()


df = df.drop('total sulfur dioxide', axis=1)
df['best quality'] = [1 if x > 5 else 0 for x in df.quality]
df.replace({'white': 1, 'red': 0}, inplace=True)
features = df.drop(['quality', 'best quality'], axis=1)
target = df['best quality']

xtrain, xtest, ytrain, ytest = train_test_split(
    features, target, test_size=0.2, random_state=40)

xtrain.shape, xtest.shape

# Standard scaling
scaler = StandardScaler()
xtrain_scaled = scaler.fit_transform(xtrain)
xtest_scaled = scaler.transform(xtest)

# Dimensionality reduction using PCA
pca = PCA(n_components=8)  # number of components we want to keep
xtrain_pca = pca.fit_transform(xtrain_scaled)
xtest_pca = pca.transform(xtest_scaled)

print("Original shape of xtrain:", xtrain.shape)
print("Shape of xtrain after PCA:", xtrain_pca.shape)

print("Original shape of xtest:", xtest.shape)
print("Shape of xtest after PCA:", xtest_pca.shape)

# Normalization of the data
norm = MinMaxScaler()
xtrain = norm.fit_transform(xtrain)
xtest = norm.transform(xtest)

# Trainning the Model 
model = XGBClassifier(learning_rate=0.1, n_estimators=100, max_depth=3)
model.fit(xtrain, ytrain)

# Predictions
predictions = model.predict(xtest)

# Evaluate  the Model
accuracy = np.sum(predictions == ytest) / len(ytest)
print('Validation Accuracy:', accuracy)