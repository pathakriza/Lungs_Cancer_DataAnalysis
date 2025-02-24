import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import accuracy_score, classification_report, r2_score
from scipy.cluster.hierarchy import dendrogram, linkage

# Load dataset
data = pd.read_csv('/content/lung_cancer_data.csv')

# Print first 5 rows
print("Dataset Head:")
print(data.head())

# Data Preprocessing
print("Dataset Info:")
print(data.info())
print("\nMissing Values:")
print(data.isnull().sum())

# Fill missing values (if any) for numeric columns only
numeric_cols = data.select_dtypes(include=np.number).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

# Encode categorical variables
label_encoders = {}
for col in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

 # Encode categorical variables EXCLUDING TumorStage
label_encoders = {}
for col in data.select_dtypes(include=['object']).columns:
    # Skip encoding 'TumorStage'
    if col == 'TumorStage':
        continue  
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Split dataset
X = data.drop(columns=['Survival_Months'])  # Replace with the target column
Y = data['Survival_Months']  # Change for classification if needed
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Machine Learning Models
## K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, Y_train)
knn_pred = knn.predict(X_test_scaled)
print("KNN Accuracy:", accuracy_score(Y_test, knn_pred))

## Naive Bayes
nb = GaussianNB()
nb.fit(X_train_scaled, Y_train)
nb_pred = nb.predict(X_test_scaled)
print("Naive Bayes Accuracy:", accuracy_score(Y_test, nb_pred))

## Support Vector Machine (SVM)
svm = SVC(kernel='linear')
svm.fit(X_train_scaled, Y_train)
svm_pred = svm.predict(X_test_scaled)
print("SVM Accuracy:", accuracy_score(Y_test, svm_pred))

## Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)
dt_pred = dt.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(Y_test, dt_pred))

## Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, Y_train)
lr_pred = lr.predict(X_test_scaled)
print("Linear Regression R2 Score:", r2_score(Y_test, lr_pred))

# Clustering
## K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_train_scaled)
plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=kmeans_labels, cmap='viridis')
plt.title('K-Means Clustering')
plt.show()

## Hierarchical Clustering
hc = AgglomerativeClustering(n_clusters=3)
hc_labels = hc.fit_predict(X_train_scaled)
plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=hc_labels, cmap='coolwarm')
plt.title('Hierarchical Clustering')
plt.show()

# Hierarchical Dendrogram
plt.figure(figsize=(10, 5))
linkage_matrix = linkage(X_train_scaled[:500], method='ward')  # Use sample for efficiency
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()

# Visualization
sns.histplot(data['Survival_Months'], bins=30, kde=True)
plt.title('Survival Months Distribution')
plt.show()

sns.boxplot(x=data['Stage'], y=data['Survival_Months'])
plt.title('Tumor Stage vs Survival Months')
plt.show()
