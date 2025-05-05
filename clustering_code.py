# Task 2: Clustering Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.cm as cm

# Assuming df is already loaded from previous code

# 1. Prepare data for clustering
# Separate numerical and categorical features
numerical_features = ['base', 'bonus', 'overtime_pay', 'other', 'age', 
                      'duration_total', 'duration_entity', 'duration_nominal', 'duration_overtime']
categorical_features = ['sector', 'section_07', 'sex', 'education', 'contract']

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# Fit and transform the data
X_processed = preprocessor.fit_transform(df[numerical_features + categorical_features])

# 2. Determine optimal number of clusters using the Elbow Method and Silhouette Score
def find_optimal_clusters(data, max_k):
    inertias = []
    silhouette_scores = []
    k_values = range(2, max_k+1)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
        
        # Calculate silhouette score
        labels = kmeans.labels_
        silhouette_avg = silhouette_score(data, labels)
        silhouette_scores.append(silhouette_avg)
        print(f"For n_clusters = {k}, the silhouette score is {silhouette_avg:.3f}")
    
    # Plot Elbow Method
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(k_values, inertias, 'o-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    
    # Plot Silhouette Scores
    plt.subplot(1, 2, 2)
    plt.plot(k_values, silhouette_scores, 'o-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal k')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return inertias, silhouette_scores

# Find optimal number of clusters
inertias, silhouette_scores = find_optimal_clusters(X_processed, 10)

# 3. Apply K-means clustering with the optimal number of clusters
# Based on the elbow method and silhouette scores, choose optimal k
# For now, let's assume k=4 (this should be adjusted based on the results)
optimal_k = 4  # This should be adjusted after seeing the elbow plot

# Apply K-means with the optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_processed)

# Add cluster labels to the original dataframe
df['cluster'] = cluster_labels

# 4. Visualize the clusters using PCA for dimensionality reduction
# Apply PCA to reduce dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_processed)

# Plot the clusters
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap=cm.tab10, alpha=0.7)
plt.colorbar(scatter, label='Cluster')
plt.title('Clusters Visualization using PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True, alpha=0.3)
plt.show()

# 5. Analyze cluster characteristics
# Calculate mean values for each numerical feature by cluster
cluster_means = df.groupby('cluster')[numerical_features].mean()
print("\nCluster Means for Numerical Features:")
print(cluster_means)

# Create a heatmap of the cluster means
plt.figure(figsize=(14, 8))
sns.heatmap(cluster_means, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=.5)
plt.title('Cluster Means for Numerical Features')
plt.tight_layout()
plt.show()

# Analyze categorical features distribution within each cluster
for feature in categorical_features:
    plt.figure(figsize=(14, 8))
    for i in range(optimal_k):
        plt.subplot(1, optimal_k, i+1)
        cluster_data = df[df['cluster'] == i]
        
        if feature == 'sex':
            cluster_data['sex_label'] = cluster_data['sex'].map({1: 'man', 2: 'woman'})
            sns.countplot(y='sex_label', data=cluster_data)
            plt.title(f'Cluster {i}: Sex Distribution')
        elif feature == 'sector':
            cluster_data['sector_label'] = cluster_data['sector'].map({1: 'public', 2: 'private'})
            sns.countplot(y='sector_label', data=cluster_data)
            plt.title(f'Cluster {i}: Sector Distribution')
        elif feature == 'section_07':
            cluster_data['section_07_label'] = cluster_data['section_07'].map({
                1: 'Public Admin', 
                2: 'Education', 
                3: 'Health & Social'
            })
            sns.countplot(y='section_07_label', data=cluster_data)
            plt.title(f'Cluster {i}: Section Distribution')
        elif feature == 'education':
            cluster_data['education_label'] = cluster_data['education'].map({
                1: 'doctorate', 
                2: 'higher', 
                3: 'post-secondary', 
                4: 'secondary', 
                5: 'basic vocational', 
                6: 'middle school and below'
            })
            sns.countplot(y='education_label', data=cluster_data)
            plt.title(f'Cluster {i}: Education Distribution')
        elif feature == 'contract':
            cluster_data['contract_label'] = cluster_data['contract'].map({
                1: 'indefinite', 
                2: 'definite'
            })
            sns.countplot(y='contract_label', data=cluster_data)
            plt.title(f'Cluster {i}: Contract Distribution')
    
    plt.tight_layout()
    plt.show()

# 6. Summarize the findings
print("\nCluster Sizes:")
print(df['cluster'].value_counts())

# Calculate the percentage of each cluster
cluster_percentages = (df['cluster'].value_counts() / len(df) * 100).round(2)
print("\nCluster Percentages:")
print(cluster_percentages)

# Create a pie chart of cluster sizes
plt.figure(figsize=(10, 8))
plt.pie(cluster_percentages, labels=[f'Cluster {i} ({p}%)' for i, p in 
                                    zip(cluster_percentages.index, cluster_percentages)],
        autopct='%1.1f%%', startangle=90, shadow=True)
plt.title('Cluster Size Distribution')
plt.axis('equal')
plt.show()

# 7. Interpret the clusters
print("\nCluster Interpretation:")
for i in range(optimal_k):
    print(f"\nCluster {i}:")
    # Get the top 3 distinguishing numerical features
    cluster_profile = cluster_means.loc[i] - cluster_means.mean()
    top_features = cluster_profile.abs().sort_values(ascending=False).head(3).index
    
    for feature in top_features:
        if cluster_profile[feature] > 0:
            print(f"- Higher {feature}: {cluster_means.loc[i, feature]:.2f} vs overall mean {cluster_means[feature].mean():.2f}")
        else:
            print(f"- Lower {feature}: {cluster_means.loc[i, feature]:.2f} vs overall mean {cluster_means[feature].mean():.2f}")