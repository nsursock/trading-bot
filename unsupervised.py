import numpy as np
import tensorflow as tf

# Convert data to tensor
def train_model_unsupervised(training_data, num_clusters=3, max_iterations=100):
    X_train = tf.convert_to_tensor(training_data, dtype=tf.float32)  # Convert to TensorFlow tensor

    kmeans_result = kmeans(X_train, num_clusters, max_iterations)
    centroids = kmeans_result['centroids'].numpy()  # Convert centroids to numpy array

    return {'centroids': centroids, 'numClusters': num_clusters}

# K-means++ initialization
def initialize_centroids_kmeans_plus_plus(data, num_clusters, num_symbols, num_features):
    num_samples = data.shape[0]
    centroids = []

    # Step 1: Randomly select the first centroid from the data points
    first_centroid_idx = np.random.randint(0, num_samples)
    centroids.append(tf.gather(data, first_centroid_idx))

    for i in range(1, num_clusters):
        # Compute distances from the already chosen centroids
        distances = tf.reduce_min(
            tf.stack([tf.norm(data - tf.reshape(centroid, (1, num_symbols, num_features)), axis=[1, 2]) for centroid in centroids]),
            axis=0
        )

        # Convert distances to probabilities
        probabilities = distances / tf.reduce_sum(distances)
        cumulative_probabilities = tf.cumsum(probabilities)

        # Randomly select the next centroid based on probabilities
        rand = np.random.rand()
        next_centroid_idx = tf.argmax(cumulative_probabilities >= rand).numpy()
        centroids.append(tf.gather(data, next_centroid_idx))

    return tf.stack(centroids)

def kmeans(data, num_clusters, max_iterations):
    num_samples = data.shape[0]
    num_symbols = data.shape[1]
    num_features = data.shape[2]

    # Step 1: Initialize centroids using K-means++
    centroids = initialize_centroids_kmeans_plus_plus(data, num_clusters, num_symbols, num_features)

    for iter in range(max_iterations):
        # Step 2: Assign clusters
        distances = tf.stack([
            tf.norm(data - tf.reshape(centroid, (1, num_symbols, num_features)), axis=[1, 2]) for centroid in centroids
        ])

        cluster_assignments = tf.argmin(distances, axis=0)

        # Step 3: Update centroids
        new_centroids = []
        for cluster_idx in range(num_clusters):
            mask = tf.equal(cluster_assignments, cluster_idx)  # No need to reshape here

            cluster_points = tf.boolean_mask(data, mask)  # Select data points for the correct cluster
            if tf.size(cluster_points) > 0:
                new_centroid = tf.reduce_mean(cluster_points, axis=0)
            else:
                new_centroid = centroids[cluster_idx]

            new_centroids.append(new_centroid)

        new_centroids = tf.stack(new_centroids)

        # Step 4: Check for convergence
        if tf.reduce_sum(tf.norm(centroids - new_centroids, axis=[1, 2])) < 1e-6:
            break

        centroids = new_centroids

    return {"centroids": centroids}

from sklearn.preprocessing import StandardScaler

# Function to normalize training data
def normalize_data(data):
    # Flatten the data to (760, 10 * 15) to apply normalization
    flat_data = data.reshape(data.shape[0], -1)
    
    # Apply standardization
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(flat_data)
    
    # Reshape back to the original (760, 10, 15)
    normalized_data = normalized_data.reshape(data.shape)
    
    return normalized_data, scaler


from sklearn.decomposition import PCA

def apply_pca(observation, n_components=10):
    # Flatten observation for PCA input
    flat_observation = observation.reshape(observation.shape[0], -1)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(flat_observation)
    
    return pca_result


# Example usage:
if __name__ == '__main__':
    from trading_bot import prepare
    from parameters import create_financial_params, selected_params
    import matplotlib.pyplot as plt  # Importing matplotlib for plotting

    financial_params = create_financial_params(selected_params)
    
    # Example training data (replace this with your actual data)
    training_data = np.random.randn(100, 5, 2)  # 760 samples, 10 symbols, 15 features
    
    training_data, _, _, _, _ = prepare(financial_params)
    print(training_data.shape)
    
    # Normalize training data
    normalized_training_data, scaler = normalize_data(training_data)

    # Now apply K-Means on the normalized data
    result = train_model_unsupervised(normalized_training_data, num_clusters=5, max_iterations=100)

    # result = train_model_unsupervised(training_data, num_clusters=3, max_iterations=100)
    print("Centroids:", result['centroids'])
    
    
    # {{ edit_1 }}: Create regime ID based on cluster assignments
    cluster_assignments = tf.argmin(tf.stack([
        tf.norm(normalized_training_data - tf.reshape(centroid, (1, normalized_training_data.shape[1], normalized_training_data.shape[2])), axis=[1, 2]) 
        for centroid in result['centroids']
    ]), axis=0).numpy()

    regime_id = cluster_assignments.reshape(-1, 1)  # Reshape to match the data dimensions

    # Reshape regime_id to match the dimensions of normalized_training_data
    regime_id_reshaped = np.repeat(regime_id, normalized_training_data.shape[1], axis=1)  # Repeat regime_id for each symbol
    regime_id_reshaped = regime_id_reshaped[:, :, np.newaxis]  # Add a new axis for concatenation

    normalized_training_data_with_regime = np.concatenate((normalized_training_data, regime_id_reshaped), axis=2)


    # Plotting the samples and centroids
    plt.figure(figsize=(10, 6))
    for i in range(normalized_training_data.shape[0]):
        plt.scatter(normalized_training_data[i, :, 0], normalized_training_data[i, :, 1], marker='x', color='blue', alpha=0.5)  # Plot samples
    plt.scatter(result['centroids'][:, 0], result['centroids'][:, 1], marker='o', color='red', s=100, label='Centroids')  # Plot centroids
    plt.title('Samples and Centroids')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()  # Display the plot
    
    # Count the number of samples for each regime
    unique_regimes, counts = np.unique(regime_id, return_counts=True)

    # Create a bar graph
    plt.figure(figsize=(8, 5))
    plt.bar(unique_regimes.flatten(), counts, color='skyblue')
    plt.title('Number of Samples for Each Regime')
    plt.xlabel('Regime ID')
    plt.ylabel('Number of Samples')
    plt.xticks(unique_regimes.flatten())  # Set x-ticks to be the regime IDs
    plt.show()  # Display the bar graph


