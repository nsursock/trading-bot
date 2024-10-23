import numpy as np
import tensorflow as tf

# Convert data to tensor
def train_model_unsupervised(training_data, num_clusters=3, max_iterations=100):
    X_train = tf.convert_to_tensor(training_data, dtype=tf.float32)  # Convert to TensorFlow tensor

    kmeans_result = kmeans(X_train, num_clusters, max_iterations)
    centroids = kmeans_result['centroids'].numpy()  # Convert centroids to numpy array

    return {'centroids': centroids, 'numClusters': num_clusters}

# K-means++ initialization
def initialize_centroids_kmeans_plus_plus(data, num_clusters):
    num_samples = data.shape[0]
    centroids = []

    # Step 1: Randomly select the first centroid from the data points
    first_centroid_idx = np.random.randint(0, num_samples)
    centroids.append(tf.gather(data, first_centroid_idx))

    for i in range(1, num_clusters):
        # Compute distances from the already chosen centroids
        distances = tf.reduce_min(
            tf.stack([tf.norm(data - centroid, axis=1) for centroid in centroids]),
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

    # Step 1: Initialize centroids using K-means++
    centroids = initialize_centroids_kmeans_plus_plus(data, num_clusters)

    for iteration in range(max_iterations):
        # Step 2: Assign clusters
        distances = tf.stack([tf.norm(data - centroid, axis=1) for centroid in centroids])
        cluster_assignments = tf.argmin(distances, axis=0)

        # Step 3: Update centroids
        new_centroids = []
        for cluster_idx in range(num_clusters):
            mask = tf.equal(cluster_assignments, cluster_idx)
            cluster_points = tf.boolean_mask(data, mask)

            if cluster_points.shape[0] == 0:
                # If no points are assigned to the cluster, keep the old centroid
                new_centroid = centroids[cluster_idx]
            else:
                new_centroid = tf.reduce_mean(cluster_points, axis=0)

            new_centroids.append(new_centroid)

        new_centroids = tf.stack(new_centroids)

        # Check for convergence
        centroid_shift = tf.norm(centroids - new_centroids)
        centroids = new_centroids

        if centroid_shift.numpy() < 1e-6:
            break

    return {'centroids': centroids}

from sklearn.decomposition import PCA

def apply_pca(observation, n_components=10):
    # Flatten observation for PCA input
    flat_observation = observation.reshape(observation.shape[0], -1)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(flat_observation)
    
    return pca_result

import matplotlib.pyplot as plt  # Import matplotlib for plotting

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


# Function to plot the number of data points by market regimes
def plot_market_regimes(data, regimes):
    # Count occurrences of each regime
    regime_counts = {regime: 0 for regime in ['high vol', 'low vol', 'ranging', 'trending up', 'trending down']}
    for regime in regimes:
        if regime in regime_counts:
            regime_counts[regime] += 1

    # Prepare data for plotting
    regimes = list(regime_counts.keys())
    counts = list(regime_counts.values())

    # Create bar graph
    plt.figure(figsize=(10, 6))
    plt.bar(regimes, counts, color='skyblue')
    plt.title('Number of Data Points by Market Regimes')
    plt.xlabel('Market Regimes')
    plt.ylabel('Number of Data Points')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# # Function to calculate mu, drift, and categorize market regimes
# def categorize_market_regimes(data):
#     mu = np.mean(data, axis=0)  # Calculate mean (mu) of the data
#     drift = data - mu  # Calculate drift

#     regimes = []
#     for point in drift:
#         if point[0] > 0.1:  # Example threshold for bullish
#             regimes.append('bullish')
#         elif point[0] < -0.1:  # Example threshold for bearish
#             regimes.append('bearish')
#         else:
#             regimes.append('sideways')

#     return mu, drift, regimes



def categorize_market_regimes(data):
    mu = np.mean(data, axis=0)  # Calculate mean (mu) of the data
    drift = data - mu  # Calculate drift
    std_dev = np.std(data)  # Calculate standard deviation for debugging

    # Calculate dynamic thresholds
    low_vol_threshold = np.percentile(data, 10)  # 25th percentile for low volatility
    high_vol_threshold = np.percentile(data, 90)  # 75th percentile for high volatility

    print("Standard Deviation:", std_dev)  # Debugging output
    print("Low Volatility Threshold:", low_vol_threshold)
    print("High Volatility Threshold:", high_vol_threshold)

    regimes = []
    for point in drift:
        # Adjusted logic for categorization
        if std_dev > high_vol_threshold:  # High volatility
            regimes.append('high vol')
        elif std_dev < low_vol_threshold:  # Low volatility
            regimes.append('low vol')
        elif np.all(np.abs(point) < 0.1):  # Ranging
            regimes.append('ranging')
        elif point[0] > 0.1:  # Trending up
            regimes.append('trending up')
        elif point[0] < -0.1:  # Trending down
            regimes.append('trending down')
        else:
            regimes.append('ranging')  # Default to ranging if no other condition is met

    # Additional check to ensure proper categorization
    if all(regime == 'high vol' for regime in regimes):
        print("Warning: All points categorized as 'high vol'. Consider adjusting thresholds.")

    return mu, drift, regimes


# Example usage:
if __name__ == '__main__':
    from trading_bot import prepare
    from parameters import create_financial_params, selected_params
    import matplotlib.pyplot as plt  # Importing matplotlib for plotting

    # Load the financial parameters
    financial_params = create_financial_params(selected_params)
    
    # Load the training data
    training_data, _, _, _, _ = prepare(financial_params)
    print(training_data.shape)
    
    # Normalize training data
    normalized_training_data, scaler = normalize_data(training_data)
    
    # Example training data (replace this with your actual data)
    # training_data = np.random.randn(760, 10, 15)  # 100 samples, 5 features
    
    # Apply PCA to reduce dimensionality
    pca_data = apply_pca(normalized_training_data, 3)

    # Train the unsupervised model
    result = train_model_unsupervised(pca_data, num_clusters=3, max_iterations=100)
    print("Centroids:", result['centroids'])
    

    # Plotting the data and centroids in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # Create a 3D axis
    ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2], marker='x', label='Data Points')  # Plot data points
    ax.scatter(result['centroids'][:, 0], result['centroids'][:, 1], result['centroids'][:, 2], marker='o', color='red', label='Centroids', s=100)  # Plot centroids
    ax.set_title('Data Points and Centroids in 3D')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    ax.legend()
    plt.show()  # Show the plot
    
    
     # Calculate mu, drift, and categorize market regimes
    mu, drift, market_regimes = categorize_market_regimes(normalized_training_data.reshape(-1, normalized_training_data.shape[-1]))

    # Plot the number of data points by market regimes
    plot_market_regimes(normalized_training_data, market_regimes)

    # Print mu and drift for reference
    print("Mean (mu):", mu)
    print("Drift:", drift)
