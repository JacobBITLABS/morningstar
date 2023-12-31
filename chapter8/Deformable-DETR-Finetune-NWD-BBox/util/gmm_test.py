import numpy as np
from sklearn.mixture import GaussianMixture

# Generate some example data drawn from a softmax distribution
np.random.seed(42)
num_samples = 100
scores = np.random.rand(num_samples, 2)  # Random scores between 0 and 1
softmax_scores = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)

print("softmax_scores")
print(softmax_scores)

# Fit a Gaussian Mixture Model with 2 components
gmm = GaussianMixture(n_components=2)
gmm.fit(softmax_scores)

# Get the assignments for each sample
gmm_assignment = gmm.predict(softmax_scores)

# Obtain the scores based on the fitted GMM
score_samples = gmm.score_samples(softmax_scores)

# Print the results
print("GMM Assignments:")
print(gmm_assignment)
print("\nScores based on GMM:")
print(score_samples)
