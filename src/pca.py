from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np


def reduce_dimensions_pca(X, n_components=None, variance_threshold=None):

   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   
   if n_components is None and variance_threshold is None:
       n_components = min(X.shape[1], 10)  # Default to 10 components
       
   if variance_threshold:
       temp_pca = PCA()
       temp_pca.fit(X_scaled)
       cumsum = np.cumsum(temp_pca.explained_variance_ratio_)
       n_components = np.argmax(cumsum >= variance_threshold) + 1
       
   pca = PCA(n_components=n_components)
   X_reduced = pca.fit_transform(X_scaled)
   
   print(f"\nVariance explained: {np.sum(pca.explained_variance_ratio_):.3f}")
   
   return X_reduced, pca, scaler