# PCA demo
# Uses PCA from sklearn.decomposition: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns; sns.set()

# Data
X_train = []
X_sample = []

# PCA
pca = PCA(n_components=2)
pca.fit(X_train)
# pca.explained_variance_
# pca.explained_variance_ratio_
# pca.components_
# pca.mean_
# pca.singular_values_

# Transform sample data
sample_weights = pca.transform(X_sample)

# Recreate from component weights
X_recreate = pca.mean_ + sample_weights.dot(pca.components_)
# OR
# X_recreate = pca.inverse_transform(sample_weights)
