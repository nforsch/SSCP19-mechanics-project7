from pyDOE import *
from scipy.stats.distributions import norm

# Latin Hypercube Sampling
# see: https://pythonhosted.org/pyDOE/randomized.html

# Run LHS for n factors
X = lhs(4, samples=100) # lhs(n, [samples, criterion, iterations])

# Transform factors to normal distributions with means and standard deviations
means = [1, 2, 3, 4]
stdvs = [0.1, 0.5, 1, 0.25]
for i in range(4):
    X[:, i] = norm(loc=means[i], scale=stdvs[i]).ppf(X[:, i])
