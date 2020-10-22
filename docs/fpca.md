# Functional Principal Component Analysis (fPCA)

## The general idea
[Principal component analysis (PCA)](https://en.wikipedia.org/wiki/Principal_component_analysis) is popular technique for dimensionality reduction and data compression. The idea is to find the directions in a high-dimensional space in which we find the most variation in the data. To reduce dimensionality (or compress the data), we then only look at projections on the first few principal directions. 

[Functional principal component analysis (fPCA)](https://en.wikipedia.org/wiki/Functional_principal_component_analysis) is an extension of this method to situations where data consist of functions, not vectors. Although mathematically more involved than regular PCA, the two are equivalent from a practical point of view. To keep it simple, we shall therefore explain the core ideas in terms of the regular PCA.

## Mathematical definition

Suppose each observation in the data is a vector $X$. The first principal component is defined as the direction $w$ in which there's the most variation:

$$v_1 = \arg\max_{\|v\| = 1} \mathrm{var}[X'v].$$

In practice, the variance of $X'v$ is approximated by the [sample variance](https://en.wikipedia.org/wiki/Variance#Sample_variance). The other principal components are defined similarly,

$$v_k = \arg\max_{\|v\| = 1} \mathrm{var}[X'v],$$

under the side-constraint that $v_{k}$ is orthogonal to all previous principal components $v_{1}, \dots, v_{k - 1}$. In practice, the collection of all principal components can be found in one go as the eigenvectors of the covariance matrix of $X$.

## PCA as a basis expansion

For a $d$-dimensional vector $X$, there are in total $d$ principal components to be found: $v^{1}, \dots, v_{d}$. The principal components form a basis of the space, i.e., every vector $X$ can be represented as 

$$X = \mu + \sum_{k = 1}^d a_k v_{k},$$

where $\mu$ is $E[X]$ and the coefficients $a_k = X'v_k$ are called *principal component scores*. When $d$ is very large, we can truncate the sum above at $K \ll d$ terms. This gives an approximation 

$$X \approx \mu + \sum_{k = 1}^K a_k v_{k}.$$

This allows us to represent the high-dimensional vector $X$  by only a few numbers $a_1, \dots, a_K$. The number $K$ determines the quality of the approximation.

## PCA for synthetic data generation

PCA can be used to generate synthetic data for the high-dimensional vector $X$. For every instance $X_i$ in the data set, we compute the principal component scores $a_{i, 1}, \dots, a_{i, K}$. Because the principal components $v_1, \dots, v_K$ are orthogonal, the scores are necessarily uncorrelated and we may treat them as independent. We then fit a model $F_k$ for the marginal distribution of each score $a_k$. From these models, we can generate synthetic scores $\tilde a_k$ and transform them into a synthetic sample of $X$ via 

$$\tilde X = \mu + \sum_{k = 1}^K \tilde a_k v_{k}.$$
