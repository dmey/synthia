# Copulas


## What copulas are

[Copulas](https://en.wikipedia.org/wiki/Copula_(probability_theory)) are probabilistic models for the dependence between random quantities $Z_1, \dots, Z_d$. The idea is to 
decouple the individual (or marginal) behavior of the quantities from the dependence. 
The key result, Sklar’s theorem  {cite}`sklar1959`, states that for any joint distribution $F$ on $d$ variables can be written as 

$$F(z_1,\dots, z_d) = C(F_1(z_1), \dots,F_d(z_d)), $$

where $F_1, \dots, F_d$ are the marginal distribution functions and $C$ is the *copula* function. The copula encodes all the information that is not contained in the marginals, i.e., the dependence between variables. 
The simplest example for a copula function is $C(u_1, \dots, u_d) = u_1 \times \cdots \times u_d$ which corresponds to independence: $F(z_1,\dots, z_d) = F_1(z_1) \times \cdots \times F_d(z_d)$. 


## The Gaussian copula

To induce dependence in $F$, it is common to consider sub-families of copulas that are conveniently parametrized. There's a variety of such parametric copula families. The most prominent one is the *Gaussian copula*. It is defined by inversion of Sklar's theorem:

$$C(z_1,\dots, z_d) = F(F_1^{-1}(z_1), \dots, F_d^{-1}(z_d)), $$

where $F$ is a multivariate Gaussian distribution and $F_1, \dots, F_d$ the corresponding marginals. The Gaussian copula is then parameterized by a correlation matrix and subsumes all possible dependence structure in a multivariate Gaussian distribution. The benefit comes from the fact that we can combine a given copula with any type of marginal distributions, not just the ones the copula was derived from. That way, we can build flexible models with arbitrary marginal distributions and Gaussian-like dependence. 


## Other copula families

The same principle applies to other multivariate distributions and many copula models have been derived, most prominently the Student t copula and Archimedean families. A comprehensive list can be found in {cite}`joe2014dependence`. When there are more than two variables ($d>2$) the types of dependence structures these models can generate is rather limited. For example, Gaussian and Student copulas only allow for symmetric dependencies. While Archimedean families allow for such asymmetries, they require all pairs of variables to have the same type and strength of dependence.

## Vine copulas

[Vine copula models](https://en.wikipedia.org/wiki/Vine_copula) ({cite}`aas2009pair`, {cite}`czado2019analyzing`) are a popular solution to this issue. The idea is to build a large dependence model from only two-dimensional building blocks. We can explain this with a simple example with just three variables $Z_1, Z_2, Z_3$.

We model the dependence between $Z_1$ and $Z_2$ by a two-dimensional copula 
$C_{1,2}$ and the dependence between $Z_2$ and $Z_3$ by another, possibly different, copula $C_{2,3}$. These two copulas already contain some information about the dependence between $Z_1$ and $Z_3$, the part of the dependence that is induced by $Z_2$. The missing piece is the dependence between $Z_1$ and $Z_3$ after the effect of $Z_2$ has been removed. Mathematically, this is the conditional dependence $Z_1$ and $Z_3$ given 
$Z_2$ and can be modeled by yet another two-dimensional copula 
$C_{1,3|2}$. The principle is easily extended to an arbitrary number of variables $Z_1, \dots, Z_d$. 

Because all two-dimensional copulas can be specified independently, such models are extremely flexible and allow for highly heterogeneous dependence structures. Algorithms for simulation and selecting the right conditioning order and parametric families for each (conditional) pair are given in {cite}`dissmann2013`. Using parametric models for pair-wise dependencies remain a limiting factor, however. If necessary, it is also possible to use nonparametric models for the two-dimensional building blocks, see {cite}`Nagler2017`.


```{bibliography} references.bib
:style: alpha
```
