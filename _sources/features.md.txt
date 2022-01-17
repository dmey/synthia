# Features

Synthia includes several software features for the generation and augmentation of multivariate data (Table 1). You can find out more about how to use these option in the [examples page](examples.rst).

**Table 1**. *Software features included in the latest version of Synthia for different fitting and generation methods. For correlated multivariate variables stretching and uniformization is available for copula models (^). All models support the handling of continuous variables (¹). Discrete (²) and categorical (³) variables are supported with vine copulas. All correlated multivariate methods support empirical and parametric fitting of marginal distributions*

| Variable relation                      | Method     | Option             |
| -------------------------------------- | ---------- | ------------------ |
| Univariate or Multivariate Independent |            |                    |
|                                        | Fitting    |                    |
|                                        |            | Empirical          |
|                                        |            | Parameterized      |
|                                        | Generation |                    |
|                                        |            | Pseudo-random      |
|                                        |            | Quasi-random       |
|                                        |            | Stretching         |
|                                        |            | Uniformization     |
| Multivariate                           |            |                    |
|                                        | Fitting    |                    |
|                                        |            | ^¹Copula: Gaussian |
|                                        |            | ^¹²³Copula: Vine   |
|                                        |            | ¹fPCA              |
|                                        |            | Empirical          |
|                                        |            | Parameterized      |
|                                        | Generation |                    |
|                                        |            | Pseudo-random      |
|                                        |            | Quasi-random       |
|                                        |            | ^Stretching        |
|                                        |            | ^Uniformization    |
