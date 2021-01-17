---
title: 'Synthia: multidimensional synthetic data generation in Python'
tags:
  - synthetic-data
  - copula
  - fpca
  - machine-learning
  - data-science
  - python
authors:
  - name: David Meyer
    orcid: 0000-0002-7071-7547
    affiliation: "1, 2"
  - name: Thomas Nagler
    orcid: 0000-0003-1855-0046
    affiliation: 3
affiliations:
 - name: Department of Meteorology, University of Reading, Reading, UK
   index: 1
 - name: Department of Civil and Environmental Engineering, Imperial College London, London, UK
   index: 2
 - name: Mathematical Institute, Leiden University, Leiden, The Netherlands
   index: 3
date: 22 October 2020
bibliography: paper.bib
---


# Summary

Synthetic data are artificially generated data, not obtainable by direct measurements [@McGraw-Hill-2003]. To serve a similar purpose to real data, they need to preserve the statistical properties in terms of their individual behavior and (inter-)dependences [@Meyer2021]. Copula and functional Principle Component Analysis (fPCA) are statistical models that allow these properties to be simulated [@Joe_2014]. As such, copula generated data have shown potential for improving the generalization of machine learning (ML) emulators [@Meyer2021] or for anonymizing real-data datasets [@Patki_2016]. Although several synthetic data generation software exist [@Patki_2016; @xu2018synthesizing], to our knowledge, none offer a simple interface for working with multidimensional labelled datasets using copula and fPCA models.

Synthia is an open source Python package to model univariate and multivariate data, parameterize data using empirical and parametric methods, and manipulate marginal distributions. It is designed to enable scientists and practitioners to handle labelled multivariate data typical of computational sciences. Synthia supports three methods of multivariate data generation through fPCA, parametric (Gaussian) copula, and vine copula models for continuous (all), discrete (vine), and categorical (vine) variables. It has a simple and succinct API to natively handle xarray's [@hoyer2017xarray] labelled arrays and datasets. It uses a pure Python implementation for fPCA and Gaussian copula, and relies on the fast and well tested C++ library vinecopulib [@nagler_thomas_2020_4287554] through pyvinecopulib's [@nagler_thomas_2020_4288292] bindings for fast and efficient computation of vines.

Synthia has already been used to generate augmented datasets in @Meyer2021 for improving the predictions of a ML emulator. With the release of Synthia, we look forward to enabling the generation of synthetic data from various scientific communities and experts alike.

# Acknowledgments

We thank Maik Riechert for his comments and contributions to the project.


# References
