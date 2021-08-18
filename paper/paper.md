---
title: 'Synthia: multidimensional synthetic data generation in Python'
tags:
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


# Summary and Statement of Need

Synthetic data -- artificially generated data that mimic the original (observed) data by preserving relationships between variables [@Nowok_2016] -- may be useful in several areas such as healthcare, finance, data science, and machine learning [@Dahmen_2019; @Kamthe_2021; @Nowok_2016; @Patki_2016]. As such, copula-based data generation models -- probabilistic models that allow for the statistical properties of observed data to be modelled in terms of individual behavior and (inter-)dependencies [@Joe_2014] -- have shown potential in finance, data science, and meteorology [@Kamthe_2021; @Li_2020; @Meyer_2021a; @Patki_2016]. Although copula-based data generation tools have been developed for tabular data -- e.g. the Synthetic Data Vault using Gaussian copulas and generative adversarial networks [@Patki_2016; @Xu_2018], or the Synthetic Data Generation via Gaussian Copula [@Li_2020] -- in computational sciences such as weather and climate, data often consist of large, labelled multidimensional datasets with complex dependencies.

Here we introduce Synthia, an open-source multidimensional synthetic data generator written in Python for xarray's [@Hoyer_2017] labelled arrays and datasets with support for parametric and vine copulas models and functional principal component analysis (fPCA) -- an extension of principal component analysis where data consist of functions instead of vectors [@Ramsay_2005] -- to allow for a wide range of data and dependent structures to be modelled. For efficiency, algorithms are implemented in NumPy [@Harris_2020] and SciPy [@SciPy_2020] for Gaussian (parametric) copula and fPCA classes and rely on the C++ library vinecopulib [@Nagler_2020a] through pyvinecopulib's [@Nagler_2020] bindings for fast computation of vines.

Recent applications of Synthia include the generation of dependent [@Meyer_2021a] and independent [@Meyer_2021b] samples for improving the prediction of machine learning emulators in weather and climate. In this release we include examples and tutorials for univariate and multivariate synthetic data generation using copula and fPCA methods and look forward to enabling the generation of synthetic data in various scientific communities and for several different applications.


# Acknowledgments

We thank Maik Riechert for his comments and contributions to the project.


# References
