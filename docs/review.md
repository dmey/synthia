# Review

TODO: some background statistical information/review of PDF/CDF and other terms used in `copula.md` and `fpca.md`?

Note: I render files with the [MyST parser](https://myst-parser.readthedocs.io/en/latest/examples/wealth_dynamics_md.html) as it makes it easier to use standard LaTeX syntax. E.g. (from [their example](https://myst-parser.readthedocs.io/en/latest/examples/wealth_dynamics_md.html) -- note that [labels are not aligned correctly in the readthedocs theme](https://github.com/readthedocs/sphinx_rtd_theme/pull/383) but we can fix this later/use another theme in the case all eqs needs to be labelled):


## Example: Citations/Reference

I use [sphinxcontrib-bibtex](https://sphinxcontrib-bibtex.readthedocs.io/en/latest/quickstart.html) to render references. This can be used to cite with something like:

For a review, see, for example,
    {cite}`deisenroth_faisal_ong_2020`.

References are in [`references.bib`](references.bib)

## Example syntax in Markdown with MyST: A Model of Wealth Dynamics

Having discussed inequality measures, let us now turn to wealth
dynamics.

The model we will study is

$$
w_{t+1} = (1 + r_{t+1}) s(w_t) + y_{t+1}
$$

where

- $w_t$ is wealth at time $t$ for a given household,
- $r_t$ is the rate of return of financial assets,
- $y_t$ is current non-financial (e.g., labor) income and
- $s(w_t)$ is current wealth net of consumption

Letting $\{z_t\}$ be a correlated state process of the form

$$z_{t+1} = a z_t + b + \sigma_z \epsilon_{t+1}$$

we'll assume that

$$R_t := 1 + r_t = c_r \exp(z_t) + \exp(\mu_r + \sigma_r \xi_t)$$

and

$$y_t = c_y \exp(z_t) + \exp(\mu_y + \sigma_y \zeta_t)$$

Here $\{ (\epsilon_t, \xi_t, \zeta_t) \}$ is IID and standard normal in
$\mathbb R^3$.

(md:sav_ah)=

```{math}
---
label: md:sav_ah
---
s(w) = s_0 w \cdot \mathbb 1\{w \geq \hat w\}
```


where $s_0$ is a positive constant.


```{bibliography} references.bib
:style: alpha
```