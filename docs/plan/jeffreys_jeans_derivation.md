# Jeffreys Prior for the NFW Jeans Likelihood

## Setup

For an NFW dark matter halo with a Plummer stellar tracer, the line-of-sight velocity dispersion at projected radius $R$ is obtained from the spherical Jeans equation. For a constant velocity anisotropy $\beta$, the equation
$$\frac{1}{\nu}\frac{d(\nu\sigma_r^2)}{dr} + \frac{2\beta\,\sigma_r^2}{r} = -\frac{GM(r)}{r^2}$$
admits the integrating factor $r^{2\beta}$ and solves to

$$\nu(r)\,\sigma_r^2(r) = G \int_r^\infty \left(\frac{r'}{r}\right)^{2\beta}\frac{\nu(r')\, M(r')}{r'^2}\, dr'$$

with NFW enclosed mass

$$M(r) = 4\pi \rho_s r_s^3\, g(x), \qquad x \equiv r/r_s, \qquad g(x) = \ln(1+x) - \frac{x}{1+x}.$$

Projecting (with constant anisotropy $\beta$):

$$\Sigma(R)\,\sigma_{\rm los}^2(R) = 2 \int_R^\infty \left(1 - \beta\frac{R^2}{r^2}\right)\frac{\nu(r)\,\sigma_r^2(r)\, r}{\sqrt{r^2 - R^2}}\, dr.$$

The Walker+2006 pseudo-likelihood for $N$ stars with velocities $V_i$, errors $\varepsilon_i$, projected radii $R_i$, and membership probabilities $p_i$ enters the pipeline as a **sum of weighted log-likelihoods** — membership probability is an *exponent*, not a multiplicative prefactor:

$$\ln \mathcal{L}(\bar V, \rho_s, r_s, \beta) = \sum_i p_i\,\ln\mathcal{N}\!\left(V_i\,;\,\bar V,\,s_i^2\right) = \sum_i p_i\!\left[-\tfrac12\ln(2\pi s_i^2) - \frac{(V_i-\bar V)^2}{2 s_i^2}\right],$$
$$s_i^2 \equiv \sigma_{\rm los}^2(R_i; \rho_s, r_s, \beta) + \varepsilon_i^2.$$

This is the form actually evaluated in the code (`src/dwarfjeans/jeans/inference.py:187, 232`: `ll = float(np.sum(p * ln_li))`). It downweights low-probability members but is *not* a normalized probability density on the data — it is a pseudo-likelihood. We treat it as the score-generating object and define Fisher information by $\mathcal{I}_{jk} \equiv -\mathbb{E}[\partial^2_{\theta_j\theta_k}\ln\mathcal{L}]$ at the model's predicted moments; this is the standard "Jeffreys for a pseudo-likelihood" construction.

We seek the Jeffreys prior $p(\theta) \propto \sqrt{\det \mathcal{I}(\theta)}$.

---

## Step 1: factor out $\rho_s$ exactly

Substituting the NFW mass into the inner integral:

$$\nu(r)\,\sigma_r^2(r) = 4\pi G \rho_s r_s^3 \int_r^\infty \left(\frac{r'}{r}\right)^{2\beta}\frac{\nu(r')\, g(r'/r_s)}{r'^2}\, dr' \equiv 4\pi G \rho_s r_s^3\, \mathcal{J}(r; r_s, \beta).$$

Define

$$\mathcal{P}(R; r_s, \beta) \equiv \int_R^\infty \left(1 - \beta\frac{R^2}{r^2}\right)\frac{\mathcal{J}(r; r_s, \beta)\, r}{\sqrt{r^2 - R^2}}\, dr.$$

Then

$$\boxed{\sigma_{\rm los}^2(R; \rho_s, r_s, \beta) = \frac{8\pi G \rho_s r_s^3}{\Sigma(R)} \cdot \mathcal{P}(R; r_s, \beta)}$$

so $\sigma_{\rm los}^2$ is **linear in $\rho_s$** at fixed $(r_s, \beta)$. This is the key structural fact. Note that $\beta$ enters $\mathcal{P}$ in two places — through the integrating factor $(r'/r)^{2\beta}$ in $\mathcal{J}$ and through the projection kernel $(1-\beta R^2/r^2)$ — but $\rho_s$ does not enter at all.

---

## Step 2: derivative with respect to $\ln \rho_s$

From the linearity:

$$\frac{\partial \sigma_{\rm los}^2}{\partial \ln \rho_s} = \sigma_{\rm los}^2.$$

---

## Step 3: derivative with respect to $\ln r_s$

The scale radius enters in three places: the prefactor $r_s^3$, the inner integrand through $g(r'/r_s)$, and the outer integrand through $\mathcal{J}(r; r_s, \beta)$. The integrating factor $(r'/r)^{2\beta}$ is independent of $r_s$, so it just rides along through the derivative.

For $g$ at fixed $r'$ with $x' = r'/r_s$:

$$\frac{\partial g(x')}{\partial \ln r_s} = g'(x') \cdot \frac{\partial x'}{\partial \ln r_s} = \frac{x'}{(1+x')^2} \cdot (-x') = -\frac{x'^2}{(1+x')^2}.$$

So

$$\frac{\partial \mathcal{J}(r; r_s, \beta)}{\partial \ln r_s} = -\int_r^\infty \left(\frac{r'}{r}\right)^{2\beta}\frac{\nu(r')}{r'^2}\,\frac{x'^2}{(1+x')^2}\, dr' \equiv -\mathcal{H}(r; r_s, \beta).$$

Define the projected version:

$$\mathcal{Q}(R; r_s, \beta) \equiv \int_R^\infty \left(1 - \beta\frac{R^2}{r^2}\right)\frac{\mathcal{H}(r; r_s, \beta)\, r}{\sqrt{r^2 - R^2}}\, dr.$$

Applying the chain rule to $\sigma_{\rm los}^2 = (8\pi G \rho_s r_s^3 / \Sigma) \cdot \mathcal{P}$:

$$\frac{\partial \sigma_{\rm los}^2}{\partial \ln r_s} = 3\sigma_{\rm los}^2 + \frac{8\pi G \rho_s r_s^3}{\Sigma} \cdot \frac{\partial \mathcal{P}}{\partial \ln r_s} = 3\sigma_{\rm los}^2 - \sigma_{\rm los}^2 \cdot \frac{\mathcal{Q}}{\mathcal{P}}.$$

Therefore

$$\boxed{\frac{\partial \sigma_{\rm los}^2}{\partial \ln r_s} = \sigma_{\rm los}^2\, T(R; r_s, \beta), \qquad T \equiv 3 - \frac{\mathcal{Q}(R; r_s, \beta)}{\mathcal{P}(R; r_s, \beta)}.}$$

The shape factor $T$ depends only on $R/r_s$ and $\beta$ — not on $\rho_s$.

---

## Step 4: Fisher information for $(\ln\rho_s, \ln r_s)$ at fixed $\beta$

The pseudo-likelihood is the $p_i$-weighted sum of the per-star Gaussian log-densities (call them $\ell_i$). The score with respect to a halo parameter $\theta_j \in \{\ln\rho_s, \ln r_s\}$ is

$$\frac{\partial \ln\mathcal{L}}{\partial \theta_j} = \sum_i p_i \,\frac{\partial \ell_i}{\partial \theta_j}, \qquad \frac{\partial \ell_i}{\partial \theta_j} = -\frac{1}{2 s_i^2}\frac{\partial \sigma_{\rm los}^2}{\partial \theta_j} + \frac{(V_i - \bar V)^2}{2 s_i^4}\frac{\partial \sigma_{\rm los}^2}{\partial \theta_j}$$

(plus a $\bar V$ score that decouples from the halo parameters under expectation). The Hessian inherits the same per-star weighting; using $\mathbb{E}[(V_i - \bar V)^2] = s_i^2$,

$$\mathcal{I}_{jk}^{(i)} = -\,\mathbb{E}\!\left[p_i\,\frac{\partial^2 \ell_i}{\partial\theta_j\partial\theta_k}\right] = \frac{p_i}{2 s_i^4}\,\frac{\partial \sigma_{\rm los}^2}{\partial \theta_j}\bigg|_{R_i}\,\frac{\partial \sigma_{\rm los}^2}{\partial \theta_k}\bigg|_{R_i}.$$

Define $A_i \equiv \sigma_{\rm los}^2(R_i)$, $T_i \equiv T(R_i; r_s, \beta)$, and the dimensionless weight

$$\tilde w_i \equiv \frac{A_i^2}{s_i^4} = \frac{A_i^2}{(A_i + \varepsilon_i^2)^2}.$$

The 2×2 Fisher matrix in $(\ln\rho_s, \ln r_s)$ at fixed $\beta$ is

$$\mathcal{I}_{2\times 2} = \frac{1}{2}\sum_i p_i\,\tilde w_i \begin{pmatrix} 1 & T_i \\ T_i & T_i^2 \end{pmatrix}.$$

The determinant:

$$\det \mathcal{I}_{2\times 2} = \frac{1}{4}\left[\left(\sum_i p_i\,\tilde w_i\right)\left(\sum_i p_i\,\tilde w_i\, T_i^2\right) - \left(\sum_i p_i\,\tilde w_i\, T_i\right)^2\right].$$

By the Cauchy–Schwarz inequality this is $\geq 0$, with equality iff all $T_i$ are equal — i.e., all stars probe the same dimensionless radius $R_i/r_s$.

---

## Step 5: the derived Jeffreys prior

In compact form, with $\mathrm{Var}_{p\tilde w}(T)$ denoting the $p_i\tilde w_i$-weighted variance of the $T_i$ across stars and $S_0 \equiv \sum_i p_i\tilde w_i$:

$$\boxed{p(\ln \rho_s, \ln r_s \mid \beta) \propto \sqrt{\det \mathcal{I}_{2\times 2}} = \frac{1}{2}\,S_0\,\sqrt{\mathrm{Var}_{p\tilde w}(T)}.}$$

This is the **conditional Jeffreys prior** on the halo parameters at fixed $\beta$. When combined with an independent prior on $\beta$ (e.g., a physically motivated prior from cosmological simulations), the joint prior is

$$p(\ln \rho_s, \ln r_s, \beta) = p(\ln \rho_s, \ln r_s \mid \beta)\, p(\beta).$$

---

## Limits and interpretation

### Resolved-dispersion limit ($A_i \gg \varepsilon_i^2$)

Then $s_i^2 \to A_i$, so $\tilde w_i = A_i^2/s_i^4 \to 1$, independent of both $\rho_s$ and $r_s$. With $N_{\rm eff} \equiv \sum_i p_i$ the membership-weighted star count, the prior reduces to

$$p(\ln\rho_s, \ln r_s \mid \beta) \propto \sqrt{N_{\rm eff}\sum_i p_i T_i^2 - \left(\sum_i p_i T_i\right)^2} = N_{\rm eff}\,\sqrt{\mathrm{Var}_p(T_i)}.$$

This is **independent of $\rho_s$** — confirming that the log-flat prior on $\rho_s$ is exactly correct in the resolved regime. The prior on $r_s$ in the same regime is $\propto \sqrt{\mathrm{Var}_p(T(R_i/r_s; \beta))}$, which is approximately log-flat over a wide range of $r_s$ but acquires data-dependent corrections.

### Unresolved-dispersion limit ($A_i \ll \varepsilon_i^2$)

Then $s_i^2 \to \varepsilon_i^2$ and $\tilde w_i \to A_i^2/\varepsilon_i^4 \propto \rho_s^2$ (the $p_i$ factor ride along but does not change the $\rho_s$ scaling). The overall scale of the prior grows as $\rho_s^2$, but expressed as a density in $\ln\rho_s$ this becomes $p(\ln\rho_s) \propto \rho_s^2$ — i.e., **strongly suppressed at small $\rho_s$**. This is the regime where the formal Jeffreys prior differs most from log-flat: log-flat permits arbitrarily small $\rho_s$ while Jeffreys correctly recognizes that the data have no information there.

### Identifiability requirement

The prior is well-defined only when $\mathrm{Var}_{p\tilde w}(T_i) > 0$. This requires that the stars sample a range of $R_i/r_s$ values — exactly the condition for $r_s$ to be identifiable from the data. For a tracer population that all sits at one effective radius, $r_s$ cannot be inferred and the prior correctly degenerates.

---

## Practical recipe

For each likelihood evaluation in MCMC:

1. **Standard Jeans solve.** Solve for $\sigma_{\rm los}^2(R_i)$ at each star using the constant-$\beta$ Jeans equation — i.e. with the $(r'/r)^{2\beta}$ integrating factor in the inner integral. This gives you $A_i$ and (with the $\rho_s r_s^3$ prefactor stripped) $\mathcal{P}_i \equiv \mathcal{P}(R_i; r_s, \beta)$.

2. **Auxiliary integral.** Repeat the Jeans solve with the same $(r'/r)^{2\beta}$ integrating factor but with the inner integrand $g(x) \to x^2/(1+x)^2$ instead of $g(x)$. Project with the same anisotropy kernel $(1 - \beta R^2/r^2)$. This gives $\mathcal{Q}_i \equiv \mathcal{Q}(R_i; r_s, \beta)$.

3. **Shape factor.** Compute $T_i = 3 - \mathcal{Q}_i / \mathcal{P}_i$.

4. **Weights.** Compute $\tilde w_i = A_i^2 / (A_i + \varepsilon_i^2)^2$ and the membership-multiplied weight $p_i \tilde w_i$ (the latter is what enters every sum below).

5. **Determinant.** Form

   $$D = \left(\sum_i p_i\tilde w_i\right)\left(\sum_i p_i\tilde w_i\, T_i^2\right) - \left(\sum_i p_i\tilde w_i\, T_i\right)^2$$

   and add $\tfrac{1}{2}\ln D$ to the log-prior. (The constant $-\ln 2$ from the $\sqrt{1/4}$ factor is irrelevant for sampling.) The numerically stable variance form actually used in the code is
   $D = S_0\,\sum_i p_i\tilde w_i (T_i - \bar T)^2$ with $S_0 = \sum_i p_i\tilde w_i$ and $\bar T = (\sum_i p_i\tilde w_i T_i)/S_0$ — algebraically identical, but free of the catastrophic cancellation that occurs as $\mathrm{Var}_{p\tilde w}(T)\to 0$ at the identifiability boundary.

**Cost:** one extra Jeans-style integral per likelihood call (~2× slowdown), with no finite-difference noise.

---

## Notes on validity

- This derivation assumes the Walker+2006 pseudo-likelihood form $\ln\mathcal{L} = \sum_i p_i \ln \mathcal{N}(V_i; \bar V, s_i^2)$ with membership entering as an exponent on each per-star Gaussian density. The $p_i$ then propagate linearly into the Fisher sums via $-\mathbb{E}[\partial^2 \ln\mathcal{L}]$. If the analysis uses a different likelihood (e.g., the alternative product-with-prefactor form $\prod_i p_i \mathcal{N}_i$, where the $\ln p_i$ pieces are constants and drop out so $p_i$ never enters Fisher; or a generalized profile likelihood, or a Bayesian likelihood with binned dispersions), the Fisher information must be recomputed.

- Because the pseudo-likelihood is not a normalized density on the data, the resulting Fisher matrix is technically a "sandwich-like" object rather than the formal Fisher of a probability model. This is the standard interpretation under which Jeffreys priors are constructed for weighted/composite likelihoods and is what the rest of the pipeline assumes.

- The expectation $\mathbb{E}[(V_i - \bar V)^2] = s_i^2$ used in the Fisher calculation assumes the model is correctly specified at the true parameters. The Jeffreys prior is therefore evaluated using the model's predicted $s_i^2$, not the empirical residuals — this is a feature, not a bug, since the Fisher information is a property of the model, not of any single dataset realization.

- The prior depends on the data through $\{R_i, \varepsilon_i\}$ (positions and measurement uncertainties), but **not** through $\{V_i\}$ (the velocity values themselves). This is why the formal Jeffreys prior is sometimes called "data-dependent" but is still a valid prior in the operational sense — it is constructed before the velocity data are inspected.

- For a strict Bayesian, this data-dependent construction is uncomfortable but defensible as a "reference prior" in the Bernardo–Berger sense: it is the prior that maximizes the expected information gain from the experiment.
