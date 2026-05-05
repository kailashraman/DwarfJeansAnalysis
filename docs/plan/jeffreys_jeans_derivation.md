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

The Walker+2006 likelihood for $N$ stars with velocities $V_i$, errors $\varepsilon_i$, projected radii $R_i$, and membership probabilities $p_i$ is

$$\mathcal{L}(\bar V, \rho_s, r_s, \beta) = \prod_i \frac{p_i}{\sqrt{2\pi s_i^2}}\exp\!\left[-\frac{(V_i - \bar V)^2}{2 s_i^2}\right], \qquad s_i^2 \equiv \sigma_{\rm los}^2(R_i; \rho_s, r_s, \beta) + \varepsilon_i^2.$$

We seek the Jeffreys prior $p(\theta) \propto \sqrt{\det \mathcal{I}(\theta)}$, where $\mathcal{I}$ is the Fisher information matrix.

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

The score for star $i$ with respect to a halo parameter $\theta_j \in \{\ln\rho_s, \ln r_s\}$ is

$$\frac{\partial \ln\mathcal{L}_i}{\partial \theta_j} = -\frac{1}{2 s_i^2}\frac{\partial \sigma_{\rm los}^2}{\partial \theta_j} + \frac{(V_i - \bar V)^2}{2 s_i^4}\frac{\partial \sigma_{\rm los}^2}{\partial \theta_j}$$

(plus a $\bar V$ score that decouples from the halo parameters under expectation). Taking the second derivative and using $\mathbb{E}[(V_i - \bar V)^2] = s_i^2$:

$$\mathcal{I}_{jk}^{(i)} = \frac{1}{2 s_i^4}\,\frac{\partial \sigma_{\rm los}^2}{\partial \theta_j}\bigg|_{R_i}\,\frac{\partial \sigma_{\rm los}^2}{\partial \theta_k}\bigg|_{R_i}.$$

Define $A_i \equiv \sigma_{\rm los}^2(R_i)$, $T_i \equiv T(R_i; r_s, \beta)$, and the dimensionless weight

$$\tilde w_i \equiv \frac{A_i^2}{s_i^4} = \frac{A_i^2}{(A_i + \varepsilon_i^2)^2}.$$

The 2×2 Fisher matrix in $(\ln\rho_s, \ln r_s)$ at fixed $\beta$ is

$$\mathcal{I}_{2\times 2} = \frac{1}{2}\sum_i \tilde w_i \begin{pmatrix} 1 & T_i \\ T_i & T_i^2 \end{pmatrix}.$$

The determinant:

$$\det \mathcal{I}_{2\times 2} = \frac{1}{4}\left[\left(\sum_i \tilde w_i\right)\left(\sum_i \tilde w_i T_i^2\right) - \left(\sum_i \tilde w_i T_i\right)^2\right].$$

By the Cauchy–Schwarz inequality this is $\geq 0$, with equality iff all $T_i$ are equal — i.e., all stars probe the same dimensionless radius $R_i/r_s$.

---

## Step 5: the derived Jeffreys prior

In compact form, with $\mathrm{Var}_{\tilde w}(T)$ denoting the weighted variance of the $T_i$ across stars:

$$\boxed{p(\ln \rho_s, \ln r_s \mid \beta) \propto \sqrt{\det \mathcal{I}_{2\times 2}} = \frac{1}{2}\left(\sum_i \tilde w_i\right)\sqrt{\mathrm{Var}_{\tilde w}(T)}.}$$

This is the **conditional Jeffreys prior** on the halo parameters at fixed $\beta$. When combined with an independent prior on $\beta$ (e.g., a physically motivated prior from cosmological simulations), the joint prior is

$$p(\ln \rho_s, \ln r_s, \beta) = p(\ln \rho_s, \ln r_s \mid \beta)\, p(\beta).$$

---

## Limits and interpretation

### Resolved-dispersion limit ($A_i \gg \varepsilon_i^2$)

Then $s_i^2 \to A_i$, so $\tilde w_i = A_i^2/s_i^4 \to 1$, independent of both $\rho_s$ and $r_s$. The prior reduces to

$$p(\ln\rho_s, \ln r_s \mid \beta) \propto \sqrt{N\sum_i T_i^2 - \left(\sum_i T_i\right)^2} = N\,\sqrt{\mathrm{Var}(T_i)}.$$

This is **independent of $\rho_s$** — confirming that the log-flat prior on $\rho_s$ is exactly correct in the resolved regime. The prior on $r_s$ in the same regime is $\propto \sqrt{\mathrm{Var}(T(R_i/r_s; \beta))}$, which is approximately log-flat over a wide range of $r_s$ but acquires data-dependent corrections.

### Unresolved-dispersion limit ($A_i \ll \varepsilon_i^2$)

Then $s_i^2 \to \varepsilon_i^2$ and $\tilde w_i \to A_i^2/\varepsilon_i^4 \propto \rho_s^2$. The overall scale of the prior grows as $\rho_s^2$, but expressed as a density in $\ln\rho_s$ this becomes $p(\ln\rho_s) \propto \rho_s^2$ — i.e., **strongly suppressed at small $\rho_s$**. This is the regime where the formal Jeffreys prior differs most from log-flat: log-flat permits arbitrarily small $\rho_s$ while Jeffreys correctly recognizes that the data have no information there.

### Identifiability requirement

The prior is well-defined only when $\mathrm{Var}_{\tilde w}(T_i) > 0$. This requires that the stars sample a range of $R_i/r_s$ values — exactly the condition for $r_s$ to be identifiable from the data. For a tracer population that all sits at one effective radius, $r_s$ cannot be inferred and the prior correctly degenerates.

---

## Practical recipe

For each likelihood evaluation in MCMC:

1. **Standard Jeans solve.** Solve for $\sigma_{\rm los}^2(R_i)$ at each star using the constant-$\beta$ Jeans equation — i.e. with the $(r'/r)^{2\beta}$ integrating factor in the inner integral. This gives you $A_i$ and (with the $\rho_s r_s^3$ prefactor stripped) $\mathcal{P}_i \equiv \mathcal{P}(R_i; r_s, \beta)$.

2. **Auxiliary integral.** Repeat the Jeans solve with the same $(r'/r)^{2\beta}$ integrating factor but with the inner integrand $g(x) \to x^2/(1+x)^2$ instead of $g(x)$. Project with the same anisotropy kernel $(1 - \beta R^2/r^2)$. This gives $\mathcal{Q}_i \equiv \mathcal{Q}(R_i; r_s, \beta)$.

3. **Shape factor.** Compute $T_i = 3 - \mathcal{Q}_i / \mathcal{P}_i$.

4. **Weights.** Compute $\tilde w_i = A_i^2 / (A_i + \varepsilon_i^2)^2$.

5. **Determinant.** Form

   $$D = \left(\sum_i \tilde w_i\right)\left(\sum_i \tilde w_i T_i^2\right) - \left(\sum_i \tilde w_i T_i\right)^2$$

   and add $\tfrac{1}{2}\ln D$ to the log-prior. (The constant $-\ln 2$ from the $\sqrt{1/4}$ factor is irrelevant for sampling.)

**Cost:** one extra Jeans-style integral per likelihood call (~2× slowdown), with no finite-difference noise.

---

## Notes on validity

- This derivation assumes the Walker+2006 Gaussian-likelihood form. If the analysis uses a different likelihood (e.g., a generalized profile likelihood, or a Bayesian likelihood with binned dispersions), the Fisher information must be recomputed.

- The expectation $\mathbb{E}[(V_i - \bar V)^2] = s_i^2$ used in the Fisher calculation assumes the model is correctly specified at the true parameters. The Jeffreys prior is therefore evaluated using the model's predicted $s_i^2$, not the empirical residuals — this is a feature, not a bug, since the Fisher information is a property of the model, not of any single dataset realization.

- The prior depends on the data through $\{R_i, \varepsilon_i\}$ (positions and measurement uncertainties), but **not** through $\{V_i\}$ (the velocity values themselves). This is why the formal Jeffreys prior is sometimes called "data-dependent" but is still a valid prior in the operational sense — it is constructed before the velocity data are inspected.

- For a strict Bayesian, this data-dependent construction is uncomfortable but defensible as a "reference prior" in the Bernardo–Berger sense: it is the prior that maximizes the expected information gain from the experiment.
