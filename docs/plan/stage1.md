# Stage 1 — Velocity-dispersion diagnostic and classification

A model-independent kinematic characterization of each galaxy, run before any halo modeling. Useful both as a diagnostic of the data itself and as the input to Stage 2's prior choices.

This document is the canonical Stage 1 reference. The high-level role in the pipeline is summarized in [`pipeline_overview.md`](./pipeline_overview.md); deviations from P&S 2018 are consolidated in the Differences section there.

---

## Likelihood

The Walker et al. 2006 constant-σ_los Gaussian likelihood. For a galaxy with N member stars, each carrying a measured line-of-sight velocity `V_i`, per-star measurement error `σ_ε,i`, and membership probability `p_i`:

```
ln L(V, σ_los) = ∑_{i=1}^{N}  p_i · {  −½ ln[2π (σ_los² + σ_ε,i²)]
                                       − (V_i − V)² / [2(σ_los² + σ_ε,i²)]  }
```

Structurally this is Stage 2's likelihood (P&S eq. 8) with `σ_los(R_i) ≡ σ_los` — the same code path with the spatial dependence collapsed to a scalar. Two free parameters: the systemic velocity `V` and the intrinsic line-of-sight velocity dispersion `σ_los`. All member stars are drawn from a single Gaussian centered at `V` with intrinsic width `σ_los`, observed with per-star measurement noise added in quadrature.

### Membership probabilities enter as log-likelihood weights

P&S 2018 eq. 8 carries `p_i` inside a *product* of per-star Gaussians, where it appears as a multiplicative prefactor `p_i / √[2π(...)]`. Multiplying each per-star factor by `p_i` rescales the likelihood by a global constant `∏ p_i` but does not modify the relative weighting of stars in the inference — the gradients and posterior shape are unchanged from `p_i ≡ 1`. We treat this as an error in the published equation: to make `p_i` actually enter the inference as a continuous weight, it must multiply terms in the *sum* of log-likelihood contributions, as written above. We adopt this corrected form as a deliberate departure from P&S 2018 — see the Differences from P&S 2018 section in `pipeline_overview.md`. In the limit where every surviving star has `p_i = 1` (which P&S enforces operationally by hard-cutting at ingest and setting survivors to `p_i = 1`), our likelihood reduces to theirs exactly. The corrected weighting matters only when membership probabilities are propagated as continuous values rather than collapsed to {0, 1}.

The sum runs over the post-membership-cut sample produced in Stage 0b. The membership cut itself is still a hard cut on the configured `p_i` threshold per galaxy (per P&S §2); the weighting acts on the surviving stars.

---

## Priors

- **`σ_los`:** log-uniform prior on `log_10 σ_los`, range `[-2, 2]`. I.e., uniform in `log_10(σ_los / km s⁻¹)` over `[−2, 2]`, equivalently `0.01 ≤ σ_los ≤ 100 km/s`. P&S 2018 §2 calls this a "Jeffreys prior" (after change of variables from the conventional Jeffreys form `p(σ) ∝ 1/σ` for a Gaussian scale parameter, the two are identical); we use the same wording as the methodology paper for cross-comparison. The log-uniform form is what enables the resolved/unresolved diagnostic: when the data fail to constrain σ_los, the posterior naturally extends toward small `log_10 σ_los`, producing the characteristic zero-tail or pure-upper-limit posteriors that drive the classification below.

- **`V`:** uniform prior centered on the inverse-variance-weighted mean of the post-selection velocities, `V̄_IVW = Σ V_i/σ_{ε,i}² / Σ 1/σ_{ε,i}²`, with half-width 10 km/s (`V̄_IVW − 10 ≤ V ≤ V̄_IVW + 10`). Centering on the data rather than on the LVDB `vlos_systemic` value avoids stale/sample-mismatch failures (Pegasus III hit the lower wall of a registry-centered prior in QA). The half-width is read from the optional registry column `vlos_prior_halfwidth_kms` if present (default 10), plumbed by `scripts/run_production.py` through to the prior builders in `src/dwarfjeans/jeans/priors.py`; add the column by hand to `data/registry/galaxies.ecsv` to widen for small samples or unusual velocity distributions (P&S 2018 footnote 6). Same convention as Stage 2.

---

## Sampler

dynesty, matching Stage 2 for consistency. Two-parameter problem with one log-uniform-prior dimension; `nlive=500` with `dlogz=0.1` is more than enough.

---

## Classification

Each galaxy is sorted into one of four categories based on the σ_los posterior shape. The discriminator is whether the posterior has a **distinct peak** in `log_10 σ_los` away from the prior boundary:

- **Resolved** — single distinct peak, no significant posterior weight at the lower prior boundary.
- **Small zero-tail** (~5% posterior at boundary) — distinct peak with a small tail extending to small σ_los; e.g., Leo V, Pegasus III, Pisces II.
- **Large zero-tail** (~40–60%) — distinct peak with a large tail; e.g., Draco II, Leo IV, Grus I.
- **Unresolved / upper limit** — no distinct peak; the posterior is monotone or rail-bound; e.g., Hydra II, Segue 2, Triangulum II, Tucana III.

The "distinct peak" criterion is operationalized via posterior-mode detection on the marginal `log_10 σ_los` distribution: a peak is distinct if it sits at least one prior unit (i.e., one decade in log σ_los) from the lower boundary and has more posterior mass than the boundary region. Edge cases (e.g., bimodal posteriors) are flagged for manual review.

---

## Outputs

- **Full posterior chain** on (V, σ_los) per galaxy, stored as a NumPy `npz` archive keyed by parameter name. Equal-weight samples after dynesty's `resample_equal`. Kept for diagnostic plots and Stage 2 sanity-checking.
- Classification flag, recorded in the registry. Feeds Stage 2 prior choices (esp. the expanded ρ_s prior range for unresolved galaxies) and Stage 4 reporting style (point + errors vs. upper limit).
- Median + 16/84% σ_los summary for cross-checking against P&S Table A1.

---

## Provenance

Each per-galaxy chain output and registry entry carries metadata sufficient for reproduction: LVDB version (v1.0.5), git commit of pipeline code at run time, dynesty version, RNG seed, run date. Recorded under a reserved `_meta` key in every `npz` archive (per-star catalogs *and* Stage 1 chain outputs) as a JSON-serialized dict — concretely, a 0-d `numpy` array of `dtype=object` holding the JSON string, accessed as `np.load(path)['_meta'].item()` and parsed with `json.loads`. The same fields are also broadcast into registry columns for query-friendly access; the `_meta` dict is the canonical record. This convention is used uniformly across Stage 0b (per-star catalogs), Stage 1 (chain archives), and Stage 2 (per-galaxy run dirs), so a single helper handles read/write everywhere.

---

## Validation

Validation runs in P&S-replication mode with the config flag `pi_weighting: replicate_ps2018`, which collapses surviving `p_i` to 1.0 before the likelihood evaluation, matching P&S's operational convention. The default production runs use the corrected `p_i`-weighted form. The two configurations agree exactly when every surviving star has `p_i = 1.0` to begin with (e.g. for source papers that report only binary member/non-member flags), so the flag is a no-op for most galaxies and only matters for the few with continuous-probability membership.

P&S's Table A1 σ_los values were themselves derived with the same Walker+06 likelihood and same log-uniform prior we use here, applied to the same membership-cut samples — so this validation tests **code correctness**, not science correctness. Specifically, it confirms that our Stage 0b ingestion (membership cuts, RR Lyrae / binary / foreground removals, projected-radius computation), our Stage 1 likelihood implementation, and our prior transform reproduce P&S's analysis pipeline. It does not separately validate the σ_los values against the underlying spectroscopy, since P&S does not re-derive them; for that, source-paper σ_los values (Walker+09, Mateo+08, Spencer+17, etc.) are the appropriate reference.

### Validation gate 1: Segue 1 (the launch blocker)

The first validation check, and the only one required to launch Stage 2, is reproducing P&S's Segue 1 results. Segue 1 is the most sensitive single-galaxy test of the full Stage 0b → Stage 1 chain because:

- Its inferred σ_los depends strongly on the membership-probability cut (`p_i > 0.8` per P&S, with the Simon et al. 2011 Bayesian probabilities). Mishandling the cut convention propagates directly to a wrong σ_los.
- It exercises the `p_i`-handling code path: in `replicate_ps2018` mode the surviving `p_i` are collapsed to 1.0 after the cut, so a bug in either the cut threshold *or* the post-cut `p_i` collapsing surfaces here.
- It also exercises Stage 0b's per-star ingestion (Simon+11 column conventions, `(R_i, V_i, σ_ε,i, p_i)` schema), the projected-radius computation from LVDB `(ra, dec)`, and the Walker+06 likelihood implementation in one shot.

Pass criteria for Segue 1:

- Median σ_los matches P&S 2018 Table A1 reference (3.1 +0.9/−0.8 km/s for Segue 1) within 5%.
- 16/84 percentile half-widths match P&S's reported asymmetric error bars within 15%.
- Posterior shape qualitatively matches P&S: distinct peak, small zero-tail (classification = "small zero-tail" or "resolved").

If Segue 1 passes, the membership-cut convention, the per-star ingestion, the likelihood code path, the log-uniform prior implementation, and the dynesty configuration are all essentially confirmed end-to-end. We treat this as sufficient to launch Stage 2.

### Validation gate 2: full Table A1 sweep (post-launch)

Once Segue 1 passes and Stage 2 is launched, we run the full P&S Table A1 reproduction across all confirmed MW dwarfs as a separate exercise. This sweep is **not** a Stage 2 launch blocker — it's a broader code-correctness audit run in parallel with (or after) Stage 2 begins producing J-factor posteriors. Per-galaxy disagreements found in this sweep are investigated individually; not every galaxy is required to match P&S to within tight tolerance, since a single-galaxy disagreement is more likely to indicate a per-galaxy ingestion subtlety than a pipeline-level bug (and pipeline-level bugs would already have been caught by Segue 1).

Reference tolerances for the sweep, used to triage which galaxies need follow-up rather than as a pass/fail gate:

- **Resolved galaxies:** median σ_los within 5% of P&S Table A1 median, 16/84 percentile half-widths within 15%.
- **Small zero-tail galaxies:** median σ_los of the non-tail portion within 10%; tail fraction within ±2 percentage points.
- **Large zero-tail galaxies:** median within 15%; tail fraction within ±5 percentage points.
- **Unresolved galaxies:** the 95.5%-percentile upper limit (after the `log_10 ρ_s > 4` cut, per P&S §4.1) within 0.1 dex of P&S's quoted upper limit.

Galaxies outside these tolerances are flagged in the sweep report for manual diagnostic. Common expected sources of disagreement (none of which are launch blockers): RNG-seed and dynesty-vs-MultiNest sampler variance, edge cases in the membership-cut convention not covered by Segue 1's `p_i > 0.8` rule, the per-galaxy modifications P&S applied to specific samples (e.g., the Grus I / Tucana II MW-background-modeling discussion in P&S §2 that gives different star counts than naive ingestion).
