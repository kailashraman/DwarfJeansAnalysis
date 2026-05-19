# Open issues

Central registry of known-open work. Detail and spec live in the topic
doc each entry links to; this file is the index.

**Conventions.** Numbered entries, `**Title.**` prefix, **Status:** /
**Where:** / **Next:** lines. Mark resolved items inline with
`RESOLVED YYYY-MM-DD` (strikethrough optional) and keep them for one
cycle before pruning. New issues append; do not renumber on close.
Pure inline `TODO`/`FIXME` comments in source code are out of scope —
list an issue here only when it warrants tracking beyond the call site.

When a topic doc previously hosted its own "Open issues" section,
replace that section with a one-line pointer to the relevant entry
here; do not duplicate the body.

---

## Active

1. **`danieli23_const` per-bin ESS pathology on compact-halo dwarfs.**
   - **Status:** Filed 2026-05-18. Empirically verified on `segue_1`
     (min per-bin ESS = 1.90), `tucana_5` (1.45), `willman_1` (4.78);
     `fattahi18`, `moster18`, `kim24` unaffected. Mechanism:
     reviewer + analyst found that the builder's global
     $\mathrm{ESS}/n_{\rm bins} \ge 30$ floor is satisfied while
     individual bins land far below it, because weights are
     heterogeneous within bins. Failure mode is μ jitter across bins
     (not σ blowup), with bins clustered in the compact-halo regime
     $\log_{10} c \in [-1.11, -0.74]$ and widths spanning $\sim$3
     decades; `np.interp` then linearly interpolates between
     unreliable bin centres.
   - **Where:** [`docs/plan/satgen_shmr.md`](plan/satgen_shmr.md)
     Steps 4–5 (binning construction and the global-floor caveat).
   - **Next:** Action items in priority order —
     - **(a)** `_load_satgen_table` in
       `src/dwarfjeans/jeans/priors.py`: emit a `warnings.warn` at
       load if `min(metadata['per_bin_ess']) < ~10`. Cheap,
       non-breaking diagnostic.
     - **(b)** Builder hard floor in
       `scripts/build_satgen_shmr_prior_tables.py`: after binning, if
       any bin falls below threshold (e.g. 5), either raise + skip the
       dwarf or coarsen `n_bins` until satisfied. Today only the
       global $\mathrm{ESS}/n_{\rm bins} \ge 30$ proxy is enforced.
     - **(c)** Boundary-clamp diagnostic in production: count fraction
       of prior-transform draws hitting `LOG10_RS_BOUNDS` /
       `LOG10_RHOS_BOUNDS` edges; surface in the per-dwarf summary.
       Edge-pileup is the observable symptom of bin-cluster pathology.
     - **(d)** Writeup prose-vs-code drift check in
       `docs/writeup/pipeline.tex` SHMR-weighted section: prose
       describes a per-bin Gaussian; code uses `np.interp` linear
       interpolation between bin centres for the conditional
       parameters (Gaussian only for the μ/σ nuisances themselves).
       Reconcile.
     - **(e)** Durable fix: replace ESS-quantile binning with LOESS /
       weighted local linear regression evaluated on a fixed
       $\log c$ grid. Eliminates bin clustering; `np.interp` then
       interacts predictably with a uniform grid. Parametrize old vs
       new and regression-test against `fattahi18` for parity before
       retiring the binned path.

     Recommended sequencing: ship (a)–(c) as one small diagnostic PR;
     (d) as its own writeup-only commit; defer (e) until (a)–(c)
     quantify how widespread the pathology is — may not be worth the
     rebuild if only `danieli23_const` is affected and the
     boundary-clamp diagnostic stays clean.

2. **Velocity gradients not modeled in the Jeans likelihood.**
   - **Status:** Not yet implemented. Affects classical dwarfs with
     measured gradients (Carina, Fornax, Sculptor) and Antlia II
     (Ji+2021 report a clear tidal-disruption gradient in the very
     catalog we ingest).
   - **Where:** [`docs/plan/stage2.md`](plan/stage2.md) § Velocity
     gradients.
   - **Next:** Three options under consideration there (ingest-stage
     subtraction; in-likelihood marginalization; document-and-skip for
     UFDs). Choice interacts with the perspective correction —
     residual gradient is the rotation/streaming component, must not
     be double-counted.

3. **Jeffreys-vs-loguniform halo-parameter bias regression.**
   - **Status:** Surfaced by the 15-realization UFD MC under the new
     Jeffreys default. `M(r_½, 3D)` picks up ~0.07 dex high-side bias
     and tighter-than-nominal z-dist (std(z)=0.79, cov68%=60%) vs.
     the clean loguniform recovery (bias <0.01 dex, std(z)≈1.1).
     Individual `log r_s`, `log ρ_s`, `log(ρ_s · r_s³)` biased by
     0.6–1.1 dex with KS p < 10⁻³.
   - **Where:** [`docs/plan/stage2.md`](plan/stage2.md) § Calibration
     status (table + paragraph following).
   - **Next:** Diagnose whether the Jeffreys term's shift toward
     higher ρ_s / lower r_s in the small-x NFW regime
     (`r_½/r_s ≈ 0.2`) is the full story; decide whether to keep
     Jeffreys default given the M(r_½) bias, or recalibrate.

4. **Population J/D MC pending under the Jeffreys prior.**
   - **Status:** `run_ufd_population.py` currently runs only the halo
     recovery; the per-realization J/D summary is no longer pushed
     inline since `run_jd_summary.py` was removed.
   - **Where:** [`docs/plan/stage3.md`](plan/stage3.md) (after the
     `analyze_asimov.py` paragraph).
   - **Next:** Reinstate an inline `J_D_factors` push inside the MC
     loop that writes `ufd_pop_jd_diagnostics.json`. Historical
     loguniform baselines are recorded there for reference.

5. **SatGen prior: 2D KDE form as a higher-moment upgrade.**
   - **Status:** Pooled-residual diagnostics (skew = +0.13, excess
     kurtosis = +0.17 across 2.4×10⁶ halos) say the per-bin
     lognormal-in-ρ_s | r_s approximation is faithful in the bulk.
     Logged as a follow-up only — not currently affecting results.
   - **Where:** [`docs/plan/stage2.md`](plan/stage2.md) §
     SatGen-conditioned prior.
   - **Next:** Revisit only if higher-moment structure starts to
     matter for a science target (no action otherwise).

6. **Leo VI presence in LVDB v1.0.5.**
   - **Status:** Discovery is very recent; key may not exist in the
     pinned LVDB version.
   - **Where:** [`docs/plan/data_sources.md`](plan/data_sources.md)
     Path B table (Leo VI row) + Items requiring on-cluster
     verification #6.
   - **Next:** Verify against `comb_all.csv`. If absent, either bump
     to a v1.1.0 LVDB pin or source from the discovery paper directly
     with a per-galaxy override.

---

## Recently closed

(none yet — when an issue resolves, move it here with `RESOLVED
YYYY-MM-DD` and a one-line note. Prune after the next sweep.)

---

## Pointers to resolved historical audits

- [`docs/plan/per_paper_combiners.md`](plan/per_paper_combiners.md) §
  Open issues (QA-sweep #5): all five items RESOLVED 2026-05-07.
  Retained as audit record; not migrated here.
