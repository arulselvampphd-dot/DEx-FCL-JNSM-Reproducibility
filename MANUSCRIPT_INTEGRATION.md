# Manuscript integration guide

Use only the values produced in `tables/` after all five seeds finish. Do not copy values from console logs or a single seed.

## Section 5 — Experimental methodology

Report exactly:
- 10 clients; full participation in the primary controlled comparison.
- Dirichlet α = 0.3 primary non-IID setting; α ∈ {1.0, 0.5, 0.3, 0.1} sensitivity.
- Seeds: 7, 19, 31, 43, 59.
- 70/10/20 stratified global train/validation/test split for each seed.
- Data preprocessing fitted without using test-set statistics; z-score scaling is fitted on training data only.
- CICIoT2023 and Edge-IIoTset are evaluated independently; feature spaces are never merged.
- Zero-day protocol is leave-one-attack-family-out. Held-out-family samples are excluded from both training and validation. The rejection threshold is the 95th percentile of known-validation novelty scores.

## Section 6 — Results

Recommended table mapping:
1. Overall FL performance → `tables/table_overall.csv`
2. Non-IID sensitivity → `tables/table_noniid.csv`
3. Continual-learning summary → `tables/table_continual.csv`
4. Zero-day detector summary → `tables/table_zeroday_summary.csv`
5. Attack-family zero-day results → `tables/table_zeroday_attackwise.csv`
6. Statistical tests → `tables/table_statistics.csv`

Recommended figures:
- `figures/fig_overall_<dataset>.png`
- `figures/fig_noniid_<dataset>.png`
- `figures/fig_continual_<dataset>.png`
- `figures/fig_forgetting_<dataset>.png`
- `figures/fig_zeroday_<dataset>.png`

## Claims discipline

Only claim DEx-FCL superiority for a metric when:
1. the five-seed mean is better;
2. the effect is practically meaningful; and
3. the paired statistical comparison supports the claim.

If the p-value is not significant, write that the methods were statistically indistinguishable under the evaluated protocol rather than claiming superiority.
