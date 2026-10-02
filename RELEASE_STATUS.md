# Reproducibility release status — 2026-10-02

This first repair proposal establishes numerical reproduction from the archived
JNSM CSVs. It does not establish the provenance of their model-training runs.
The historical source/result files remain available for inspection.

## Fixed in this proposal

- Canonical archived CSV selection, schema validation and complete seed checks.
- Numerical regeneration of scientific summaries and paired statistical tests.
- All 11 historical figure names exported as 600-dpi PNG, PDF and EPS, without internal titles.
- Separation of archived and per-seed campaign aggregation; no duplicate ablation pooling.
- Edge 10% sampler validation, repeatability checks and source/output hash sidecars.
- Updated tracked-file checksums without a self-checksum.
- Automated tests and a numerical-export CI workflow.
- README distinguishes archived numerical reproduction from unverified training.

## Blocking a clean end-to-end release

| Area | Paper protocol | Current legacy pipeline | Needed evidence/change |
| --- | --- | --- | --- |
| CIC input | random_3way Parquet; supplied splits | part-*.csv; sample then split | Recover generating loader or implement/validate supplied-split loader |
| Edge preprocessing | seed-42 10% subset; fixed 70:10:20 | Original DNN CSV; sample before model-seed split | Integrate subset; preserve fixed evaluation distributions |
| Imputation | Training medians | Zero filling | Training-only medians with persisted feature/scaler manifest |
| Training-only caps | CIC 8,000/family; Edge 10,000/family | Global cap before splitting | Apply caps only to train and assert reported row counts |
| Primary model | 5 clients; 10 rounds; 2 epochs; batch 4096; hidden/latent 32/16; dropout .15; AdamW LR .003 | 10 clients; 30 rounds; 3 epochs; batch 512; 128/64; .25; LR .0005 | Recover exact resolved configs and generating commit; a fresh configuration is not historical provenance |
| SCAFFOLD | SGD LR .1 | Shared LR .0005 | Separate method optimizer settings |
| Partitioning | Shared class proportions for train/validation | Independently drawn train/validation allocations | Persist shared proportions, split IDs and partition hashes |
| Continual budget | 3 rounds/task | 8 rounds/task | Recover exact task protocol/Fisher state and validate retention results |
| Drift | Benign/normal to DDoS/DoS_DDoS; windows 100 | Mirai/Malware; windows 256 | Implement documented transition/calibration; preserve undetected delays |
| Edge assimilation | Information_Gathering | Injection | Match held-out support protocol |
| Ablation | Remove adaptive/drift/retention/explanation components | Incremental continual variants | Implement the reported removals and validate canonical outputs |
| Provenance | Dataset hashes, generating source, environment and resolved configs | Not recorded with archived runs | Recover original evidence; otherwise rerun and update paper |
| License | Explicit reuse terms | No LICENSE | Author must choose code reuse terms |

The JSC overlay uses a new ten-seed/replay/scalability design and depends on the
legacy processed NPZ pipeline. It cannot be presented as the recovered JNSM
generator. No full model-training run has been performed for this proposal.

The archived PNGs contain approximately 220-dpi metadata. Their numerical
content can now be redrawn at 600 dpi; this is a new export, not resampling of
those images. Historical raw timing is missing for zero-day runs. These gaps
are reported rather than filled with fabricated measurements.

## Acceptance before a new release

1. Recover and verify the generating source, or complete a newly documented campaign.
2. Assert dataset feature counts, fixed splits, train-only caps and no overlap/leakage.
3. Persist exact model/partition seeds, resolved configs, environment and source/data hashes.
4. Reconcile every scientific output with the manuscript; inspect any changed conclusions.
5. Choose licensing and verify DOI/citation details, then regenerate checksums.
6. Approve merge and publication as separate final actions.
