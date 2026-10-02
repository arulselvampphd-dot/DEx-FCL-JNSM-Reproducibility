# Source recovery evidence

Audit reference: main commit `f9c77256fa0bd4acd7825f4d55ecc52f57f8d81b`.
All live snapshot files, including binary figures, were matched to Git blob hashes
before editing. The authoritative paper comparison was Final JNSM manuscript.docx,
Section 5 and Tables 2–3, available on 2026-10-02.

## Archives examined

| Archive | SHA-256 | Finding |
| --- | --- | --- |
| DExFCL_JNSM_Reproducibility_Repository.zip | edb23ac338c5625707493cb5df84a8aabc4e3d18085edfc91d734cdbee941f59 | All 14 Python source modules identical to audited GitHub source |
| DExFCL_Real_Benchmark_Package_CLEAN.zip | 16a4addcc455cffcb9023df4326dd7661902d74e3afee2f3568744a5855fb473 | All 14 Python source modules identical to audited GitHub source |
| DEx_FCL_Real_Results_and_Figures.zip | 4f14b7e8eef848622b677cdd5ba10d22f1fad9b742844860287a84f7c03dd7db | Every contained CSV identical to audited GitHub results; no training source |

The corrected JSC rerun ZIP and Colab notebook were also examined. They add a
new ten-seed campaign, replay baselines and scalability measurements, and still
call the legacy DatasetBundle/stratified_split_scale preprocessing. They do not
contain the generating random_3way loader for the JNSM experiment.

## Conclusion and limits

The examined archives do not recover the source/configuration combination that
produced the documented JNSM protocol. This is a provenance gap, not evidence
that the archived measurements are synthetic. CSV/table consistency can be
checked independently, which is the scope of this repair.

Required next evidence: the original benchmark runner or notebook, dataset
construction manifests, exact raw/processed dataset hashes, resolved settings
and run environment. If that evidence is unavailable, a new campaign must be
labelled as a rerun and the paper updated to its actual outputs.
