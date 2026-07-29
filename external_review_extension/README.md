# Gate-placement and DoLP-sensitivity extension

This snapshot contains the frozen gate-placement ablation code and run identifiers, matched M6 subset calculation, synthetic DoLP translation/rotation and quality-stress code, fixed perturbation logic, T1/T2/T3 threshold tables, complete 45-row quality summary, Table S14-S16 source data, and Figure 5/7/S5 source scripts and exports.

No new training was run to create this archive. Checkpoints, datasets, predictions, and caches are intentionally excluded. The experiment plan records the frozen run scope. Synthetic perturbations are digital stress tests and are not hardware specifications.

All public tables use repository-relative or logical manifest references. Set `PFE_SOURCE_PROJECT_ROOT` and `PFE_EVIDENCE_ROOT` only when rerunning the scripts against separately stored source artifacts. Model identifiers and checksums are retained so that the released evidence remains traceable without exposing machine-specific paths.
