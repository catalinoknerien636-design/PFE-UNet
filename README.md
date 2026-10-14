# PFE-UNet: corrected JEI source-inspection release

This is the prepared code companion for the three-author JEI manuscript, "Camouflaged object segmentation with bounded feature modulation conditioned on degree of linear polarization". It is distinct from the historical public PFE-UNet v1.0.0 repository.

The author confirmed rights and authorized public publication of this specific code/configuration candidate on2026-10-15. This independent version covers corrected-study source inspection; the historical main/v1.0.0 release remains separate. Raw datasets, weights, pixel probabilities and the complete numerical records are not part of this public code release. Read ACCESS_AND_LICENSE_SCOPE.md.

## Contents

- review_source/: 109 byte-identical Python reading views from the authorized sanitized source archive; six of these already contain recorded host-path substitutions.
- review_config_views/: 58 frozen, path-sanitized configuration wrappers: 28 PCOD and30 RGBP runs. Their historical training flags are evidence only.
- review_split_membership/: six metadata-only split CSVs; no dataset pixels.
- numeric_replay/: two unchanged standard-library scripts from the separately authorized numerical archive.
- review_environment/: recorded version/wheel metadata. This does not constitute a portable runtime.
- third_party_license_texts/: four unchanged, scope-limited notices.
- CONFIGURATION_INDEX.csv and PROVENANCE_MANIFEST.json: direct entrypoints and per-file byte identity.

The source/configuration payload is preserved exactly. Preparation did not train models, run inference, import model modules, alter metrics or run the experiment queue. No images, masks, weights, probability arrays or vendor/native distributions are included.

## Inspection entrypoints

Read SOURCE_ENTRYPOINTS.md for architecture, half-pixel interpolation, training/loss/sampling, full fixed-population evaluation, device checks and PCOD computation/timing. Configurations are reading wrappers with explicit host placeholders, not executable JSON trainer inputs. They preserve historical labels/status strings; the current authorization is the separately confirmed confidential-review scope described here.

The numeric replay scripts require the matching numerical archive's records and manifests. These records are not part of this code-only candidate and must be provided separately through the authorized confidential review route. They recompute saved quantities and do not regenerate pixel predictions.

## Scope and limits

This is code for source inspection and configuration traceability, not a clean installable training application or a demonstrated cross-platform reproduction. Dataset permissions, actual review access, native ABI, host mappings and archived checkpoints remain separate. A code link alone cannot establish statistical superiority, physical-scene independence, complete task-specific baselines or CPU/DirectML equivalence.
