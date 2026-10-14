# Inspection entrypoints

| Purpose | File |
|---|---|
| PCOD training/loss/model definitions | review_source/15_pcod_training/train_pcod.py; review_source/15_pcod_training/pcod_models.py |
| Corrected PCOD interpolation/protocol | review_source/25_pcod_portable_runner/portable_core.py; review_source/25_pcod_portable_runner/execution_protocol.py |
| RGBP training | review_source/12_training/train_rgbp.py |
| RGBP declared runner/protocol | review_source/41_rgbp_portable_runner/rgbp_core.py; review_source/41_rgbp_portable_runner/rgbp_protocol.py |
| PCOD full-test evaluation | review_source/53_pcod_formal_evaluation/pcod_eval.py |
| RGBP metrics/export | review_source/45_rgbp_common_evaluation/evaluation.py; review_source/46_rgbp_prediction_export/export_core.py |
| Original-device evaluation | review_source/73_device_specific_evaluation/run73_v2.py |
| PCOD capacity/arithmetic | review_source/83_capacity_compute_evidence/analyze83.py |
| PCOD timing boundary | review_source/87_controlled_inference_cost/timing87.py |
| Saved primary numerical replay | numeric_replay/reproduce_primary.py |
| Saved cost numerical replay | numeric_replay/cost/reproduce_cost.py |

These are reading entrypoints; no model module or checkpoint is invoked by this preparation. All58 configuration wrappers are indexed in CONFIGURATION_INDEX.csv. Reading views and host substitutions do not equal the frozen executable identity.
