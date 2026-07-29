param([string]$DatasetRoot,[string]$Checkpoint)
& python analysis\smoke_test.py --dataset-root $DatasetRoot --formal-checkpoint $Checkpoint --output-dir results\smoke_test
