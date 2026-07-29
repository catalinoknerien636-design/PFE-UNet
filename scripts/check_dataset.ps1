$root = Join-Path $PSScriptRoot '..\data\PCOD_1200\full-dataset'
& python -c "from pathlib import Path; from datasets.pcod import build_samples; r=Path(r'$root'); print(len(build_samples(r,'train')), len(build_samples(r,'test')))"
