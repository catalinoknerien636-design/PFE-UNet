$configs = Get-ChildItem (Join-Path $PSScriptRoot '..\configs\*.yaml')
foreach($config in $configs){ & python train.py --config $config.FullName --device cpu }
