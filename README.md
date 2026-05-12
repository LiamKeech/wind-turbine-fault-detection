# Wind Turbine Fault Detection (LOF Demo)

Minimal Local Outlier Factor (LOF) implementation and demo runner.

## Quickstart

```powershell
pip install -r requirements.txt
```

```powershell
python scripts/lof_demo.py --limit 5000
```

## Notes

- Uses time-based split (70/15/15).
- Threshold set by validation score quantile.
- Model artifact saved under `outputs/models/lof/<version>/`.
