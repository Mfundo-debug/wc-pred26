from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
print("100,000 simulation outputs:")
for f in sorted((ROOT/"outputs").glob("*.csv")): print(f.name)
