from pathlib import Path
import subprocess, shutil
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"/"worldcup_repo_csv"; DATA.mkdir(parents=True,exist_ok=True)
repo=ROOT/"_tmp_worldcup_repo"
if repo.exists(): shutil.rmtree(repo)
subprocess.check_call(["git","clone","--depth","1","https://github.com/jfjelstul/worldcup.git",str(repo)])
for p in (repo/"data-csv").glob("*.csv"): shutil.copy(p,DATA/p.name)
print("CSV files copied", len(list(DATA.glob("*.csv"))))
