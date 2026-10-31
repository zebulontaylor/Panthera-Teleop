import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('publish',Path(__file__).parents[1]/'scripts/publish_processed.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)

def test_exact_replacement_removes_old_archives_and_provenance():
 remote=['.gitattributes','README.md','data/new.parquet','data/old.parquet','trimmed_full_rate/old.tar.gz','processing/source_metadata/old.json']
 local={'README.md','data/new.parquet'}
 assert set(p.obsolete_files(remote,local,True))=={'data/old.parquet','trimmed_full_rate/old.tar.gz','processing/source_metadata/old.json'}

def test_incremental_update_keeps_historical_archives():
 remote=['data/old.parquet','trimmed_full_rate/old.tar.gz']
 assert p.obsolete_files(remote,set(),False)==['data/old.parquet']
