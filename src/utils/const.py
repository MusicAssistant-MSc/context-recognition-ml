from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent[2]
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
INTERIM_DATA_DIR = DATA_DIR / 'interim'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# --- SiTunes dataset constants ---
SITUNES_DATA_DIR = RAW_DATA_DIR / 'SiTunes'

SITUNES_MUSIC_METADATA_DIR = [SITUNES_DATA_DIR / 'music_metadata']
SITUNES_MUSIC_METADATA_FILENAMES = ['music_info.csv', 'music_info_withname.csv']

SITUNES_STAGES = ['Stage2', 'Stage3']
SITUNES_STAGE_FILES = {
    'environment': 'env.json', 
    'interactions': 'interactions.csv', 
    'wrist': 'wrist.npy',
}

