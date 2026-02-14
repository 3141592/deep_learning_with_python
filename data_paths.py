from pathlib import Path
import os

def get_data_root():
    return Path(
        os.environ.get("DATA_ROOT", Path.home() / "src" / "data")
    )
