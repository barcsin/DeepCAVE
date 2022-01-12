from pathlib import Path
from typing import Union


def make_dirs(filename: Union[str, Path], parents=True):
    path = Path(filename)
    if path.suffix != "":  # Is file
        path = path.parent

    path.mkdir(exist_ok=True, parents=parents)
