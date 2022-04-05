from pathlib import Path


def get_root_dir():
    current_dir = Path(__file__)
    return [p for p in current_dir.parents if p.parts[-1] == 'mezcal'][0]
