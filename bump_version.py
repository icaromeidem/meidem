"""
bump_version.py
---------------
Reads the latest Git tag and writes the version to meidem/_version.py
Run before building: python bump_version.py
"""

import subprocess
import pathlib

def get_version():
    result = subprocess.run(
        ['git', 'describe', '--tags', '--abbrev=0'],
        capture_output=True, text=True
    )
    tag = result.stdout.strip()
    if not tag.startswith('v'):
        raise ValueError(f"Tag '{tag}' does not start with 'v'")
    return tag[1:]  # remove o 'v'

version = get_version()
pathlib.Path("meidem/_version.py").write_text(f'__version__ = "{version}"\n')
print(f"✓ version set to {version}")