"""Hash tracked files except the checksum file itself. Run from a Git checkout."""
import hashlib
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
paths = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
lines = []
for name in sorted(p for p in paths if p and p != 'SHA256SUMS.txt'):
    lines.append(f"{hashlib.sha256((root/name).read_bytes()).hexdigest()}  ./{name}")
(root/'SHA256SUMS.txt').write_text('\n'.join(lines)+'\n')
print(f'Hashed {len(lines)} tracked files; excluded SHA256SUMS.txt')
