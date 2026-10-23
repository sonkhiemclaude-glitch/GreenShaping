"""Bounded output budgets for JSON-only offline preparation, not pcap acquisition."""
import os
import shutil
from pathlib import Path

GIB=1024**3
MIB=1024**2


def output_bytes(root):
    root=Path(root)
    total=0
    if root.exists():
        for parent,dirs,files in os.walk(root,followlinks=False):
            for name in files:total+=(Path(parent)/name).lstat().st_size
    return total


def check(root,max_bytes=128*MIB,reserve_bytes=2*GIB,upcoming_bytes=4*MIB):
    root=Path(root);volume=root
    while not volume.exists():volume=volume.parent
    used=output_bytes(root)
    free=shutil.disk_usage(volume).free
    if used+upcoming_bytes>max_bytes:raise RuntimeError('offline output budget exceeded; stop acquisition')
    if free<reserve_bytes+upcoming_bytes:raise RuntimeError('offline storage reserve reached; stop acquisition')
    return {'output_bytes':used,'max_bytes':max_bytes,'free_bytes':free,'reserve_bytes':reserve_bytes}
