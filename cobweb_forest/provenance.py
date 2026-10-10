"""Record which code a run used (compute nodes can see stale NFS copies)."""
import glob
import hashlib
import os
import socket
import subprocess


def code_hashes(root=None):
    root = root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = {}
    for p in sorted(glob.glob(os.path.join(root, "cobweb_forest", "*.py")) +
                    glob.glob(os.path.join(root, "cobweb_forest", "*.so"))):
        out[os.path.relpath(p, root)] = hashlib.md5(open(p, "rb").read()).hexdigest()[:12]
    try:
        out["git"] = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"],
                                    capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    out["host"] = socket.gethostname()
    return out
