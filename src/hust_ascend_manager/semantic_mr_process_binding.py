"""Read physical worker placement from the driver, including manager-only FDs."""
import re
import subprocess


def process_npu_memberships():
    output=subprocess.check_output(['npu-smi','info'],text=True)
    memberships={}
    for line in output.splitlines():
        match=re.match(r'^\|\s*(\d+)\s+\d+\s*\|\s*(\d+)\s*\|',line)
        if match:
            npu,pid=map(int,match.groups())
            memberships.setdefault(pid,set()).add(npu)
    return memberships
