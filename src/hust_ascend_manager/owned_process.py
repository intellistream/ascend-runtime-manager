"""Bounded CPU pilot lifecycle; signals apply only to processes started here."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def process_stat(pid):
    raw = Path(f'/proc/{pid}/stat').read_text()
    fields = raw[raw.rfind(')') + 2:].split()
    return {'state': fields[0], 'start_ticks': fields[19],
            'cpu_ticks': int(fields[11]) + int(fields[12]),
            'rss_pages': int(fields[21])}


class OwnedProcess:
    def __init__(self, name, argv, cwd, env, directory):
        self.name = name
        self.directory = Path(directory)
        self.log = (self.directory / f'{name}.log').open('wb')
        self.proc = subprocess.Popen(argv, cwd=cwd, env=env,
                                     stdout=self.log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        self.identity = process_stat(self.proc.pid)['start_ticks']
        self.receipt('start', argv=argv, cwd=str(cwd))

    def receipt(self, action, **extra):
        with (self.directory / 'lifecycle.private.jsonl').open('a') as f:
            f.write(json.dumps({'time_ns': time.time_ns(), 'service': self.name,
                                'pid': self.proc.pid, 'start_ticks': self.identity,
                                'action': action, **extra}) + '\n')

    def send(self, sig):
        if self.proc.poll() is not None:
            raise RuntimeError(f'{self.name} exited: {self.proc.returncode}')
        if process_stat(self.proc.pid)['start_ticks'] != self.identity:
            raise RuntimeError('process identity changed; refusing signal')
        os.kill(self.proc.pid, sig)
        self.receipt(sig.name)

    def close(self):
        if self.proc.poll() is None:
            self.send(signal.SIGCONT)
            self.send(signal.SIGTERM)
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.send(signal.SIGKILL)
                self.proc.wait(timeout=5)
        self.receipt('exit', returncode=self.proc.returncode)
        self.log.close()
