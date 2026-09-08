"""Bounded CPU pilot lifecycle; signals apply only to processes started here."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import threading


def process_stat(pid):
    raw = Path(f'/proc/{pid}/stat').read_text()
    fields = raw[raw.rfind(')') + 2:].split()
    return {'state': fields[0], 'start_ticks': fields[19],
            'cpu_ticks': int(fields[11]) + int(fields[12]),
            'rss_pages': int(fields[21])}


class OwnedProcess:
    def __init__(self, name, argv, cwd, env, directory, on_stdout=None):
        self.name = name
        self.directory = Path(directory)
        self.log = (self.directory / f'{name}.log').open('wb')
        self.proc = subprocess.Popen(argv, cwd=cwd, env=env,
                                     stdout=subprocess.PIPE if on_stdout else self.log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        self.identity = process_stat(self.proc.pid)['start_ticks']
        self.receipt('start', argv=argv, cwd=str(cwd))
        self.reader = None
        self.reader_errors = []
        if on_stdout:
            def drain():
                for line in self.proc.stdout:
                    self.log.write(line)
                    self.log.flush()
                    try:
                        on_stdout(line)
                    except Exception as error:
                        self.reader_errors.append(repr(error))
                self.proc.stdout.close()
            self.reader = threading.Thread(target=drain, daemon=True)
            self.reader.start()

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
        if self.reader:
            self.reader.join(timeout=5)
            if self.reader.is_alive():
                raise RuntimeError('stdout collector did not drain')
        self.receipt('exit', returncode=self.proc.returncode, collector_errors=self.reader_errors)
        self.log.close()
        if self.reader_errors:
            raise RuntimeError('stdout collection failed: ' + repr(self.reader_errors))
