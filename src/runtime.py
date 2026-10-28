"""Owned subprocess watchdog. No ML dependencies; dummy-process testable."""
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from common import atomic, now


def rss_bytes(pids):
    result = subprocess.run(['/bin/ps', '-o', 'pid=,rss=', '-p', ','.join(map(str, pids))],
                            text=True, capture_output=True, timeout=.3)
    values = {}
    for line in result.stdout.splitlines():
        pid, kib = map(int, line.split())
        values[pid] = kib * 1024
    # A missing parent or inaccessible sampler is a failure, not zero memory.
    if os.getpid() not in values:
        raise RuntimeError('RSS sampler did not return controller')
    return sum(values.values()), values


def sizes(folder):
    folder = os.environ.get('ESS_PUBLIC_RUN', folder)
    total = logs = 0
    for p in Path(folder).rglob('*'):
        if p.is_file():
            b = p.stat().st_size
            total += b
            if p.suffix in ['.log', '.tsv'] or 'events.out' in p.name or 'trace' in p.name:
                logs += b
    return total, logs


def kill_owned(proc):
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    proc.wait(timeout=2)


def supervise(command, folder, phase, seconds, total_deadline, limits, tick):
    """Samples every <=.5s; soft RSS threshold may be overshot between samples."""
    start = time.monotonic()
    deadline = min(start + seconds, total_deadline) - limits['stop_grace_seconds_within_each_phase']
    peak = 0
    reason, exit_code, overshoot, sample_gap = 'completed', None, 0, 0.
    maximum_gap = 0.
    last_sample = start
    proc = None
    path = Path(folder) / (phase + '.log')
    with path.open('x') as log:
        try:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while True:
                at = time.monotonic()
                sample_gap = at - last_sample
                maximum_gap = max(maximum_gap, sample_gap)
                last_sample = at
                rss, components = rss_bytes([os.getpid(), proc.pid])
                peak = max(peak, rss)
                total, log_bytes = sizes(folder)
                tick(dict(current_rss_bytes=rss, rss_components=components, peak_rss_bytes=peak,
                          output_bytes=total, logs_bytes=log_bytes))
                if sample_gap > .5:
                    reason = 'RSS_sampler_gap'
                elif rss >= limits['RSS_sampled_parent_plus_worker_threshold_bytes']:
                    reason = 'RSS_threshold'
                    overshoot = rss - limits['RSS_sampled_parent_plus_worker_threshold_bytes']
                elif total >= limits['output_all_including_pending_bytes_max']:
                    reason = 'output_limit'
                elif log_bytes >= limits['logs_traces_bytes_max']:
                    reason = 'log_limit'
                elif at >= deadline:
                    reason = 'time_limit'
                elif proc.poll() is not None:
                    exit_code = proc.returncode
                    reason = 'completed' if exit_code == 0 else 'worker_failed'
                    break
                if reason != 'completed':
                    kill_owned(proc)
                    exit_code = proc.returncode
                    break
                # Target .25s start-to-start; record actual scheduling gaps.
                time.sleep(min(max(0., .25 - (time.monotonic() - at)),
                               max(0., deadline - time.monotonic())))
        except BaseException as exc:
            if proc is not None:
                kill_owned(proc)
                exit_code = proc.returncode
            reason = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'watchdog_failed'
            error = type(exc).__name__ + ': ' + str(exc)
        else:
            error = None
    result = dict(phase=phase, reason=reason, exit_code=exit_code,
                  elapsed=time.monotonic() - start, cap_seconds=seconds,
                  sampled_peak_rss_bytes=peak, rss_overshoot_bytes=overshoot,
                  maximum_sample_gap_seconds=maximum_gap,
                  poll_target_seconds=.5, memory_is_hard_OS_cap=False,
                  error=error, owned_pid=None if proc is None else proc.pid)
    atomic(Path(folder) / (phase + '.supervision.json'), result)
    return result


def progress(run, record):
    atomic(Path(run) / 'progress.json', record)
    with (Path(run) / 'progress.tsv').open('a') as f:
        f.write('\t'.join(str(record.get(k, '')) for k in
                ['heartbeat', 'phase', 'status', 'elapsed', 'remaining_wall_seconds',
                 'main_completed', 'calibrators_completed', 'reference_completed',
                 'current_rss_bytes', 'peak_rss_bytes', 'output_bytes']) + '\n')
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)

if __import__('sys').platform == 'darwin':
    from native_rss import rss_bytes
