"""Darwin owned-process RSS reader using the public libproc API."""
import ctypes as ct
import errno
import os


class TaskInfo(ct.Structure):
    _fields_ = [(name, ct.c_uint64) for name in [
        'virtual_size', 'resident_size', 'total_user', 'total_system', 'threads_user', 'threads_system'
    ]] + [(name, ct.c_int32) for name in [
        'policy', 'faults', 'pageins', 'cow_faults', 'messages_sent', 'messages_received',
        'syscalls_mach', 'syscalls_unix', 'csw', 'threadnum', 'numrunning', 'priority'
    ]]


LIB = ct.CDLL('/usr/lib/libproc.dylib', use_errno=True)
LIB.proc_pidinfo.argtypes = [ct.c_int, ct.c_int, ct.c_uint64, ct.c_void_p, ct.c_int]
LIB.proc_pidinfo.restype = ct.c_int
assert ct.sizeof(TaskInfo) == 96


def rss_bytes(pids):
    values = {}
    for pid in pids:
        info = TaskInfo()
        ct.set_errno(0)
        n = LIB.proc_pidinfo(pid, 4, 0, ct.byref(info), ct.sizeof(info))
        if n != ct.sizeof(info):
            error = ct.get_errno()
            if pid != os.getpid() and error in [0, errno.ESRCH, errno.ENOENT]:
                values[pid] = 0  # owned child has already exited
                continue
            raise OSError(error, 'proc_pidinfo failed', pid)
        values[pid] = int(info.resident_size)
    if values.get(os.getpid(), 0) <= 0:
        raise RuntimeError('Missing controller RSS')
    return sum(values.values()), values
