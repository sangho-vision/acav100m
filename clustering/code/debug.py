def get_open_fds():
    '''
    return the number of open file descriptors for current process

    .. warning: will only work on UNIX-like os-es.
    '''
    import subprocess
    import os

    pid = os.getpid()
    procs = subprocess.check_output(
        ["lsof", '-w', '-Ff', "-p", str(pid)])

    procs = procs.decode('utf-8')
    procs = procs.split('\n')
    procs = list(filter(lambda s: s and s[0] == 'f' and s[1:].isdigit(), procs))
    return procs
