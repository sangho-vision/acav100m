import time
import datetime
from multiprocessing import Pool


def multiprocess(func, data, num_workers=1, granularity='shards',
                 log_every=1000, verbose=False):
    start = time.time()
    if num_workers > 1:
        if verbose:
            print("parallel processing")
        out = {}
        with Pool(num_workers) as p:
            count = 0
            chunksize = max(1, len(data) // (num_workers))
            for i, res in p.imap_unordered(func, enumerate(data), chunksize=chunksize):
                out[i] = res
                count += 1
                if verbose:
                    if (count + 1) % log_every == 0:
                        elasped = time.time() - start
                        elasped = str(datetime.timedelta(seconds=elasped))
                        print("{}/{} {} processed (elasped: {})".format(count, len(data), granularity, elasped))
    else:
        if verbose:
            print("sequential processing")
        out = []
        count = 0
        for i, x in enumerate(data):
            i, res = func((i, x))
            out.append(res)
            count += 1
            if verbose:
                if (count + 1) % log_every == 0:
                    elasped = time.time() - start
                    elasped = str(datetime.timedelta(seconds=elasped))
                    print("{}/{} {} processed (elasped: {})".format(count, len(data), granularity, elasped))
        out = dict(enumerate(out))
    if verbose:
        print("sorting multiprocess outputs")
    out = [out[k] for k in sorted(list(out.keys()))]
    return out
