from pathlib import Path

import webdataset as wds
from .utils import Curried


class MetaWebDataset(wds.Dataset):
    def __init__(self, *args, initial_pipeline=None, skip_lists={}, **kwargs):
        if initial_pipeline is None:
            initial_pipeline = []
        initial_pipeline.append(group_by_keys())
        self.skip_lists = skip_lists
        super().__init__(*args, initial_pipeline=initial_pipeline, **kwargs)

    def raw_samples(self, urls):
        assert isinstance(urls, list)
        for url in urls:
            self.shard_hook()
            stream = None
            try:
                shard_name = Path(url).stem
                # shard_size = get_tar_size(url)
                skip_list = self.skip_lists[shard_name] if shard_name in self.skip_lists else []
                with self.open_fn(url) as stream:
                    files_of_archive = wds.dataset.tardata(stream, handler=self.tarhandler)
                    for fname, content in files_of_archive:
                        if Path(fname).stem not in skip_list:
                            # yield shard_name, shard_size, fname, content
                            yield shard_name, fname, content
                    wds.dataset.maybe_collect()
            except Exception as exn:
                if self.tarhandler(exn):
                    continue
                else:
                    break


def group_by_keys_(data, keys=wds.dataset.base_plus_ext, lcase=True, suffixes=None):
    """Returns function over iterator that groups key, value pairs into samples.
    keys: function that splits the key into key and extension (base_plus_ext)
    lcase: convert suffixes to lower case (Default value = True)
    """

    current_sample = None
    # for shard_name, shard_size, fname, value in data:
    for shard_name, fname, value in data:
        prefix, suffix = keys(fname)
        if wds.dataset.trace:
            print(
                prefix,
                suffix,
                current_sample.keys() if isinstance(current_sample, dict) else None,
            )
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        if current_sample is None or prefix != current_sample["__key__"]:
            if wds.dataset.valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix)
        if suffix in current_sample:
            raise ValueError(
                f"{fname}: duplicate file name in tar file {suffix} {current_sample.keys()}"
            )
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
        current_sample['shard_name'] = shard_name
        # current_sample['shard_size'] = shard_size
    if wds.dataset.valid_sample(current_sample):
        yield current_sample


group_by_keys = Curried(group_by_keys_)
