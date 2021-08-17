from .derangement import get_derangements
from .sharded_derangement import get_sharded_derangements
from .sample_level import get_sample_level_sharded_derangements
from .split import split_dataset


def derangement(views, *args, sample_level=False, num_shards=None, train_ratio=None, **kwargs):
    if train_ratio is not None:
        assert train_ratio > 0 and train_ratio < 1, "train ratio must be within (0, 1), got {}".format(train_ratio)
        train, test, train_len, test_len = split_dataset(views, train_ratio, sample_level)
        print("splitted dataset into train ({}) and test ({})".format(train_len, test_len))
    else:
        test = views
        train = None
    if sample_level:
        test = get_sample_level_sharded_derangements(test, *args, num_shards=num_shards, **kwargs)
    elif num_shards is None:
        test = get_derangements(test, *args, **kwargs)
    else:
        test = get_sharded_derangements(test, *args, num_shards=num_shards, **kwargs)
    return train, test
