defaults = {
    'data': {
        'path': 'data',
        'output': {
            'path': 'output.csv',
        },
        'meta': {
            'path': None
        }
    },
    'computation': {
        'random_seed': 0,
        'num_workers': 40,
        'use_gpu': True,
        'master_port': 6105,
        'dist_backend': 'nccl',
        'dist_init_method': 'tcp://localhost:9967',
        'shard_id': 0,
        'num_shards': 1,
        'use_distributed': True,
        'load_async': False,
        'multiprocess_meta_loading': True
    },
    'subset': {
        'ratio': 0.2,  # size = round(ratio * dataset_size)
        'size': None  # if this is set, ignore ratio
    },
    'clustering': {
        'pairing': 'combination'  # ['diagonal', 'bipartite', 'combination']
    },
    'batch': {
        'batch_size': 20,
        'selection_size': 4,
        'keep_unselected': True,
    },
    'contrastive': {
        'num_epochs': 3,
        'num_warmup_steps': 1,
        'base_lr': 2e-4,
        'train_batch_size': 128,
        'test_batch_size': 128,
        'cached_epoch': None,
        'train_from_cached': False,
    },
    'measure_name': 'batch_mi',
    'shuffle_candidates': True,
    'chunk_size': None,
    'save_cache_as_csvs': True,
    'log_every': 1000,
    'log_times': 10,
    'verbose': True,
    'debug': False
}
