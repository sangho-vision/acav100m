defaults = {
    'models': ['layer_vggish', 'layer_slow_fast'],
    'model_types': {
        'audio': ['vggish', 'layer_vggish'],
        'visual': ['slow_fast', 'layer_slowfast']
    },
    'root': '../',
    'data': {
        'path': 'data',
        'meta_file': None,
        'cache_dir': 'cache',
        'batch_size': 32,
        'media': {
            'path': 'video/shard-000000-000004-first20',
            'num_frames': 32
        },
        'meta': {
            'path': None
        },
        'output': {
            'path': 'output',
            'chunk_size': 1000,
            'shard_ok_ratio': 0.99
        },
        'types': {
            'FSDD': 'audio_only',
        },
    },
    'computation': {
        'random_seed': 0,
        'device': 'cuda',
        'num_workers': 40,
        'master_port': 6105,
        'use_distributed': False,
        'dist_init_method': 'tcp://localhost:9999',
        'dist_backend': 'nccl',
        'shard_id': 0,
        'num_shards': 1,
        'shuffle_bufsize': 100,
        'discard_shards': False,
        'num_gpus': None
    },
    'clustering': {
        'ncentroids': 32,
        'epochs': 2,
        'cached_epoch': None,
        'resume_training': False,
        'save_scheme_ver1': False,
        'load_cache_from_shard_subset': True,
    },
    'acav': {
        'duration': 10,
        'skip_shorter_ratio': 1 / 4,
        'model_cache_path': '../../data/cache',
        'save_cache_every': 1,
        'force_new_shards': False,
        'force_cache_restart': False,
        'use_replicates': False
    },
    'debug': False,
    'log_period': 1
}


tests = {
    'data': {
        'meta_file': 'cut_all_samples_500008.tsv',
    }
}
