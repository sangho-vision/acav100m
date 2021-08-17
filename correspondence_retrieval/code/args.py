from munch import Munch


def get_args(**cmd_args):
    # Hyperparameters
    args = Munch()
    args.niters = 20  # # of k-means iterations
    args.seed = 1  # reproducibility
    args.ncentroids = 10  # # of k-means centroids
    args.batch_size = 100
    args.nclasses = 10
    args.kmeans_iters = args.niters
    args.nsamples_per_class = 100
    args.ntargets_per_class = args.nsamples_per_class // 2
    args.nexprs = 10
    args.finetune = False
    args.start_indices_selection = 'zero'
    args.clustering_func_type = 'faiss'  # 'scipy', 'faiss', 'sgd_kmeans'
    '''
    data_name: ['image_pair_mnist', 'image_pair_rotation',
                'image_pair_flip']
    '''
    args.data_name = 'image_pair_mnist'
    args.deranged_classes_ratio = 0.5
    args.out_root = '../../data/correspondence_retrieval/output'
    args.out_dir = '.'
    args.image_pair_data_path = '../../data/correspondence_retrieval/image_pair_data'
    args.data_requires_extraction = ['cifar10', 'cifar10-rotated', 'cifar10-flipped',
                                     'mnist']
    args.measure = 'efficient_mem_mi'
    args.optimization_algorithm = 'greedy'
    args.celf_ratio = 0
    args.cluster_pairing = 'combination'  # 'combination', 'bipartite', 'diagonal'
    args.pca_dim = None  # default to min dim over all features
    args.get_intermediate_stats = True
    args.shuffle_true_ids = True
    args.num_workers = 4
    args.model_name = 'ResNet50'
    args.extract_each_layer = False
    args.num_shards = None
    args.share_clustering = False
    args.chunk_size = 10000
    args.sample = False
    args.shuffle_each_cluster = False
    args.sample_level_correspondence_data = ['kinetics_sounds']
    args.sample_level = False
    args.train_ratio = None
    args.num_epochs = 20
    args.contrastive_batch_size = 10
    args.use_test_for_train = False
    args.base_lr = 2e-4
    args.save = True
    args.log_keys = []

    args.use_gpu = False
    args.batch_size = 100
    args.selection_size = 10
    args.keep_unselected = True

    args = Munch({**args, **cmd_args})

    return args
