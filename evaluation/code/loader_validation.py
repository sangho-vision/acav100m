from data import loader
import utils.distributed as du

def loader_validate(cfg):
    # Create the audio-visual pretrain loader.
    pretrain_loader = loader.construct_loader(cfg, 'pretrain')

    num_batches_per_epoch = len(pretrain_loader)
    print(f"NUM_BATCHES: {num_batches_per_epoch}")
    rank = du.get_rank()
    for i, (visual_clip, audio_clip) in enumerate(pretrain_loader):
        batch_size = visual_clip[0].size(0)
        print(f"[RANK {rank}] step_{i}: batch_size={batch_size}")
