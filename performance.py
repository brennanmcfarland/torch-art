import torch.backends as backends
import torch.cuda as cuda
import torch.nn as nn
import torch.utils as utils


# TODO: need to add functionality to add move feature maps out of VRAM - https://medium.com/syncedreview/how-to-train-a-very-large-and-deep-model-on-one-gpu-7b7edfe2d072 https://arxiv.org/pdf/1602.08124.pdf https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9
# TODO: do both the checkpointing and feature map moving approach and compare tradeoffs w benchmarking


# performs optimizations in the backend if all input sizes are the same
# if they're not though this will likely just make performance worse
# NOTE: only works on cuda


def optimize_cuda_for_fixed_input_size():
    assert cuda.is_available()
    backends.cudnn.benchmark = True


# checkpoint but only operates on top-level layers in a sequential model
# NOTE: there can be no generalized implementation for checkpointing at the leaf module level since checkpointing at
#  varying levels of submodule recursion could cause problems for modules that contain extra logic besides just
#  executing their submodules
# TODO: other related functions may be useful in the future:
#  do convolutions/dense only
#  pass in what layers we want to checkpoint
#  checkpoint at a given submodule recursion depth
# TODO: need an adaptive algorithm that figures out min # checkpoints needed
# TODO: for memory constraints and warns if over sqrt(n) (n = # layers)
# TODO: make it work for any ssuch checkpointing func
def checkpoint_sequential(module, num_checkpoints):
    return CheckpointedSequential(module, num_checkpoints)


# checkpoint container class holding all sequential layers scoped by a checkpoint
class CheckpointedSequential(nn.Module):
    # as with pytorch implementation of checkpoint_sequential, sequential can be an nn.Sequential or list of modules
    def __init__(self, sequential, num_checkpoints):
        super(CheckpointedSequential, self).__init__()
        self.sequential = sequential
        self.num_checkpoints = num_checkpoints

    def forward(self, x):
        return utils.checkpoint.checkpoint_sequential(self.sequential, self.num_checkpoints + 1, x)

    def children(self):
        return self.sequential
