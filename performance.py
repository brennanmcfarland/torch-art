import torch.backends as backends
import torch.cuda as cuda


# performs optimizations in the backend if all input sizes are the same
# if they're not though this will likely just make performance worse
# NOTE: only works on cuda
def optimize_cuda_for_fixed_input_size():
    assert cuda.is_available()
    backends.cudnn.benchmark = True