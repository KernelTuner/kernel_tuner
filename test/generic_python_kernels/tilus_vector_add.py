import tilus
from tilus import float32, int32
from tilus.utils import cdiv

kernel_name = "VectorAdd"


class VectorAdd(tilus.Script):
    def __init__(self):
        super().__init__()
        self.block_size_x = 128
        self.num_warps = 4

    def __call__(self, c_ptr: ~float32, a_ptr: ~float32, b_ptr: ~float32, n: int32):
        self.attrs.blocks = [cdiv(n, self.block_size_x)]
        self.attrs.warps = self.num_warps

        offset: int32 = self.block_size_x * self.blockIdx.x
        ga = self.global_view(a_ptr, dtype=float32, shape=[n])
        gb = self.global_view(b_ptr, dtype=float32, shape=[n])
        gc = self.global_view(c_ptr, dtype=float32, shape=[n])

        a = self.load_global(ga, offsets=[offset], shape=[self.block_size_x])
        b = self.load_global(gb, offsets=[offset], shape=[self.block_size_x])
        self.store_global(gc, a + b, offsets=[offset])


def arguments(c, a, b, n):
    return [c, a, b, n]


def tune_params(n):
    return {"block_size_x": [128, 256], "num_warps": [4]}


def call_function(kernel_function, args, kwargs):
    kernel_function(*args, **kwargs)
