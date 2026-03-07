import math

import pandas
import tilus
import torch
from tilus import float16, float32, int32
from tilus.utils import benchmark_func
from kernel_tuner import tune_kernel, run_kernel



class MatmulV4(tilus.Script):
    def __init__(self):
        super().__init__()
        self.block_m = 128
        self.block_n = 128
        self.block_k = 16
        self.num_warps = 4
        self.num_stages = 4

    def __call__(
        self,
        m_size: int32,
        n_size: int,
        k_size: int,
        a_ptr: ~float16,
        b_ptr: ~float16,
        c_ptr: ~float16,
    ):
        self.attrs.blocks = [
            self.utils.ceil_div(m_size, self.block_m),
            self.utils.ceil_div(n_size, self.block_n),
        ]
        self.attrs.warps = self.num_warps

        block_m, block_n, block_k = self.block_m, self.block_n, self.block_k
        offset_m: int32 = block_m * self.blockIdx.x
        offset_n: int32 = block_n * self.blockIdx.y

        ga = self.global_view(a_ptr, dtype=float16, shape=[m_size, k_size])
        gb = self.global_view(b_ptr, dtype=float16, shape=[k_size, n_size])
        sa = self.shared_tensor(dtype=float16, shape=[self.num_stages, block_m, block_k])
        sb = self.shared_tensor(dtype=float16, shape=[self.num_stages, block_k, block_n])
        acc = self.register_tensor(dtype=float32, shape=[block_m, block_n], init=0.0)

        for stage in range(self.num_stages - 1):
            offset_k = stage * self.block_k
            self.copy_async(src=ga, dst=sa[stage], offsets=[offset_m, offset_k])
            self.copy_async(src=gb, dst=sb[stage], offsets=[offset_k, offset_n])
            self.copy_async_commit_group()

        self.copy_async_wait_group(n=self.num_stages - 2)
        self.sync()

        current_stage: int32 = 0
        preload_stage: int32 = self.num_stages - 1
        for offset_k in self.range(0, k_size, block_k, unroll=self.num_stages):
            # computation for current tile
            a = self.load_shared(sa[current_stage])
            b = self.load_shared(sb[current_stage])
            self.dot(a, b, acc, out=acc)

            # preload the next tile of A and B into shared memory
            preload_offset_k = offset_k + (self.num_stages - 1) * block_k
            self.copy_async(
                src=ga,
                dst=sa[preload_stage],
                offsets=[offset_m, preload_offset_k],
            )
            self.copy_async(
                src=gb,
                dst=sb[preload_stage],
                offsets=[preload_offset_k, offset_n],
            )
            self.copy_async_commit_group()

            # update the stage
            current_stage = (current_stage + 1) % self.num_stages
            preload_stage = (preload_stage + 1) % self.num_stages
            self.copy_async_wait_group(n=self.num_stages - 2)
            self.sync()

        self.free_shared(sa)
        self.free_shared(sb)

        casted_acc = self.cast(acc, dtype=float16)
        gc = self.global_view(c_ptr, dtype=float16, shape=[m_size, n_size])
        self.store_global(gc, casted_acc, offsets=[offset_m, offset_n])


class MatmulGroupedOrdering(tilus.Script):
    def __init__(self):
        super().__init__()
        self.block_m = 128
        self.block_n = 128
        self.block_k = 16
        self.num_warps = 4
        self.num_stages = 4
        self.group_size_m = 8

    def __call__(
        self,
        m_size: int32,
        n_size: int,
        k_size: int,
        a_ptr: ~float16,
        b_ptr: ~float16,
        c_ptr: ~float16,
    ):
        block_m, block_n, block_k = self.block_m, self.block_n, self.block_k
        
        num_pid_m = self.utils.ceil_div(m_size, block_m)
        num_pid_n = self.utils.ceil_div(n_size, block_n)
        self.attrs.blocks = [num_pid_m * num_pid_n]

        pid = self.blockIdx.x
        num_pid_in_group = self.group_size_m * num_pid_n
        group_id = pid // num_pid_in_group

        first_pid_m = group_id * self.group_size_m
        group_size_m = min(num_pid_m - first_pid_m, self.group_size_m)

        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        self.attrs.warps = self.num_warps
        offset_m: int32 = pid_m * block_m
        offset_n: int32 = pid_n * block_n

        ga = self.global_view(a_ptr, dtype=float16, shape=[m_size, k_size])
        gb = self.global_view(b_ptr, dtype=float16, shape=[k_size, n_size])
        sa = self.shared_tensor(dtype=float16, shape=[self.num_stages, block_m, block_k])
        sb = self.shared_tensor(dtype=float16, shape=[self.num_stages, block_k, block_n])
        acc = self.register_tensor(dtype=float32, shape=[block_m, block_n], init=0.0)

        for stage in range(self.num_stages - 1):
            offset_k = stage * self.block_k
            self.copy_async(src=ga, dst=sa[stage], offsets=[offset_m, offset_k])
            self.copy_async(src=gb, dst=sb[stage], offsets=[offset_k, offset_n])
            self.copy_async_commit_group()

        self.copy_async_wait_group(n=self.num_stages - 2)
        self.sync()

        current_stage: int32 = 0
        preload_stage: int32 = self.num_stages - 1
        for offset_k in self.range(0, k_size, block_k, unroll=self.num_stages):
            # computation for current tile
            a = self.load_shared(sa[current_stage])
            b = self.load_shared(sb[current_stage])
            self.dot(a, b, acc, out=acc)

            # preload the next tile of A and B into shared memory
            preload_offset_k = offset_k + (self.num_stages - 1) * block_k
            self.copy_async(
                src=ga,
                dst=sa[preload_stage],
                offsets=[offset_m, preload_offset_k],
            )
            self.copy_async(
                src=gb,
                dst=sb[preload_stage],
                offsets=[preload_offset_k, offset_n],
            )
            self.copy_async_commit_group()

            # update the stage
            current_stage = (current_stage + 1) % self.num_stages
            preload_stage = (preload_stage + 1) % self.num_stages
            self.copy_async_wait_group(n=self.num_stages - 2)
            self.sync()

        self.free_shared(sa)
        self.free_shared(sb)

        casted_acc = self.cast(acc, dtype=float16)
        gc = self.global_view(c_ptr, dtype=float16, shape=[m_size, n_size])
        self.store_global(gc, casted_acc, offsets=[offset_m, offset_n])



def main():
    headers = ["m", "n", "k", "name", "latency (ms)", "tflops"]
    workloads = [
        [4096, 4096, 4096],
        [1024, 1024, 14336],
    ]

    rows = []
    for m, n, k in workloads:
        matmul = MatmulGroupedOrdering() #MatmulV4()

        a = (torch.rand(m, k, dtype=torch.float16).cuda() - 0.5) / math.sqrt(k)
        b = (torch.rand(k, n, dtype=torch.float16).cuda() - 0.5) / math.sqrt(k)
        c_actual = torch.empty(m, n, dtype=torch.float16).cuda()
        c_expect = a @ b
        matmul(m, n, k, a, b, c_actual)

        # check correctness
        torch.testing.assert_close(c_expect, c_actual)

        # benchmark
        for name, func in [
            ("torch", lambda: torch.matmul(a, b, out=c_expect)),
            ("tilus", lambda: matmul(m, n, k, a, b, c_actual)),
        ]:
            latency = benchmark_func(func, warmup=5, repeat=20)
            tflops = 2 * m * n * k / latency * 1e-9
            rows.append([m, n, k, name, latency, tflops])

    df = pandas.DataFrame(rows, columns=headers)
    print(df)

def call_tilus(kernel_function, args, kwargs, grid, threads, params):
    kernel_function(*args, **kwargs) 


def tune_matmul(m, n, k):
    a = (torch.rand(m, k, dtype=torch.float16).cuda() - 0.5) / math.sqrt(k)
    b = (torch.rand(k, n, dtype=torch.float16).cuda() - 0.5) / math.sqrt(k)
    c_actual = torch.empty(m, n, dtype=torch.float16).cuda()
    c_expect = a @ b

    size = m * n #(m, n)
    args = [m, n, k, a, b, c_actual]
    tune_params = dict()
    tune_params["block_m"] = [32, 64, 128, 256]
    tune_params["block_n"] = [32, 64, 128, 256]
    tune_params["block_k"] = [32, 64, 128]
    tune_params["group_size_m"] = [4, 8]
    tune_params["num_stages"] = [2, 3, 4]
    tune_params["num_warps"] = [4, 8]
    

    restrictions = [
        # tile size budget
        "block_m * block_n <= 16384",

        # aspect ratio <= 4 (no max/min allowed, so expand manually)
        "block_m <= 4 * block_n",
        "block_n <= 4 * block_m",

        # large K only with reasonably large M/N
        "not (block_k == 128 and block_m < 64 and block_n < 64)",

        # 32x32 requires 8 warps
        "not (block_m == 32 and block_n == 32 and num_warps < 8)",
    ]


    answer = [None] * 6
    answer[-1] = c_expect.cpu()
    atol = 1e-2 #m * 2**(-11)

    results, env = tune_kernel("MatmulGroupedOrdering", MatmulGroupedOrdering, size, args, tune_params, grid_div_x = ["block_m", "block_n"],
                               answer = answer, atol=atol, restrictions=restrictions, 
                               lang="generic_python", call_function=call_tilus, 
                               block_size_names=["block_m", "block_n", "block_k"], strategy="simulated_annealing")


if __name__ == "__main__":
    #m, n, k = 4096, 4096, 4096
    m, n, k = 8192, 8192, 8192
    tune_matmul(m, n, k)

    #main()