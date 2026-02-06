import tilus
from tilus import float16, float32, int32
from tilus.utils import cdiv, benchmark_func
import torch
import math
from kernel_tuner import tune_kernel, run_kernel

'''
@tilus.autotune("num_warps", [4, 8])
@tilus.autotune("block_m, block_n", [(128, 128), (128, 64), (64, 128), (32, 256)])
@tilus.autotune("block_k", [16, 32])
@tilus.autotune("num_stages", [3, 4, 5])
@tilus.autotune("split_k_factor", [1, 4, 12, 16])
'''
class MatmulV5(tilus.Script):
    '''
    def __init__(self, block_m, block_n, block_k, num_warps, num_stages, split_k_factor):
        super().__init__()
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.split_k_factor = split_k_factor
    '''
    def __init__(self):
        super().__init__()
        self.block_m = 128
        self.block_n = 128
        self.block_k = 16
        self.num_warps = 4
        self.num_stages = 4
        self.split_k_factor = 4

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
            cdiv(m_size, self.block_m),
            cdiv(n_size, self.block_n),
            self.split_k_factor,
        ]
        self.attrs.warps = self.num_warps

        # the k_size for each thread block
        block_k_size = (
            cdiv(cdiv(k_size, self.split_k_factor), self.block_k) * self.block_k
        )
        start_offset_k = self.blockIdx.z * block_k_size
        end_offset_k = min(start_offset_k + block_k_size, k_size)

        block_m, block_n, block_k = self.block_m, self.block_n, self.block_k
        offset_m: int32 = block_m * self.blockIdx.x
        offset_n: int32 = block_n * self.blockIdx.y

        ga = self.global_view(a_ptr, dtype=float16, shape=[m_size, k_size])
        gb = self.global_view(b_ptr, dtype=float16, shape=[k_size, n_size])
        sa = self.shared_tensor(dtype=float16, shape=[self.num_stages, block_m, block_k])
        sb = self.shared_tensor(dtype=float16, shape=[self.num_stages, block_k, block_n])
        acc = self.register_tensor(dtype=float32, shape=[block_m, block_n], init=0.0)

        for stage in range(self.num_stages - 1):
            offset_k = start_offset_k + stage * self.block_k
            self.copy_async(src=ga, dst=sa[stage], offsets=[offset_m, offset_k])
            self.copy_async(src=gb, dst=sb[stage], offsets=[offset_k, offset_n])
            self.copy_async_commit_group()

        self.copy_async_wait_group(n=self.num_stages - 2)
        self.sync()

        current_stage: int32 = 0
        preload_stage: int32 = self.num_stages - 1
        for offset_k in self.range(
            start_offset_k, end_offset_k, block_k, unroll=self.num_stages
        ):
            # computation for current tile
            a = self.load_shared(sa[current_stage])
            b = self.load_shared(sb[current_stage])
            self.dot(a, b, acc, out=acc)

            # preload the next tile of A and B into shared memory
            preload_offset_k = offset_k + (self.num_stages - 1) * block_k
            if preload_offset_k < end_offset_k:
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

        # free the shared memory tensors for A and B
        self.free_shared(sa)
        self.free_shared(sb)

        # cast the accumulator to float16 and change the register tensor's layout
        sc = self.shared_tensor(dtype=float16, shape=[block_m, block_n])
        casted_acc = self.cast(acc, dtype=float16)
        self.store_shared(sc, casted_acc)
        self.sync()
        rc = self.load_shared(sc)
        self.free_shared(sc)

        m_blocks, n_blocks = cdiv(m_size, block_m), cdiv(n_size, block_n)
        gc = self.global_view(c_ptr, dtype=float16, shape=[m_size, n_size])
        if self.split_k_factor == 0:
            self.store_global(gc, rc, offsets=[offset_m, offset_n])
        else:
            semaphores = self.global_tensor(
                dtype=int32, shape=[m_blocks, n_blocks], requires_clean=True
            )
            semaphore: ~int32 = ~semaphores[self.blockIdx.x, self.blockIdx.y]

            # load and accumulate the partial result in global memory
            if self.blockIdx.z > 0:
                self.lock_semaphore(semaphore, value=self.blockIdx.z)
                partial_rc = self.load_global(
                    gc, offsets=[offset_m, offset_n], shape=[block_m, block_n]
                )
                self.add(rc, partial_rc, out=rc)

            # store the result to global memory and release the semaphore
            self.store_global(gc, rc, offsets=[offset_m, offset_n])

            # release the semaphore
            self.sync()  # we need to make sure the previous store_global is finished
            self.release_semaphore(
                semaphore, value=(self.blockIdx.z + 1) % self.split_k_factor
            )


def call_tilus(kernel_function, args, kwargs, grid, threads, params):
    kernel_function(*args, **kwargs) 


def without_kernel_tuner():
    tilus.option.clear_cache = True

    tilus.option.verbose_autotune = True
    m, n, k = 4096, 4096, 4096

    # create an instance of the kernel we have just defined
    matmul = MatmulV5()

    a = (torch.rand(m, k, dtype=torch.float16).cuda() - 0.5) / math.sqrt(k)
    b = (torch.rand(k, n, dtype=torch.float16).cuda() - 0.5) / math.sqrt(k)
    c_actual = torch.empty(m, n, dtype=torch.float16).cuda()
    c_expect = a @ b
    
    
    torch.cuda.synchronize()
    # launch the kernel by passing required arguments
    matmul(m, n, k, a, b, c_actual)
    torch.cuda.synchronize()

    # check correctness
    torch.testing.assert_close(c_expect, c_actual, atol=1e-2, rtol=1e-2)


    import pandas
    rows = []
    headers = ["m", "n", "k", "name", "latency (ms)", "tflops"]
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



#best performing configuration:
#block_m=128, block_n=64, block_k=16, num_warps=4, num_stages=4, split_k_factor=1, time=2.027ms

def main():
    m, n, k = 4096, 4096, 4096

    # create an instance of the kernel we have just defined
    #matmul = MatmulV5()

    a = (torch.rand(m, k, dtype=torch.float16).cuda() - 0.5) / math.sqrt(k)
    b = (torch.rand(k, n, dtype=torch.float16).cuda() - 0.5) / math.sqrt(k)
    c_actual = torch.empty(m, n, dtype=torch.float16).cuda()
    c_expect = a @ b
    
    '''
    torch.cuda.synchronize()
    # launch the kernel by passing required arguments
    matmul(m, n, k, a, b, c_actual)
    torch.cuda.synchronize()

    # check correctness
    torch.testing.assert_close(c_expect, c_actual, atol=1e-2, rtol=1e-2)
    '''

    args = [m, n, k, a, b, c_actual]
    tune_params = dict()
    tune_params["block_m"] = [32, 64, 128] #[16, 32, 64, 128, 256]
    tune_params["block_n"] = [32, 64, 128] #[16, 32, 64, 128, 256]
    tune_params["block_k"] = [16, 32] #[16, 32, 64, 128, 256]
    tune_params["num_warps"] = [4, 8] #[2, 4, 8, 16]
    tune_params["num_stages"] = [3, 4, 5] #[2, 3, 4, 5, 6]
    tune_params["split_k_factor"] = [1, 4, 12, 16] #[1, 4, 12, 16, 20]

#@tilus.autotune("num_warps", [4, 8])
#@tilus.autotune("block_m, block_n", [(128, 128), (128, 64), (64, 128), (32, 256)])
#@tilus.autotune("block_k", [16, 32])
#@tilus.autotune("num_stages", [3, 4, 5])
#@tilus.autotune("split_k_factor", [1, 4, 12, 16])

    '''
    restrictions = [
        "block_m * block_n <= 4096",
        "block_m * block_k <= 2048",
        "block_k * block_n <= 4096",
        "block_k >= 16",
        "2 * (num_stages * block_k * (block_m + block_n) + block_m * block_n) <= 65536", # shared mem
        "num_warps * 32 <= 1024",
        "block_k * split_k_factor <= 4096",
    ]
    '''
    restrictions = ["block_m * block_n  >= 8192", "block_m * block_n <= 16384"]


    
    results, env = tune_kernel(
        kernel_name="MatmulV5", # This has to be a string of the actual name. TODO is this always the case?
        kernel_source=MatmulV5,
        problem_size=[m, n],
        arguments=args,
        tune_params=tune_params,
        lang="generic_python",
        answer=[None, None, None, None, None, c_expect.cpu()],
        atol=1e-2,
        restrictions=restrictions,
        block_size_names=["block_m", "block_n", "block_k"],
        call_function=call_tilus,
        #strategy="random_sample",
    )
    



if __name__ == "__main__":
    main()
    #without_kernel_tuner()