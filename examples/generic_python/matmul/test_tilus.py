import torch
from tilus_matmul import MatmulBasic

sizes = [
    (65, 65, 17),
    (67, 71, 19),
    (1, 1, 1),
    (63, 63, 15),
    (129, 130, 33),
]

matmul = MatmulBasic()
for m, n, k in sizes:
    print(m, n, k)

    a = torch.randn(m, k, dtype=torch.float16, device="cuda")
    b = torch.randn(k, n, dtype=torch.float16, device="cuda")

    c_actual = torch.empty(m, n, dtype=torch.float16, device="cuda")
    c_expect = a @ b

    matmul(m, n, k, a, b, c_actual)

    torch.cuda.synchronize()

    torch.testing.assert_close(c_expect, c_actual, atol=1e-2, rtol=1e-2)