from kernel_tuner.utils.nvcuda import to_valid_nvrtc_gpu_arch_cc


def test_to_valid_nvrtc_gpu_arch_cc():
    assert to_valid_nvrtc_gpu_arch_cc("89") == "89"
    assert to_valid_nvrtc_gpu_arch_cc("88") == "87"
    assert to_valid_nvrtc_gpu_arch_cc("86") == "80"
    assert to_valid_nvrtc_gpu_arch_cc("40") == "52"
    assert to_valid_nvrtc_gpu_arch_cc("90b") == "90a"
    assert to_valid_nvrtc_gpu_arch_cc("91c") == "90a"
    assert to_valid_nvrtc_gpu_arch_cc("1234") == "52"
