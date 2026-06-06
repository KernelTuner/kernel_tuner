try:
    from hip import hip, hiprtc
except (ImportError, RuntimeError):
    hip = None


def hip_check(call_result):
    """helper function to check return values of hip calls"""
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        _, error_name = hip.hipGetErrorName(err)
        _, error_str = hip.hipGetErrorString(err)
        raise RuntimeError(f"{error_name}: {error_str}")
    return result

