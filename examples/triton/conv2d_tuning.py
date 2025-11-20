import torch
import triton.language as tl
import numpy as np
from kernel_tuner.interface import tune_kernel
import os
import json
from datetime import datetime

# Check for required environment variable
cache_dir = os.getenv('KERNEL_TUNER_CACHE_DIR')
cache_file_name = os.getenv('KERNEL_TUNER_CACHE_FILE', 'conv2d_tuning_results.json')

if cache_dir is None:
    raise ValueError("Environment variable KERNEL_TUNER_CACHE_DIR must be set")

cache_file = os.path.join(cache_dir, cache_file_name)


def conv2d_output_size(
    in_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
) -> int:
    """
    Determines the output size of a 2D convolution operation.

    Args:
        in_size: Input size.
        kernel_size: Kernel size.
        stride: Stride.
        padding: Padding.

    Returns:
        Output size of 2D convolution.
    """
    return (in_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


def conv2d_forward_kernel(
    input_pointer,
    weight_pointer,
    output_pointer,
    in_n,
    input_height,
    input_width,
    out_c,
    out_height,
    out_width,
    input_n_stride,
    input_c_stride,
    input_height_stride,
    input_width_stride,
    weight_n_stride,
    weight_c_stride,
    weight_height_stride,
    weight_width_stride,
    output_n_stride,
    output_c_stride,
    output_height_stride,
    output_width_stride,
    weight_c: tl.constexpr,
    weight_height: tl.constexpr,
    weight_width: tl.constexpr,
    stride_height: tl.constexpr,
    stride_width: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    dilation_height: tl.constexpr,
    dilation_width: tl.constexpr,
    groups: tl.constexpr,
    BLOCK_NI_HO_WO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid_ni_ho_wo = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_group = tl.program_id(2)

    # caculate in_n out_height out_weight value in kernel
    ni_ho_wo_offset = pid_ni_ho_wo * BLOCK_NI_HO_WO + tl.arange(0, BLOCK_NI_HO_WO)
    ni_ho_offset = ni_ho_wo_offset // out_width
    in_n_point_value = ni_ho_offset // out_height
    output_height_point_value = ni_ho_offset % out_height
    output_width_point_value = ni_ho_wo_offset % out_width

    # Load the input and weight pointers. input and weight are of shape
    # [in_n, groups, in_c, input_height, input_width] and [groups, out_c, in_c, weight_height, weight_width]
    out_per_group_c = out_c // groups
    output_c_offset = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    input_pointer += (
        input_n_stride * in_n_point_value + input_c_stride * pid_group * weight_c
    )[:, None]
    weight_pointer += (
        weight_n_stride * output_c_offset
        + weight_n_stride * pid_group * out_per_group_c
    )[None, :]

    accum = tl.zeros((BLOCK_NI_HO_WO, BLOCK_CO), dtype=tl.float32)
    BLOCK_CI_COUNT = (weight_c + BLOCK_CI - 1) // BLOCK_CI
    for hwc in range(weight_height * weight_width * BLOCK_CI_COUNT):
        c = (hwc % BLOCK_CI_COUNT) * BLOCK_CI
        hw = hwc // BLOCK_CI_COUNT
        h = hw // weight_width
        w = hw % weight_width

        input_c_offset = c + tl.arange(0, BLOCK_CI)
        input_height_offset = (
            h * dilation_height
            - padding_height
            + stride_height * output_height_point_value
        )
        input_width_offset = (
            w * dilation_width - padding_width + stride_width * output_width_point_value
        )

        curr_input_pointer = (
            input_pointer
            + (input_c_stride * input_c_offset)[None, :]
            + (input_height_stride * input_height_offset)[:, None]
            + (input_width_stride * input_width_offset)[:, None]
        )
        curr_weight_pointer = (
            weight_pointer
            + (weight_c_stride * input_c_offset)[:, None]
            + (weight_height_stride * h)
            + (weight_width_stride * w)
        )

        input_mask = (
            (in_n_point_value < in_n)[:, None]
            & (input_c_offset < weight_c)[None, :]
            & (0 <= input_height_offset)[:, None]
            & (input_height_offset < input_height)[:, None]
            & (0 <= input_width_offset)[:, None]
            & (input_width_offset < input_width)[:, None]
        )
        weight_mask = (input_c_offset < weight_c)[:, None] & (
            output_c_offset < out_per_group_c
        )[None, :]

        input_block = tl.load(curr_input_pointer, mask=input_mask)
        weight_block = tl.load(curr_weight_pointer, mask=weight_mask)

        accum += tl.dot(input_block, weight_block, allow_tf32=False)

    output_pointer += (
        (output_n_stride * in_n_point_value)[:, None]
        + (output_c_stride * (pid_group * out_per_group_c + output_c_offset))[None, :]
        + (output_height_stride * output_height_point_value)[:, None]
        + (output_width_stride * output_width_point_value)[:, None]
    )
    output_mask = (
        (in_n_point_value < in_n)[:, None]
        & (output_c_offset < out_per_group_c)[None, :]
        & (output_height_point_value < out_height)[:, None]
        & (output_width_point_value < out_width)[:, None]
    )

    tl.store(output_pointer, accum, mask=output_mask)


def tune_conv2d(batch_size=1, in_channels=64, height=32, width=32, 
                out_channels=128, kernel_size=3, stride=1, padding=1, 
                groups=1):
    """
    Tune the conv2d kernel with different configurations.
    """
    # Create sample inputs
    input = torch.randn(batch_size, in_channels, height, width, 
                       device='cuda', dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels//groups, kernel_size, kernel_size, 
                        device='cuda', dtype=torch.float32)
    
    # Calculate output dimensions
    out_height = conv2d_output_size(height, kernel_size, stride, padding, 1)
    out_width = conv2d_output_size(width, kernel_size, stride, padding, 1)
    output = torch.empty((batch_size, out_channels, out_height, out_width),
                        device='cuda', dtype=torch.float32)

    # Prepare all arguments for the kernel
    arguments = [
        input, weight, output,
        np.int32(batch_size),
        np.int32(height),
        np.int32(width),
        np.int32(out_channels),
        np.int32(out_height),
        np.int32(out_width),
        np.int32(input.stride(0)),
        np.int32(input.stride(1)),
        np.int32(input.stride(2)),
        np.int32(input.stride(3)),
        np.int32(weight.stride(0)),
        np.int32(weight.stride(1)),
        np.int32(weight.stride(2)),
        np.int32(weight.stride(3)),
        np.int32(output.stride(0)),
        np.int32(output.stride(1)),
        np.int32(output.stride(2)),
        np.int32(output.stride(3)),
        np.int32(in_channels//groups),  # weight_c
        np.int32(kernel_size),          # weight_height
        np.int32(kernel_size),          # weight_width
        np.int32(stride),               # stride_height
        np.int32(stride),               # stride_width
        np.int32(padding),              # padding_height
        np.int32(padding),              # padding_width
        np.int32(1),                    # dilation_height
        np.int32(1),                    # dilation_width
        np.int32(groups),               # groups
    ]

    # Define tuning parameters - only powers of 2
    tune_params = {
        'BLOCK_NI_HO_WO': [2 ** i for i in range(4, 10)],
        'BLOCK_CI': [2 ** i for i in range(4, 10)],
        'BLOCK_CO': [2 ** i for i in range(4, 10)],
        'num_stages': [1, 2, 3, 4],
        'num_warps': [1, 2, 4, 8],
    }

    print(tune_params)

    # Define constraints
    constraints = [
        "BLOCK_CI <= %d" % (in_channels//groups),
        "BLOCK_CO <= %d" % out_channels,
    ]

    # Problem size for the grid
    problem_size = (
        batch_size * out_height * out_width,  # Grid dimension 0
        out_channels,                         # Grid dimension 1
        groups,                               # Grid dimension 2
    )

    # Grid divisor expressions
    grid_div_x = ["BLOCK_NI_HO_WO"]
    grid_div_y = ["BLOCK_CO"]
    grid_div_z = ["1"]

    results, env = tune_kernel(
        kernel_name='conv2d_forward_kernel',
        kernel_source=conv2d_forward_kernel,
        problem_size=problem_size,
        arguments=arguments,
        tune_params=tune_params,
        restrictions=constraints,
        lang='TRITON',
        grid_div_x=grid_div_x,
        grid_div_y=grid_div_y,
        grid_div_z=grid_div_z,
        block_size_names=['BLOCK_NI_HO_WO', 'BLOCK_CI', 'BLOCK_CO'],
        strategy='genetic_algorithm',
        strategy_options={
            'maxiter': 1000,
            'popsize': 100,
        },
        cache=cache_file,
    )

    return results


if __name__ == '__main__':
    # Run tuning with moderately large input dimensions
    results = tune_conv2d(
        batch_size=16,
        in_channels=128,
        height=112,
        width=112,
        out_channels=256,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=1
    )
    
        
    # Filter out failed compilations and find best config
    valid_results = [result for result in results if isinstance(result['time'], (int, float))]
    if valid_results:
        best_config = min(valid_results, key=lambda x: x['time'])
        print("\nBest configuration:")
        print(json.dumps(best_config, indent=2))
    else:
        print("\nNo valid configurations found - all compilations failed")

    # Create results dictionary with GPU info
    all_results = {
        "gpu_info": {
            "gpu_name": torch.cuda.get_device_name()
        },
        "results": valid_results
    }
    
    # Add timestamp to filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'conv2d_results_{timestamp}.json'
    
    # Save results
    import json
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)