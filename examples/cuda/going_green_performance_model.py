#!/usr/bin/env python
"""
This example demonstrates how to use the power-frequency model presented in

  * Going green: optimizing GPUs for energy efficiency through model-steered auto-tuning
    R. Schoonhoven, B. Veenboer, B. van Werkhoven, K. J. Batenburg
    International Workshop on Performance Modeling, Benchmarking and Simulation of High Performance Computer Systems (PMBS) at Supercomputing (SC22) 2022

to reduce the number of frequencies for GPU energy tuning.

In particular, this example creates a plot with the modeled power consumption vs
frequency curve, highlighting the ridge frequency and the frequency range
selected by the user.

This example requires CUDA and NVML as well as PyCuda and a CUDA-capable
GPU with the ability (and permissions) to set applications clocks. GPUs
that do support locked clocks but not application clocks may use the
locked_clocks=True option.

"""
import argparse
from collections import OrderedDict
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import optimize
import time

try:
    from pycuda import driver as drv
except ImportError as e:
    drv = None
    raise e

from kernel_tuner.energy import energy
from kernel_tuner.observers.nvml import get_nvml_gr_clocks

def get_default_parser():
    parser = argparse.ArgumentParser(
        description='Find energy efficient frequencies')
    parser.add_argument("-d", dest="device", nargs="?",
                        default=0, help="GPU ID to use")
    parser.add_argument("-s", dest="samples", nargs="?",
                        default=10, help="Number of frequency samples")
    parser.add_argument("-r", dest="range", nargs="?",
                        default=10, help="Frequency spread (10%% of 'optimum')")
    parser.add_argument("-n", dest="number", nargs="?", default=10,
                        help="Maximum number of suggested frequencies")
    parser.add_argument("-l", dest="locked_clocks", nargs="?", default=False,
                        help="Whether to use locked clocks over application clocks")
    parser.add_argument("-nsf", dest="nvidia_smi_fallback", nargs="?", default=None,
                        help="Path to nvidia-smi as fallback when missing NVML permissions")


    return parser


if __name__ == "__main__":
    parser = get_default_parser()
    args = parser.parse_args()

    ridge_frequency, freqs, nvml_power, fitted_params, scaling = energy.create_power_frequency_model(device=args.device,
                                                                                               n_samples=int(args.samples),
                                                                                               verbose=True,
                                                                                               nvidia_smi_fallback=args.nvidia_smi_fallback,
                                                                                               use_locked_clocks=args.locked_clocks)

    all_frequencies = np.array(get_nvml_gr_clocks(args.device, quiet=True)['nvml_gr_clock'])

    frequency_selection = energy.get_frequency_range_around_ridge(ridge_frequency, all_frequencies, args.range, args.number, verbose=True)
    print(f"Search space reduction: {np.round(100 - len(frequency_selection) / len(all_frequencies) * 100, 1)} %")

    xs = np.linspace(all_frequencies[0], all_frequencies[-1], 100)
    # scale to start at 0
    xs -= scaling[0]
    modelled_power = energy.estimated_power(xs, *fitted_params)
    # undo scaling
    xs += scaling[0]
    modelled_power *= scaling[1]

    # Add point for ridge frequency
    P_ridge = energy.estimated_power([ridge_frequency - scaling[0]], *fitted_params) * scaling[1]

    # Add the frequency range
    min_freq = 1e-2 * (100 - int(args.range)) * ridge_frequency
    max_freq = 1e-2 * (100 + int(args.range)) * ridge_frequency

    # plot measurements with model
    try:
        import seaborn as sns
        sns.set_theme(style="darkgrid")
        sns.set_context("paper", rc={"font.size":10,
                        "axes.titlesize":9, "axes.labelsize":12})
        fig, ax = plt.subplots()
    except ImportError:
        fig, ax = plt.subplots()
        plt.grid()

    plt.scatter(x=freqs, y=nvml_power, label='NVML measurements')
    plt.scatter(x=ridge_frequency, y=P_ridge, color='g',
                label='Ridge frequency (MHz)')
    plt.plot(xs, modelled_power, label='Modelled power consumption')
    ax.axvspan(min_freq, max_freq, alpha=0.15, color='green',
               label='Recommended frequency range')
    plt.title('GPU modelled power consumption', size=18)
    plt.xlabel('Core frequency (MHz)')
    plt.ylabel('Power consumption (W)')
    plt.legend()
    plt.show()

    plt.savefig("GPU_power_consumption_model.pdf")
