import numpy as np

from kernel_tuner import core
from kernel_tuner.interface import Options, _kernel_options

from kernel_tuner.integration import TuneResults

class PythonKernel(object):

    def __init__(self, kernel_name, kernel_string, problem_size, arguments, params=None, inputs=None, outputs=None, device=0, platform=0,
                 block_size_names=None, grid_div_x=None, grid_div_y=None, grid_div_z=None, verbose=True, lang=None,
                 results_file=None):
        """ Construct Python helper object to compile and call the kernel from Python

            This object compiles a GPU kernel parameterized using the parameters in params.
            GPU memory is allocated for each argument using its size and type as listed in arguments.
            The object can be called directly as a function with the kernel arguments as function arguments.
            Kernel arguments marked as inputs will be copied to the GPU on every kernel launch.
            Only the kernel arguments marked as outputs will be returned, note that the result is always
            returned in a list, even when there is only one output.

            Most of the arguments to this function are the same as with tune_kernel or run_kernel in Kernel Tuner,
            and are therefore not duplicated here. The two new arguments are:

            :param inputs: a boolean list of length arguments to signal whether an argument is input to the kernel
            :type inputs: list(bool)

            :param outputs: a boolean list of length arguments to signal whether an argument is output of the kernel
            :type outputs: list(bool)

        """
        #construct device interface
        kernel_source = core.KernelSource(kernel_name, kernel_string, lang)
        self.dev = core.DeviceInterface(kernel_source, device=device, quiet=True)
        if not params:
            params = {}

        #if results_file is passed use the results file to lookup tunable parameters
        if results_file:
            results = TuneResults(results_file)
            params.update(results.get_best_config(self.dev.name, problem_size))
        self.params = params

        #construct kernel_options to hold information about the kernel
        opts = locals()
        kernel_options = Options([(k, opts[k]) for k in _kernel_options.keys() if k in opts.keys()])

        #instantiate the kernel given the parameters in params
        self.kernel_instance = self.dev.create_kernel_instance(kernel_source, kernel_options, params, verbose)

        #compile the kernel
        self.func = self.dev.compile_kernel(self.kernel_instance, verbose)

        #setup GPU memory
        self.gpu_args = self.dev.ready_argument_list(arguments)
        if inputs:
            self.inputs = inputs
        else:
            self.inputs = [True for _ in arguments]
        if outputs:
            self.outputs = outputs
        else:
            self.outputs = [True for _ in arguments]

    def update_gpu_args(self, args):
        for i, arg in enumerate(args):
            if self.inputs[i]:
                if isinstance(args[i], np.ndarray):
                    self.dev.dev.memcpy_htod(self.gpu_args[i], arg)
                else:
                    self.gpu_args[i] = arg
        return self.gpu_args

    def get_gpu_result(self, args):
        results = []
        for i, _ in enumerate(self.gpu_args):
            if self.outputs[i] and isinstance(args[i], np.ndarray):
                res = np.zeros_like(args[i])
                self.dev.memcpy_dtoh(res, self.gpu_args[i])
                results.append(res)
        return results

    def run_kernel(self, args):
        """Run the GPU kernel

        Copy the arguments marked as inputs to the GPU
        Call the GPU kernel
        Copy the arguments marked as outputs from the GPU
        Return the outputs in a list of numpy arrays

        :param args: A list with the kernel arguments as numpy arrays or numpy scalars
        :type args: list(np.ndarray or np.generic)
        """
        self.update_gpu_args(args)
        self.dev.run_kernel(self.func, self.gpu_args, self.kernel_instance)
        return self.get_gpu_result(args)

    def __call__(self, *args):
        """Run the GPU kernel

        Copy the arguments marked as inputs to the GPU
        Call the GPU kernel
        Copy the arguments marked as outputs from the GPU
        Return the outputs in a list of numpy arrays

        :param *args: A variable number of kernel arguments as numpy arrays or numpy scalars
        :type *args: np.ndarray or np.generic
        """
        return self.run_kernel(args)

    def __del__(self):
        if hasattr(self, 'dev'):
            self.dev.__exit__([None, None, None])
