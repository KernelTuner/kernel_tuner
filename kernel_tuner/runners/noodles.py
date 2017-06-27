#module private functions
import subprocess

from collections import OrderedDict

from noodles import schedule_hint, gather, lift
from noodles.run.runners import run_parallel_with_display, run_parallel
from noodles.display import NCDisplay
from noodles.interface import AnnotatedValue

from kernel_tuner.util import get_instance_string
from kernel_tuner.core import DeviceInterface

class RefCopy:
    def __init__(self, obj):
        self.obj = obj

    def __deepcopy__(self, _):
        return self.obj


class NoodlesRunner:
    def run(self, kernel_name, original_kernel, problem_size, arguments,
            tune_params, parameter_space, grid_div,
            answer, atol, verbose,
            lang, device, platform, cmem_args, compiler_options=None, quiet=False, iterations=7, sample_fraction=None):
        """ Iterate through the entire parameter space using a multiple Python processes

        :param kernel_name: The name of the kernel in the code.
        :type kernel_name: string

        :param original_kernel: The CUDA, OpenCL, or C kernel code as a string.
        :type original_kernel: string

        :param problem_size: See kernel_tuner.tune_kernel
        :type problem_size: tuple(int or string, int or string)

        :param arguments: A list of kernel arguments, use numpy arrays for
                arrays, use numpy.int32 or numpy.float32 for scalars.
        :type arguments: list

        :param tune_params: See kernel_tuner.tune_kernel
        :type tune_params: dict( string : [int, int, ...] )

        :param parameter_space: A list of lists that contains the entire parameter space
                to be searched. Each list in the list represents a single combination
                of parameters, order is imported and it determined by the order in tune_params.
        :type parameter_space: list( list() )

        :param grid_div_x: See kernel_tuner.tune_kernel
        :type grid_div_x: list

        :param grid_div_y: See kernel_tuner.tune_kernel
        :type grid_div_y: list

        :param answer: See kernel_tuner.tune_kernel
        :type answer: list

        :param atol: See kernel_tuner.tune_kernel
        :type atol: float

        :param verbose: See kernel_tuner.tune_kernel
        :type verbose: boolean

        :param lang: See kernel_tuner.tune_kernel
        :type lang: string

        :param device: See kernel_tuner.tune_kernel
        :type device: int

        :param platform: See kernel_tuner.tune_kernel
        :type device: int

        :param cmem_args: See kernel_tuner.tune_kernel
        :type cmem_args: dict(string: numpy object)

        :returns: A dictionary of all executed kernel configurations and their
            execution times.
        :rtype: dict( string, float )
        """
        workflow = self._parameter_sweep(lang, device, arguments,
                                         RefCopy(cmem_args), RefCopy(answer),
                                         RefCopy(tune_params),
                                         RefCopy(parameter_space),
                                         problem_size, grid_div,
                                         original_kernel, kernel_name, atol,
                                         platform, compiler_options, iterations)

        if verbose:
            with NCDisplay(self.error_filter) as display:
                answer = run_parallel_with_display(workflow, self.max_threads, display)
        else:
            answer = run_parallel(workflow, self.max_threads)

        if answer is None:
            print("Tuning did not return any results, did an error occur?")
            return None

        # Filter out null times
        answer = [d for d in answer if d['time']]

        return answer, {}

    def __init__(self, max_threads=1):
        self._max_threads = max_threads


    @property
    def max_threads(self):
        return self._max_threads


    @max_threads.setter
    def set_max_threads(self, max_threads):
        self._max_threads = max_threads


    def my_registry(self):
        return serial.pickle() + serial.base()


    def error_filter(self, errortype, value=None, tb=None):
        if errortype is subprocess.CalledProcessError:
            return value.stderr
        elif "cuCtxSynchronize" in str(value):
            return value
        return None


    @schedule_hint(display="Batching ... ",
                   ignore_error=True,
                   confirm=True)
    def _parameter_sweep(self, lang, device, arguments, cmem_args,
                         answer, tune_params, parameter_space, problem_size,
                         grid_div, original_kernel, kernel_name,
                         atol, platform, compiler_options, iterations):
        results = []
        for element in parameter_space:
            params = dict(OrderedDict(zip(tune_params.keys(), element)))

            instance_string = get_instance_string(params)

            time = self.run_single(lang, device, kernel_name, original_kernel, params,
                                   problem_size, grid_div,
                                   cmem_args, answer, atol, instance_string,
                                   platform, arguments, compiler_options, iterations)

            params['time'] = time
            results.append(lift(params))

        return gather(*results)


    @schedule_hint(display="Testing {instance_string} ... ",
                   ignore_error=True,
                   confirm=True)
    def run_single(self, lang, device, kernel_name, original_kernel, params,
                   problem_size, grid_div, cmem_args, answer,
                   atol, instance_string, platform, arguments,
                   compiler_options, iterations):

        #detect language and create high-level device interface
        dev = DeviceInterface(device, platform, original_kernel, lang=lang, compiler_options=compiler_options, iterations=iterations)

        #move data to the GPU
        gpu_args = dev.ready_argument_list(arguments)

        try:
            time = dev.compile_and_benchmark(gpu_args, kernel_name, original_kernel, params,
                                             problem_size, grid_div,
                                             cmem_args, answer, atol, False)

            return AnnotatedValue(time, None)
        except Exception as e:
            return AnnotatedValue(None, str(e))
