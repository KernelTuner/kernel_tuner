#!/usr/bin/env python
"""This is the minimal example from the README"""

import re
import numpy
from collections import OrderedDict

import kernel_tuner
from kernel_tuner import tune_kernel, run_kernel
from kernel_tuner import util
from kernel_tuner.file_utils import store_output_file, store_metadata_file
from kernel_tuner.core import KernelSource, DeviceInterface
from kernel_tuner.interface import (_kernel_options, _device_options,
                                    _check_user_input, Options)
#from chatgpt_queries import *
import openai
import datetime
import logging


logging.basicConfig(filename='logs/log{}.log'.format(datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S_%f')), level=logging.DEBUG)

logger = logging.getLogger(__name__)

#hdlr = logging.FileHandler('logs/log{}.log'.format(datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S_%f')))
#formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
#hdlr.setFormatter(formatter)
#logger.addHandler(hdlr)
#logger.setLevel(logging.DEBUG)

chatGPT_queries = {
    "start": lambda kernel_string:
        f"""
        I will give you a simple CUDA kernel as a starting point and will ask you to rewrite the code in some way.
        Do not change the name or the argument list of the kernel.
        First write "START", then write the rewritten kernel and write "END" straight after the kernel.
        This is the kernel we will be working with:
        START
        {kernel_string}
        END
        """,
    "vary_work_per_thread_x" : {
        "Initial_2elem_x": lambda kernel_string:
            """Rewrite this kernel such that each thread processes 2 elements in the x dimension instead of one.""",
             #  Make sure that each thread works on 2 adjacent elements, rewrite the calculation that uses threadIdx and blockIdx accordingly.
             #  Do not change the data types of variables.
        "Tunable_nr_elem_x": lambda:
            f"""
Please rewrite the code to allow the number of elements processed by each thread in the x-dimension to vary using a for loop.
This means we need to change the calculation that uses the threadIdx and blockIdx to account for this change.
This number is specified using a C preprocessor defined constant.
Please call this constant 'tile_size_x' in lower case.
Omit the definition of the 'tile_size_x' constant.
Make sure processed elements beyond the first adhere to the same bound checks.
            """,
# Make sure that threads work on elements that are blockDim.x apart.

    },
    "Incorrect_kernel": lambda error:
        f"""
        This kernel does not produce the correct result. I received the following error:
        {error}
        Can you try again?
        """,
    "Markdown_response": lambda:
        f"""
        Thank you, the response is in markdown, can you make it a code block?
        """,
    "Fails_to_compile": lambda error:
        f"""
This kernel is not correct, I get the error:
{error}
Can you try again?
         """,
}


class ChatBot:
    def __init__(self, system="", temperature=0.0, verbose=True):
        r"""
        Temperature is between 0 and 2.
        """
        self.system = system
        self.messages = []
        self.temperature = temperature
        self.verbose = verbose
        self.model = "gpt-3.5-turbo"
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message):
        if self.verbose:
            print("Query to ChatGPT:")
            print(message)
            print()
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        if self.verbose:
            print("Response given by ChatGPT:")
            print(result)
            print()
        return result

    def execute(self):
        logger.debug("Doing another query:")
        logger.debug(f"Using model {self.model}")
        logger.debug(f"Using temperature {self.temperature}")
        for msg in self.messages:
            logger.debug("Role: "+msg["role"])
            logger.debug("content: "+msg["content"])
        completion = openai.ChatCompletion.create(model=self.model,
                                                  messages=self.messages,
                                                  temperature=self.temperature)
        # Uncomment this to print out token usage each time, e.g.
        # {"completion_tokens": 86, "prompt_tokens": 26, "total_tokens": 112}
        #print(completion.usage)
        logger.debug("Response: "+completion.choices[0].message.content)
        return completion.choices[0].message.content


class ChatGPTuner:
    def __init__(self,
                 kernel_name,
                 start_kernel,
                 size,
                 args,
                 tune_params,
                 compiler_options=None,
                 max_turns=5,
                 verbose=True,
                 answer=None,
                 prompt=None,
                 temperature=0.1):
        self.kernel_name = kernel_name
        self.starting_kernel = start_kernel
        self.size = size
        self.args = args
        self.tune_params = tune_params
        self.compiler_options = compiler_options
        self.prompt = prompt
        self.max_turns = max_turns
        self.verbose = verbose

        self.inputs = []
        self.answers = []

        if not prompt:
            prompt = """You are a helpful assistant that rewrites CUDA code."""

        # Set up the chat bot
        if self.verbose:
            print("Setting up ChatGPTuner...")
        self.bot = ChatBot(self.prompt, temperature=temperature, verbose=verbose)

        self.started = False
        self.start_query = chatGPT_queries['start'](start_kernel)

        # Obtain kernel desired output for correctness checking:
        if not answer:
            if self.verbose:
                print("Calculating validation answer...")
            self.answer = run_kernel(self.kernel_name,
                                     self.starting_kernel,
                                     self.size,
                                     self.args,
                                     self.tune_params,
                                     compiler_options=self.compiler_options)
            for i, x in enumerate(self.answer):
                if not isinstance(x, numpy.ndarray):
                    self.answer[i] = None
        else:
            self.answer = answer

    def vary_work_per_thread_x(self):
        self.initial_2elem_x()
        response_kernel, tune_pars = self.tunable_nr_elem_x()
        return response_kernel, tune_pars

    def tunable_nr_elem_x(self):
        query = chatGPT_queries['vary_work_per_thread_x']['Tunable_nr_elem_x']()
        tune_pars = self.tune_params.copy()
        tune_pars['tile_size_x'] = [1, 2, 4]

        response_kernel = self.query_kernel(query)

        if "#define tile_size_x" in response_kernel:
            response_kernel = "\n".join([s for s in response_kernel.split('\n') if "#define tile_size_x" not in s])

        correct, response_kernel = self.check_correctness(response_kernel,
                                         tune_pars,
                                         grid_divs=['block_size_x','tile_size_x'])

        if correct:
            return response_kernel, tune_pars
        else:
            print("Was not able to produce a correct tunable kernel.")
            return "", None

    def initial_2elem_x(self):
        query = chatGPT_queries['vary_work_per_thread_x']['Initial_2elem_x'](self.starting_kernel)
        tune_pars = self.tune_params.copy()
        tune_pars['tile_size_x'] = 2

        response_kernel = self.query_kernel(query)

        correct, response_kernel = self.check_correctness(response_kernel,
                                         tune_pars,
                                         grid_divs=['block_size_x','tile_size_x'])
        if correct:
            return response_kernel, tune_pars
        else:
            print("Was not able to produce a correct tunable kernel.")
            return "", None

    def query_kernel(self, query):

        # On the first call to this method, we check if ChatGPT has been started already,
        # if not, we prepend the query with the start_query that includes the original kernel
        if not self.started:
            query = self.start_query + query
            self.started = True

        response = self.bot(query)

        # despite instructions to omit the definition of tile_size_x ChatGPT is very persistent to insert it
        try:
            response_kernel = self.extract_kernel_string(response)
        except Exception as e:
            if self.verbose:
                print("Failed to extract Kernel code from:")
                print(response)
                print()

            query = str(e) + ' Remember to first write "START", then write the rewritten kernel and write "END" straight after the kernel.'
            if self.verbose:
                print("New query to ChatGPT:")
                print(query)
                print()
            response = self.bot(query)
            try:
                response_kernel = self.extract_kernel_string(response)
            except Exception as e:
                print("Again failed to extract Kernel code from:")
                print(response)
                raise Exception("ChatGPT does not want to listen, cannot extract kernel string.")

        return response_kernel

    def check_correctness(self, response_kernel, tune_pars, grid_divs=None, tries=3):
        correct = False
        iter_count = 0
        while not correct:
            iter_count += 1
            if self.verbose:
                print(f"Testing chatGPT response {iter_count}")

            # TODO: How to deal with grid_div_x
            correct, err = self.validate_kernel(self.kernel_name,
                                           response_kernel,
                                           self.size,
                                           self.args,
                                           tune_pars,
                                           grid_div_x=grid_divs,
                                           compiler_options=self.compiler_options)

            print("Kernel is correct is", correct)
            if not correct:
                print("There was an error in compiling and running the kernel.")
                if isinstance(err, util.CompilerError):
                    query = chatGPT_queries['Fails_to_compile'](str(err))
                elif isinstance(err, util.VerificationError):
                    query = chatGPT_queries['Incorrect_kernel'](str(err))
                else:
                    print("Unrecognized Exception")
                    raise err

                # Break the while loop if conditions are met
                if iter_count >= tries:
                    print("Failed to obtain a valid kernel from chatGPT...")
                    break

                response_kernel = self.query_kernel(query)

        return correct, response_kernel

    def extract_kernel_string(self, response):
        if 'START' not in response:
            raise Exception(f"Could not extract kernel code from response.")
        if 'END' not in response:
            raise Exception(f"Could not extract kernel code from response.")
        x = response.split('START')[1]
        x = x.split('END')[0]
        return x

    def validate_kernel(self, kernel_name, kernel_string, size, args, tune_params,
                        compiler_options=None, **kwargs):
        # Does the kernel code compile and run
        try:
            tune_pars = tune_params.copy()
            tune_pars.update({k:[v] for k,v in tune_params.items() if not isinstance(v,list)})
            output = tune_kernel(kernel_name,
                       kernel_string,
                       size,
                       args,
                       tune_pars,
                       answer=self.answer,
                       verbose=True,
                       compiler_options=compiler_options,
                       **kwargs)
            return True, None
        except Exception as e:
            print(e)
            return False, e



"""
CUDA_type_converter = {
    'int': int,
    'float': float,
    'bool': bool,
    'int*': 'numpy.int',
    'float*': numpy.float32,
    'bool*': 'numpy.bool',
}


def verify_kernel_string(kernel_name,
                         kernel_source,
                         problem_size,
                         arguments,
                         params,
                         device=0,
                         platform=0,
                         quiet=False,
                         compiler=None,
                         compiler_options=None,
                         block_size_names=None,
                         lang=None,
                         defines=None):
    grid_div_x=None
    grid_div_y=None
    grid_div_z=None
    smem_args=None
    cmem_args=None
    texmem_args=None

    # DeviceInterface has a compile_and_benchmark function
    kernelsource = KernelSource(kernel_name, kernel_source, lang, defines)

    _check_user_input(kernel_name, kernelsource, arguments, block_size_names)

    #sort options into separate dicts
    opts = locals()
    kernel_options = Options([(k, opts[k]) for k in _kernel_options.keys()])
    device_options = Options([(k, opts[k]) for k in _device_options.keys()])

    #detect language and create the right device function interface
    dev = DeviceInterface(kernelsource, iterations=1, **device_options)

    #TODO: MAKE PROPER TRY EXCEPT ETC>
    instance = None
    instance = dev.create_kernel_instance(kernelsource,
                                          kernel_options,
                                          params,
                                          True)
    if instance is None:
        raise RuntimeError("cannot create kernel instance,"+
                           " too many threads per block")
    # see if the kernel arguments have correct type
    kernel_tuner.util.check_argument_list(instance.name,
                                          instance.kernel_string,
                                          arguments)

    #compile the kernel
    func = dev.compile_kernel(instance, True)
    if func is None:
        raise RuntimeError("cannot compile kernel"+
                           " too much shared memory used")
    return kernel_options, device_options


def parse_function_sign(kernel_string, size=128):
    ###  Parse kernel name
    kernel_name = kernel_string.split("(")[0].split(" ")[-1]

    ###  Parse arguments to function, and create default python args
    args_str = kernel_string.split("(")[1].split(")")[0]
    args_str = args_str.replace(',', '').split(" ")

    # Detect 2D/3D cases:
    is_2d = False
    is_3d = False
    for arg in args_str:
        if '.y' in arg:
            is_2d = True
        if '.z' in arg:
            is_3d = True
    if is_3d:
        is_2d = False

    args = dict()
    for k in range(0, len(args_str), 2):# Steps of 2
        CUDA_type_str = args_str[k]
        var_name = args_str[k+1]
        # Sometimes chatGPT/people put the * before the variable name
        if var_name[0] == '*':
            var_name = var_name[1:]
            CUDA_type_str += "*"

        python_type = CUDA_type_converter[CUDA_type_str]
        if 'numpy' in str(python_type):# Is an array in this case
            psize = size
            if is_2d:
                psize = (size, size)
            if is_3d:
                psize = (size, size, size)

            if str(python_type) == 'numpy.int':
                args[var_name] = numpy.random.randn(psize).astype(int)
            elif str(python_type) == 'numpy.bool':
                args[var_name] = numpy.random.randn(psize).astype(bool)
            else:
                args[var_name] = numpy.random.randn(psize).astype(python_type)
        elif CUDA_type_str == 'int':
            # Guessing that we can assign 'size' to this int
            # This is because almost all kernels have some int param
            # that regulates the size of the for loop/array.
            value = numpy.int32(size)
            #value = python_type.__new__(python_type)
            #value = size
            args[var_name] = value
        else:
            value = python_type.__new__(python_type)
            args[var_name] = value

    ###  Parse tunable params
    #Select only those lines with dots in them
    tune_params_str = [x for x in kernel_string.split("\n") if '.' in x]
    # Sometimes I ask chatGPT to include TUNE in the variable names for tunable params
    tune_params_str += [x for x in kernel_string.split("\n") if 'TUNE' in x]
    #TODO: This is not enough to find all the lines!

    # Clean it up
    tune_strs =  [x.replace("+", '') for x in tune_params_str]
    tune_strs =  [x.replace("-", '') for x in tune_strs]
    tune_strs =  [x.replace("=", '') for x in tune_strs]
    tune_strs =  [x.replace("*", '') for x in tune_strs]
    tune_strs =  [x.replace(";", '') for x in tune_strs]
    tune_strs =  [x.replace(",", '') for x in tune_strs]

    # Select potential candidates for tunable params
    # TODO: This will be hard to do accurately
    candidates = [y for x in tune_strs for y in x.split(" ")]
    #candidates = [x for x in candidates if '.' in x]
    candidates = [x for x in candidates if len(x) > 7]

    # Remove those with 'Idx' in the name because they are CUDA idx variables
    candidates = [x for x in candidates if 'Idx' not in x]

    # Dots in names are not "valid identifiers", so we replace them
    valid_cands = [x.replace(".", "") for x in candidates]
    valid_cands = [x for x in valid_cands if x.isidentifier()]
    valid_cands = [x for x in valid_cands if '__' not in x]
    for i in range(len(valid_cands)):
        kernel_string = kernel_string.replace(candidates[i], valid_cands[i])

    tune_params = dict()
    for cand in valid_cands:
        tune_params[cand] = 1

    ## Remove double argument and tune_params
    #for param in tune_params.keys():
    #    if param in args.keys():
    #        del args[param]
    return kernel_string, kernel_name, list(args.values()), tune_params


def setup_var_defs(kernel_options, tune_params):
    grid_div = (kernel_options.grid_div_x,
                kernel_options.grid_div_y,
                kernel_options.grid_div_z)

    threads, grid = kernel_tuner.util.setup_block_and_grid(
                        kernel_options.problem_size,
                        grid_div,
                        tune_params,
                        kernel_options.block_size_names)

    defines = OrderedDict()
    grid_dim_names = ["grid_size_x", "grid_size_y", "grid_size_z"]
    for i, g in enumerate(grid):
        defines[grid_dim_names[i]] = g
    for i, g in enumerate(threads):
        defines[kernel_options.block_size_names[i]] = g
    for k, v in tune_params.items():
        defines[k] = 256 # <--- again, how to set this in general?
    defines["kernel_tuner"] = 1
    return defines


def validate_kernel(kernel_string, size, compiler_options=None):
    # Parse as much as possible from the ChatGPT kernel string
    kernel_string, kernel_name, args, tune_params = parse_function_sign(
                                                         kernel_string,
                                                         size=size)

    # Verify if this kernel string compiles
    kernel_options, device_options = verify_kernel_string(kernel_name,
                         kernel_string,
                         size,
                         args,
                         tune_params,
                         compiler_options=compiler_options)
                         #compiler_options=['-Wno-deprecated-gpu-targets'])

    # Setup the variables for pre-definition
    defines = setup_var_defs(kernel_options, tune_params)

    # Run kernel
    run_kernel(kernel_name,
               kernel_string,
               size,
               args,
               tune_params,
               defines=defines,
               compiler_options=compiler_options)
    return kernel_name, kernel_string, size, args, tune_params, defines
"""
