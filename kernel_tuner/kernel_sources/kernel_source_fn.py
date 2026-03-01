import inspect
import ast
import copy
import uuid
import sys
import logging

import astor
import tempfile
import importlib.util

from typing import Any

from kernel_tuner.language import Language
from kernel_tuner.kernel_sources.kernel_source import KernelSource
from kernel_tuner.kernel_sources.model.prepared_kernel_source_data import PreparedKernelSourceData



class KernelSourceFn(KernelSource):
    """
    Class that holds the Python-function-based kernel sources.

    There is a primary kernel source for function-based kernels in Python. The source must be 
    a callable. This can be a function of a class. The function should not be decorated, but
    a decorator can be supplied to wrap the function. 
    
    A call function to specify how the kernel should be launched must be supplied. The call 
    function must take the following arguments:
    - kernel_function: the callable function with the tuning parameters inserted. If provided, the 
      kernel_function is decorated with the de decorator. 
    - args: list of kernel arguments, as provided by the user in the <args> argument.
    - kwargs: dictionary of kernel keyword arguments. If a tuning parameter is in the kernel signature, 
      the tuning parameter will be added as a keyword argument.
    - grid: the launch grid (tuple with 3 values), as computed by KernelTuner
    - threads: the thread block size (tuple with 3 values), as computed by KernelTuner
    - params: dictionary with the values of the tuning params for this configuration.
    """

    def __init__(self, kernel_name, kernel_source, lang, defines=None, call_function=None, decorator=None):
        super().__init__(kernel_name, kernel_source, lang, defines)
        if isinstance(kernel_source, list):
            raise ValueError("KernelSourceFn only supports a single kernel source function")
       
        if self.lang == Language.GENERIC_PYTHON: 
            if call_function is None:
                raise ValueError("call_function must be supplied for language Generic Python")
            if not callable(call_function):
                raise TypeError(f"call_function of type {type(call_function)} is not a callable object.")
        self.call_function = call_function # TODO ceck signature

        if decorator:
            if not isinstance(decorator, str):
                raise TypeError(f"The decorator should be a string, got {type(decorator)} instead.")
            if decorator[0] != '@':
                raise ValueError(f"The decorator should start with a '@', got {decorator} instead.")
        self.decorator = decorator 

        self.source_kernel_fn = kernel_source # This kernel source remains the original object
        self.kernel_fn = self.source_kernel_fn # This is the kernel source that we will modify
        
        try:
            self.source = inspect.getsource(self.source_kernel_fn)
        except TypeError as e:
            raise TypeError(
                f"{e}. Did you forget to remove a decorator before tuning?"
            ) from e
        
        self.signature = inspect.signature(self.source_kernel_fn)
        self.source_tree = ast.parse(self.source)
        self.import_nodes = self._find_import_nodes(inspect.getfile(self.source_kernel_fn))
        self.dependencies = self._find_dependencies()


    def prepare_kernel_instance(self, kernel_options, params, grid, threads):
        '''
        Given the dictionary of tuning parameter values for this configuration,
        generate a kernel instance with these tuning parameters inserted. kernel_options, 
        grid and threads are not needed for Python kernels.
        '''
        new_kernel_fn, temp_file_path = self.apply_params_to_source_fn(params)
        self.kernel_fn = new_kernel_fn

        return PreparedKernelSourceData(
            temp_files=[temp_file_path],
            kernel_name=self.kernel_name,
            kernel_fn=new_kernel_fn,
            kernel_str=None
        )


    def check_argument_lists(self, kernel_name, arguments):
        '''
        Check if the kernel arguments have the correct types.
        Not implemented for Python, because type hinting is not always supplied.
        '''
        logging.debug("Checking of arguments list not supported yet for Python kernels")
        return True


    def apply_params_to_source_fn(self, params):
        '''
        Create a module with the kernel imports and local dependencies from the kernel source file.
        Find instances of the tuning parameters in the kernel and local dependencies and replace these
        instances by the values of this configuration.
        Return the new kernel and the path to the new module, where the new kernel lives.
        '''
        transformer = ReplaceVars(params)
        
        # Create a new module to store the transformed AST in.
        new_module = ast.Module(body=[], type_ignores=[])
        
        # Add imports
        new_module.body.extend(self.import_nodes)
        
        # Add dependencies (functions)
        for dep_node in self.dependencies.values():
            dep_node_copy = copy.deepcopy(dep_node)
            transformed_dep_node = transformer.visit(dep_node_copy) # apply tuning params to dependencies
            new_module.body.append(transformed_dep_node)
        
        # Transform main kernel
        source_tree_copy = copy.deepcopy(self.source_tree)
        transformed_tree = transformer.visit(source_tree_copy)
        
        # Add decorator if needed
        if self.decorator:
            dummy = f"{self.decorator}\ndef _dummy():\n pass\n" 
            decorator_node = ast.parse(dummy).body[0].decorator_list[0]
            for node in transformed_tree.body:
                if isinstance(node, ast.FunctionDef):
                    node.decorator_list.insert(0, decorator_node)
                    break   # only apply to the top level function

        # Add transformed main kernel to new module
        new_module.body.extend(transformed_tree.body)
        
        # Fix locations and generate source
        ast.fix_missing_locations(new_module)
        new_source = astor.to_source(new_module)

        # Create a unique module name and write new source to it.
        module_name = f'temp_kernel_module_{uuid.uuid4().hex}'
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(new_source)
            temp_file_path = temp_file.name

        # Register the module in sys.modules before executing it
        spec = importlib.util.spec_from_file_location(module_name, temp_file_path)
        temp_module = importlib.util.module_from_spec(spec)       
        sys.modules[module_name] = temp_module
        spec.loader.exec_module(temp_module)
        new_fn = getattr(temp_module, self.kernel_name)

        return new_fn, temp_file_path

        
    def _find_import_nodes(self, source_file):
        '''
        Parse kernel source file to find import statements. Return those
        statements as AST import nodes. 
        '''
        with open(source_file, "r") as f:
            tree = ast.parse(f.read(), filename=source_file)

        import_nodes = []
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_nodes.append(node)

        return import_nodes


    def _find_function_dependecies(self, tree, local_funcs):
        '''
        Given an AST and a list of local function names, return the funtions that are invoked
        somewhere in the file, and are in the local_funcs list.
        '''
        class FunctionCallVisitor(ast.NodeVisitor):
            def __init__(self):
                self.called = set()

            def visit_Call(self, node):
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                    if name in local_funcs:
                        self.called.add(name)
                self.generic_visit(node)
        
        visitor = FunctionCallVisitor()
        visitor.visit(tree)
        return visitor.called


    def _find_dependencies(self):
        '''
        Find all local function dependencies in the file where the kernel source is
        defined. Return a dicionary indexed by the function node name with the function
        body as value.
        '''
        source_file = inspect.getfile(self.source_kernel_fn)
        with open(source_file, "r") as f: 
            source_code = f.read() 
        tree = ast.parse(source_code, filename=source_file) 

        # Find the locally defined functions in the source file
        local_funcs = set()
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                local_funcs.add(node.name)
        
        # Given source file AST and the set of locally defined functions, find the names of 
        # the local functions that are invoked somewhere in the AST.
        called_functions = self._find_function_dependecies(self.source_tree, local_funcs)

        # Traverse the tree one more time to save the dependency functions in a dictionary. 
        # Skip the main kernel.
        dependencies = {}
        for node in tree.body:
            if (isinstance(node, ast.FunctionDef) and node.name in called_functions and 
                node.name != self.kernel_name):  # Skip the main kernel itself
                dependencies[node.name] = node

        return dependencies


    def __del__(self):
        '''
        Clean up temporary modules when the instance is destroyed
        '''
        for key in list(sys.modules.keys()):
            if key.startswith('temp_kernel_module_'):
                del sys.modules[key]


class ReplaceVars(ast.NodeTransformer):
    '''
    AST transformer that replaces occurrences of tuning parameters with constant values.
    The tuning parameters (params) should be provided as a dictionary.
    The transformation supports the following cases:

    - Variable reads:
        occurrence of a variable name that matches a key in `params` are replaced if the 
        value is being read.

    - Object attribute reads:
        expressions of the form `self.<param>` are replaced with constants if
        `<param>` exists in `params` and is being read (not assigned to).

    - Assignments to parameters:
        assignments like `param = ...` or `self.param = ...` are overridden,
        replacing the right-hand side with the constant value from `params`.

    Examples:
        Given `params = {"block_size": 32}`:
        x = 2 * block_size          ->      x = 2 * 32
        x = 2 * self.block_size     ->      x = 2 * 32
        self.block_size = 64        ->      self.block_size = 32
        block_size = 64             ->      block_size = 32

    '''

    def __init__(self, params: dict):
        self.params = params

    def visit_Name(self, node: ast.Name) -> Any:
        if isinstance(node.ctx, ast.Load) and node.id in self.params.keys():
            return ast.copy_location(
                ast.Constant(value=self.params[node.id]),
                node
            )
        return node
    

    def visit_Attribute(self, node: ast.Attribute):
        self.generic_visit(node)
    
        # Replace self.<param> with constant, but only if it's being read
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and node.attr in self.params
            and isinstance(node.ctx, ast.Load)  
        ):
            return ast.copy_location(ast.Constant(self.params[node.attr]), node)
        
        return node


    def visit_Assign(self, node: ast.Assign):
        # Replace assignments like: mock_param = ...
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in self.params
        ):
            node.value = ast.Constant(value=self.params[node.targets[0].id])
            return node  

        # Replace assignment like self.<attr> = ...
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Attribute)
            and isinstance(node.targets[0].value, ast.Name)
            and node.targets[0].value.id == "self"
            and node.targets[0].attr in self.params
        ):
            node.value = ast.Constant(value=self.params[node.targets[0].attr])
            return node

        return self.generic_visit(node)

    

