import inspect
import ast
import copy
import uuid
import sys

import astor
import tempfile
import importlib.util

from typing import Any

from kernel_tuner.language import Language
from kernel_tuner.kernel_sources.kernel_source import KernelSource
from kernel_tuner.kernel_sources.model.prepared_kernel_source_data import PreparedKernelSourceData



class KernelSourceFn(KernelSource):

    def __init__(self, kernel_name, kernel_source, lang, defines=None, call_function=None, decorator=None):
        super().__init__(kernel_name, kernel_source, lang, defines)
        if isinstance(kernel_source, list):
            raise ValueError("KernelSourceFn only supports a single kernel source function")

        try:
            self.lang = Language(lang.upper())
        except ValueError:
            raise TypeError(f"Supported languages are {[l.value for l in Language]}")
        
        if call_function is None:
            raise ValueError("call_function must be supplied for language Generic Python")
        if not callable(call_function):
            raise TypeError(f"call_function {call_function} is not a callable object.")
        self.call_function = call_function # TODO ceck signature

        if decorator:
            if not isinstance(decorator, str):
                raise TypeError(f"{decorator} is not a decorator")
            if decorator[0] != '@':
                raise ValueError(f"{decorator} is not a valid decorator")
        self.decorator = decorator 

        self.source_kernel_fn = kernel_source
        self.kernel_fn = self.source_kernel_fn
        self.signature = inspect.signature(kernel_source)
        try:
            self.source = inspect.getsource(kernel_source)
        except TypeError as e:
            raise TypeError(
                f"{e}. Did you forget to remove a decorator before tuning?"
            ) from e
        self.source_tree = ast.parse(self.source)
        self.import_nodes = self._find_import_nodes(inspect.getfile(kernel_source))

        # Find the module where the kernel function is defined
        self.module = inspect.getmodule(kernel_source)
        # Get dependencies by analyzing the AST
        if self.lang == Language.TRITON:
            self.dependencies = self._find_triton_dependencies()
        else:
            self.dependencies = self._find_dependencies()

        


    def _find_import_nodes(self, source_file):
        with open(source_file, "r") as f:
            tree = ast.parse(f.read(), filename=source_file)

        import_nodes = []
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_nodes.append(node)

        return import_nodes

    def _find_function_dependencies(self):
        """Find all function calls in the kernel."""
        class FunctionCallVisitor(ast.NodeVisitor):
            def __init__(self):
                self.called_functions = set()
                
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name):
                    self.called_functions.add(node.func.id)
                self.generic_visit(node)
        
        visitor = FunctionCallVisitor()
        visitor.visit(self.source_tree)
        return visitor.called_functions

    def _is_triton_jit_function(self, node):
        """Check if a function has the @triton.jit decorator. Is triton specific"""
        if not isinstance(node, ast.FunctionDef):
            return False
        
        for decorator in node.decorator_list:
            if (isinstance(decorator, ast.Name) and decorator.id == 'jit' or
                isinstance(decorator, ast.Attribute) and decorator.attr == 'jit' or
                isinstance(decorator, ast.Call) and 
                isinstance(decorator.func, ast.Attribute) and 
                decorator.func.attr == 'jit'):
                return True
        return False

    def _find_triton_dependencies(self):
        """Find all Triton JIT functions that are called by the kernel. Is Triton specific"""
        dependencies = {}
        called_functions = self._find_function_dependencies()
        
        # Parse the entire module to find Triton JIT functions
        module_source = inspect.getsource(self.module)
        module_tree = ast.parse(module_source)
        
        for node in module_tree.body:
            if (self._is_triton_jit_function(node) and 
                node.name in called_functions and 
                node.name != self.kernel_name):  # Skip the main kernel itself
                dependencies[node.name] = node
                
        return dependencies

    def _find_function_dependecies2(self, tree, local_funcs):
        # Non triton specific
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

    def _find_local_functions(self, tree):
        local_funcs = set()
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                local_funcs.add(node.name)
        return local_funcs

    def _find_dependencies(self):
        source_file = inspect.getfile(self.source_kernel_fn)
        with open(source_file, "r") as f: 
            source_code = f.read() 
        tree = ast.parse(source_code, filename=source_file) 
        local_funcs = self._find_local_functions(tree)
        called_functions = self._find_function_dependecies2(self.source_tree, local_funcs)

        dependencies = {}
        for node in tree.body:
            if (isinstance(node, ast.FunctionDef) and node.name in called_functions and 
                node.name != self.kernel_name):  # Skip the main kernel itself
                dependencies[node.name] = node

        return dependencies

    def _add_decorator(self, function):
        pass
    
    def prepare_kernel_instance(self, kernel_options, params, grid, threads):
        new_kernel_fn, temp_file_path = self.apply_params_to_source_fn(params)
        self.kernel_fn = new_kernel_fn

        return PreparedKernelSourceData(
            temp_files=[temp_file_path],
            kernel_name=self.kernel_name,
            kernel_fn=new_kernel_fn,
            kernel_str=None
        )

    def check_argument_lists(self, kernel_name, arguments):
        return True

    def apply_params_to_source_fn(self, params):
        transformer = ReplaceVars(params)
        
        # Create a new module with all necessary functions
        new_module = ast.Module(body=[], type_ignores=[])
        
        # Add imports
        new_module.body.extend(self.import_nodes)
        
        # Add dependencies (functions)
        for dep_node in self.dependencies.values():
            dep_node_copy = copy.deepcopy(dep_node)
            transformed_dep_node = transformer.visit(dep_node_copy)
            #new_module.body.append(copy.deepcopy(dep_node))
            new_module.body.append(transformed_dep_node)
        
        # Add transformed main kernel
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

        new_module.body.extend(transformed_tree.body)
        
        # Fix locations and generate source
        ast.fix_missing_locations(new_module)
        new_source = astor.to_source(new_module)

        #print(new_source)

        # Create a unique module name
        module_name = f'temp_kernel_module_{uuid.uuid4().hex}'
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(new_source)
            temp_file_path = temp_file.name

        spec = importlib.util.spec_from_file_location(module_name, temp_file_path)
        temp_module = importlib.util.module_from_spec(spec)
        
        # Register the module in sys.modules before executing it
        sys.modules[module_name] = temp_module
        spec.loader.exec_module(temp_module)
        new_fn = getattr(temp_module, self.kernel_name)

        return new_fn, temp_file_path

    def __del__(self):
        # Clean up temporary modules when the instance is destroyed
        for key in list(sys.modules.keys()):
            if key.startswith('temp_kernel_module_'):
                del sys.modules[key]


class ReplaceVars(ast.NodeTransformer):

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
            and isinstance(node.ctx, ast.Load)  # <- check context
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

    

