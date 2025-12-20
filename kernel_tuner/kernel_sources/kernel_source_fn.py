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

    def __init__(self, kernel_name, kernel_source, lang, defines=None):
        super().__init__(kernel_name, kernel_source, lang, defines)
        if isinstance(kernel_source, list):
            raise ValueError("KernelSourceFn only supports a single kernel source function")


        try:
            self.lang = Language(lang)
        except ValueError:
            raise TypeError(f"Supported languages are {[l.value for l in Language]}")
        self.source_kernel_fn = kernel_source
        self.kernel_fn = self.source_kernel_fn
        self.source = inspect.getsource(kernel_source)
        self.source_tree = ast.parse(self.source)
        if self.lang == Language.TRITON:
            self.import_nodes = [
                ast.Import(names=[ast.alias(name='triton', asname=None)]),
                ast.ImportFrom(
                    module='triton',
                    names=[ast.alias(name='language', asname='tl')],
                    level=0
                )
            ]
        else:
            self.import_nodes = [n for n in self.source_tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]


        # Find the module where the kernel function is defined
        self.module = inspect.getmodule(kernel_source)
        # Get dependencies by analyzing the AST
        if self.lang == Language.lang:
            self.dependencies = self._find_triton_dependencies()
        else:
            self.dependencies = None # TODO

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
        
        if self.lang == Language.TRITON:
            # Add all Triton JIT dependencies first
            for dep_node in self.dependencies.values():
                new_module.body.append(copy.deepcopy(dep_node))

        # TODO the kernel can not have dependencies yet. Obviously this needs fixing.
        
        # Add transformed main kernel
        source_tree_copy = copy.deepcopy(self.source_tree)
        transformed_tree = transformer.visit(source_tree_copy)
        new_module.body.extend(transformed_tree.body)
        
        # Fix locations and generate source
        ast.fix_missing_locations(new_module)
        new_source = astor.to_source(new_module)

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