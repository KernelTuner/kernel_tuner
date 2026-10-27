"""Functions to detect the programming language of a kernel, and the DSL used by Python kernels.

The DSL of a Python kernel is detected from the source code alone: the decorators (or base
classes) of the kernel are resolved using the imports in the source file. This way, none of
the DSLs, which are optional dependencies, have to be imported to detect them.
"""

import ast
from pathlib import Path

# Maps the fully qualified names of the decorators and base classes of Python kernels to their DSL.
# A name matches if it is equal to, or starts with, one of these prefixes.
PYTHON_DSL_PREFIXES = {
    "triton": "triton",
    "numba.cuda": "numba",
    "cupyx.jit": "cupyx",
    "warp": "warp",
    "taichi": "taichi",
    "cutlass.cute": "cute",
    "tilus": "tilus",
    "tilelang": "tilelang",
    "cuda.tile": "cutile",
}


def detect_language(kernel_string):
    """Attempt to detect language from the kernel_string."""
    kernel_string = kernel_string.lower().strip()
    if "__global__" in kernel_string:
        lang = "CUDA"
    elif "__kernel" in kernel_string:
        lang = "OpenCL"
    elif any(token in kernel_string for token in ["@cuda", "@kernel", "@device_code"]):
        lang = "Julia"
    else:
        lang = "C"
    return lang


def is_python_file(kernel_source):
    """Return True if kernel_source is the path to an existing Python file."""
    if not isinstance(kernel_source, (str, Path)):
        return False
    try:
        path = Path(kernel_source)
        return path.suffix == ".py" and path.is_file()
    except (OSError, ValueError):  # for example, source code strings that are too long to be a path
        return False


def _import_aliases(tree):
    """Map the names bound by the import statements in tree to the fully qualified names they refer to."""
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    aliases[alias.asname] = alias.name
                else:
                    # "import a.b" binds "a", attribute access then resolves the rest
                    root = alias.name.split(".")[0]
                    aliases[root] = root
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return aliases


def _qualified_name(node, aliases):
    """Return the fully qualified name of a decorator or base class expression, or None."""
    # decorators with arguments, such as @jit.rawkernel(), are calls of the decorator
    if isinstance(node, ast.Call):
        node = node.func
    # indexing, such as ct.Constant[int], is not a decorator or base class we recognize
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(aliases.get(node.id, node.id))
    return ".".join(reversed(parts))


def _dsl_of_name(name):
    """Return the DSL for a fully qualified name, using the longest matching prefix."""
    matches = [prefix for prefix in PYTHON_DSL_PREFIXES if name == prefix or name.startswith(prefix + ".")]
    if not matches:
        return None
    return PYTHON_DSL_PREFIXES[max(matches, key=len)]


def detect_python_dsl(kernel_name, filepath):
    """Detect the DSL of a Python kernel from its decorators or base classes.

    :param kernel_name: name of the kernel function or class
    :type kernel_name: string

    :param filepath: path to the Python file that contains the kernel
    :type filepath: string or Path

    :returns: the name of the DSL, one of the values of PYTHON_DSL_PREFIXES, or None if not detected
    :rtype: string or None
    """
    from kernel_tuner.util import get_kernel_ast, read_file

    kernel_ast = get_kernel_ast(kernel_name, filepath)
    aliases = _import_aliases(ast.parse(read_file(filepath), filename=str(filepath)))

    # class based kernels (Tilus) are recognized by their base class or the decorators of __call__
    if isinstance(kernel_ast, tuple):
        class_node, call_node = kernel_ast
        candidates = class_node.bases + class_node.decorator_list + call_node.decorator_list
    else:
        candidates = kernel_ast.decorator_list

    for node in candidates:
        name = _qualified_name(node, aliases)
        dsl = name and _dsl_of_name(name)
        if dsl:
            return dsl
    return None
