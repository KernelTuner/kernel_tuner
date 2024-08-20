from typing import Any, Tuple
from abc import ABC, abstractmethod
import numpy as np

# Function templates
cpp_template: str = """
<!?PREPROCESSOR?!>
<!?USER_DEFINES?!>
#include <chrono>

extern "C" <!?SIGNATURE?!> {
<!?INITIALIZATION?!>
<!?BODY?!>
<!?DEINITIALIZATION?!>
}
"""

f90_template: str = """
<!?PREPROCESSOR?!>
<!?USER_DEFINES?!>

module kt
use iso_c_binding
contains

<!?SIGNATURE?!>
<!?INITIALIZATION?!>
<!?BODY?!>
<!?DEINITIALIZATION?!>
end function <!?NAME?!>

end module kt
"""


class Directive(ABC):
    """Base class for all directives"""

    @abstractmethod
    def get(self) -> str:
        pass


class Language(ABC):
    """Base class for all languages"""

    @abstractmethod
    def get(self) -> str:
        pass


class OpenACC(Directive):
    """Class to represent OpenACC"""

    def get(self) -> str:
        return "openacc"


class Cxx(Language):
    """Class to represent C++ code"""

    def get(self) -> str:
        return "cxx"

    def end_string(self) -> str:
        return "#pragma tuner stop"


class Fortran(Language):
    """Class to represent Fortran code"""

    def get(self) -> str:
        return "fortran"

    def end_string(self) -> str:
        return "!$tuner stop"


class Code(object):
    """Class to represent the directive and host code of the application"""

    def __init__(self, directive: Directive, lang: Language):
        self.directive = directive
        self.language = lang


class ArraySize(object):
    """Size of an array"""

    def __init__(self):
        self.size = list()

    def __iter__(self):
        for i in self.size:
            yield i

    def __len__(self):
        return len(self.size)

    def clear(self):
        self.size.clear()

    def get(self) -> int:
        length = len(self.size)
        if length == 0:
            return 0
        elif length == 1:
            return self.size[0]
        else:
            product = 1
            for i in self.size:
                product *= i
            return product

    def add(self, dim: int) -> None:
        # Only allow adding valid dimensions
        if dim >= 1:
            self.size.append(dim)


def fortran_md_size(size: ArraySize) -> list:
    """Format a multidimensional size into the correct Fortran string"""
    md_size = list()
    for dim in size:
        md_size.append(f":{dim}")
    return md_size


def is_openacc(directive: Directive) -> bool:
    """Check if a directive is OpenACC"""
    return isinstance(directive, OpenACC)


def is_cxx(lang: Language) -> bool:
    """Check if language is C++"""
    return isinstance(lang, Cxx)


def is_fortran(lang: Language) -> bool:
    """Check if language is Fortran"""
    return isinstance(lang, Fortran)


def line_contains_openacc_directive(line: str, lang: Language) -> bool:
    """Check if line contains an OpenACC directive or not"""
    if is_cxx(lang):
        return line_contains_openacc_directive_cxx(line)
    elif is_fortran(lang):
        return line_contains_openacc_directive_fortran(line)
    return False


def line_contains_openacc_directive_cxx(line: str) -> bool:
    """Check if a line of code contains a C++ OpenACC directive or not"""
    return line_contains(line, "#pragma acc")


def line_contains_openacc_directive_fortran(line: str) -> bool:
    """Check if a line of code contains a Fortran OpenACC directive or not"""
    return line_contains(line, "!$acc")


def line_contains_openacc_parallel_directive(line: str, lang: Language) -> bool:
    """Check if line contains an OpenACC parallel directive or not"""
    if is_cxx(lang):
        return line_contains_openacc_parallel_directive_cxx(line)
    elif is_fortran(lang):
        return line_contains_openacc_parallel_directive_fortran(line)
    return False


def line_contains_openacc_parallel_directive_cxx(line: str) -> bool:
    """Check if a line of code contains a C++ OpenACC parallel directive or not"""
    return line_contains(line, "#pragma acc parallel")


def line_contains_openacc_parallel_directive_fortran(line: str) -> bool:
    """Check if a line of code contains a Fortran OpenACC parallel directive or not"""
    return line_contains(line, "!$acc parallel")


def line_contains(line: str, target: str) -> bool:
    """Generic helper to check if a line contains the target"""
    return target in line


def openacc_directive_contains_clause(line: str, clauses: list) -> bool:
    """Check if an OpenACC directive contains one clause from a list"""
    for clause in clauses:
        if clause in line:
            return True
    return False


def openacc_directive_contains_data_clause(line: str) -> bool:
    """Check if an OpenACC directive contains one data clause"""
    data_clauses = ["copy", "copyin", "copyout", "create", "no_create", "present", "device_ptr", "attach"]
    return openacc_directive_contains_clause(line, data_clauses)


def create_data_directive_openacc(name: str, size: ArraySize, lang: Language) -> str:
    """Create a data directive for a given language"""
    if is_cxx(lang):
        return create_data_directive_openacc_cxx(name, size)
    elif is_fortran(lang):
        return create_data_directive_openacc_fortran(name, size)
    return ""


def create_data_directive_openacc_cxx(name: str, size: ArraySize) -> str:
    """Create C++ OpenACC code to allocate and copy data"""
    return f"#pragma acc enter data create({name}[:{size.get()}])\n#pragma acc update device({name}[:{size.get()}])\n"


def create_data_directive_openacc_fortran(name: str, size: ArraySize) -> str:
    """Create Fortran OpenACC code to allocate and copy data"""
    if len(size) == 1:
        return f"!$acc enter data create({name}(:{size.get()}))\n!$acc update device({name}(:{size.get()}))\n"
    else:
        md_size = fortran_md_size(size)
        return (
            f"!$acc enter data create({name}({','.join(md_size)}))\n!$acc update device({name}({','.join(md_size)}))\n"
        )


def exit_data_directive_openacc(name: str, size: ArraySize, lang: Language) -> str:
    """Create code to copy data back for a given language"""
    if is_cxx(lang):
        return exit_data_directive_openacc_cxx(name, size)
    elif is_fortran(lang):
        return exit_data_directive_openacc_fortran(name, size)
    return ""


def exit_data_directive_openacc_cxx(name: str, size: ArraySize) -> str:
    """Create C++ OpenACC code to copy back data"""
    return f"#pragma acc exit data copyout({name}[:{size.get()}])\n"


def exit_data_directive_openacc_fortran(name: str, size: ArraySize) -> str:
    """Create Fortran OpenACC code to copy back data"""
    if len(size) == 1:
        return f"!$acc exit data copyout({name}(:{size.get()}))\n"
    else:
        md_size = fortran_md_size(size)
        return f"!$acc exit data copyout({name}({','.join(md_size)}))\n"


def correct_kernel(kernel_name: str, line: str) -> bool:
    """Checks if the line contains the correct kernel name"""
    return f" {kernel_name} " in line or (kernel_name in line and len(line.partition(kernel_name)[2]) == 0)


def find_size_in_preprocessor(dimension: str, preprocessor: list) -> int:
    """Find the dimension of a directive defined value in the preprocessor"""
    ret_size = 0
    for line in preprocessor:
        if f"#define {dimension}" in line:
            try:
                ret_size = int(line.split(" ")[2])
                break
            except ValueError:
                continue
    return ret_size


def extract_code(start: str, stop: str, code: str, langs: Code, kernel_name: str = None) -> dict:
    """Extract an arbitrary section of code"""
    found_section = False
    sections = dict()
    tmp_string = list()
    name = ""
    init_found = 0

    for line in code.replace("\\\n", "").split("\n"):
        if found_section:
            if stop in line:
                found_section = False
                sections[name] = "\n".join(tmp_string)
                tmp_string = list()
                name = ""
            else:
                tmp_string.append(line)
        else:
            if start in line:
                if kernel_name is None or correct_kernel(kernel_name, line):
                    found_section = True
                    try:
                        if is_cxx(langs.language):
                            name = line.strip().split(" ")[3]
                        elif is_fortran(langs.language):
                            name = line.strip().split(" ")[2]
                    except IndexError:
                        name = f"init_{init_found}"
                        init_found += 1

    return sections


def parse_size(size: Any, preprocessor: list = None, dimensions: dict = None) -> ArraySize:
    """Converts an arbitrary object into an integer representing memory size"""
    ret_size = ArraySize()
    if type(size) is not int:
        try:
            # Try to convert the size to an integer
            ret_size.add(int(size))
        except ValueError:
            # If size cannot be natively converted to an int, we try to derive it from the preprocessor
            try:
                if preprocessor is not None:
                    if "," in size:
                        for dimension in size.split(","):
                            ret_size.add(find_size_in_preprocessor(dimension, preprocessor))
                    else:
                        ret_size.add(find_size_in_preprocessor(size, preprocessor))
            except TypeError:
                # At least one of the dimension cannot be derived from the preprocessor
                pass
            # If size cannot be natively converted, nor retrieved from the preprocessor, we check user provided values
            if dimensions is not None:
                if size in dimensions.keys():
                    try:
                        ret_size.add(int(dimensions[size]))
                    except ValueError:
                        # User error, no mitigation
                        return ret_size
                elif "," in size:
                    for dimension in size.split(","):
                        try:
                            ret_size.add(int(dimensions[dimension]))
                        except ValueError:
                            # User error, no mitigation
                            return ret_size
    else:
        # size is already an int. no need for conversion
        ret_size.add(size)

    return ret_size


def wrap_timing(code: str, lang: Language) -> str:
    """Helper to wrap timing code around the provided code"""
    if is_cxx(lang):
        return end_timing_cxx(start_timing_cxx(code))
    elif is_fortran(lang):
        return wrap_timing_fortran(code)
    return ""


def start_timing_cxx(code: str) -> str:
    """Wrap C++ timing code around the provided code"""

    start = "auto kt_timing_start = std::chrono::steady_clock::now();"
    end = "auto kt_timing_end = std::chrono::steady_clock::now();"
    timing = "std::chrono::duration<float, std::milli> elapsed_time = kt_timing_end - kt_timing_start;"

    return "\n".join([start, code, end, timing])


def wrap_timing_fortran(code: str) -> str:
    """Wrap Fortran timing code around the provided code"""

    start = "call system_clock(kt_timing_start, kt_rate)"
    end = "call system_clock(kt_timing_end)"
    timing = "timing = (real(kt_timing_end - kt_timing_start) / real(kt_rate)) * 1e3"

    return "\n".join([start, code, end, timing])


def end_timing_cxx(code: str) -> str:
    """In C++ we need to return the measured time"""
    return "\n".join([code, "return elapsed_time.count();\n"])


def wrap_data(code: str, langs: Code, data: dict, preprocessor: list = None, user_dimensions: dict = None) -> str:
    """Insert data directives before and after the timed code"""
    intro = str()
    outro = str()
    for name in data.keys():
        if "*" in data[name][0]:
            size = parse_size(data[name][1], preprocessor=preprocessor, dimensions=user_dimensions)
            if is_openacc(langs.directive) and is_cxx(langs.language):
                intro += create_data_directive_openacc_cxx(name, size)
                outro += exit_data_directive_openacc_cxx(name, size)
            elif is_openacc(langs.directive) and is_fortran(langs.language):
                intro += create_data_directive_openacc_fortran(name, size)
                outro += exit_data_directive_openacc_fortran(name, size)
    return "\n".join([intro, code, outro])


def extract_directive_code(code: str, langs: Code, kernel_name: str = None) -> dict:
    """Extract explicitly marked directive sections from code"""
    if is_cxx(langs.language):
        start_string = "#pragma tuner start"
    elif is_fortran(langs.language):
        start_string = "!$tuner start"

    return extract_code(start_string, langs.language.end_string(), code, langs, kernel_name)


def extract_initialization_code(code: str, langs: Code) -> str:
    """Extract the initialization section from code"""
    if is_cxx(langs.language):
        start_string = "#pragma tuner initialize"
    elif is_fortran(langs.language):
        start_string = "!$tuner initialize"

    init_code = extract_code(start_string, langs.language.end_string(), code, langs)
    if len(init_code) >= 1:
        return "\n".join(init_code.values()) + "\n"
    else:
        return ""


def extract_deinitialization_code(code: str, langs: Code) -> str:
    """Extract the deinitialization section from code"""
    if is_cxx(langs.language):
        start_string = "#pragma tuner deinitialize"
    elif is_fortran(langs.language):
        start_string = "!$tuner deinitialize"

    init_code = extract_code(start_string, langs.language.end_string(), code, langs)
    if len(init_code) >= 1:
        return "\n".join(init_code.values()) + "\n"
    else:
        return ""


def format_argument_fortran(p_type: str, p_size: int, p_name: str) -> str:
    """Format the argument for Fortran code"""
    argument = ""
    if "float*" in p_type:
        argument = f"real (c_float), dimension({p_size}) :: {p_name}"
    elif "double*" in p_type:
        argument = f"real (c_double), dimension({p_size}) :: {p_name}"
    elif "int*" in p_type:
        argument = f"integer (c_int), dimension({p_size}) :: {p_name}"
    elif "float" in p_type:
        argument = f"real (c_float), value :: {p_name}"
    elif "double" in p_type:
        argument = f"real (c_double), value :: {p_name}"
    elif "int" in p_type:
        argument = f"integer (c_int), value :: {p_name}"
    return argument


def extract_directive_signature(code: str, langs: Code, kernel_name: str = None) -> dict:
    """Extract the user defined signature for directive sections"""

    if is_cxx(langs.language):
        start_string = "#pragma tuner start"
    elif is_fortran(langs.language):
        start_string = "!$tuner start"
    signatures = dict()

    for line in code.replace("\\\n", "").split("\n"):
        if start_string in line:
            if kernel_name is None or correct_kernel(kernel_name, line):
                tmp_string = line.strip().split(" ")
                if is_cxx(langs.language):
                    name = tmp_string[3]
                    tmp_string = tmp_string[4:]
                elif is_fortran(langs.language):
                    name = tmp_string[2]
                    tmp_string = tmp_string[3:]
                params = list()
                for param in tmp_string:
                    if len(param) == 0:
                        continue
                    p_name = param.split("(")[0]
                    param = param.replace(p_name, "", 1)
                    p_type = param[1:-1]
                    p_type = p_type.split(":")[0]
                    if "*" in p_type:
                        p_type = p_type.replace("*", " * restrict")
                    if is_cxx(langs.language):
                        params.append(f"{p_type} {p_name}")
                    elif is_fortran(langs.language):
                        params.append(p_name)
                if is_cxx(langs.language):
                    signatures[name] = f"float {name}({', '.join(params)})"
                elif is_fortran(langs.language):
                    signatures[
                        name
                    ] = f"function {name}({', '.join(params)}) result(timing)\nuse iso_c_binding\nimplicit none\n"
                    params = list()
                    for param in tmp_string:
                        if len(param) == 0:
                            continue
                        p_name = param.split("(")[0]
                        param = param.replace(p_name, "", 1)
                        p_type = param[1:-1]
                        p_size = p_type.split(":")[1]
                        p_type = p_type.split(":")[0]
                        params.append(format_argument_fortran(p_type, p_size, p_name))
                    signatures[name] += "\n".join(params) + "\n"
                    signatures[
                        name
                    ] += "integer(c_int):: kt_timing_start\nreal(c_float):: kt_rate\ninteger(c_int):: kt_timing_end\nreal(c_float):: timing\n"

    return signatures


def extract_directive_data(code: str, langs: Code, kernel_name: str = None) -> dict:
    """Extract the data used in the directive section"""

    if is_cxx(langs.language):
        start_string = "#pragma tuner start"
    elif is_fortran(langs.language):
        start_string = "!$tuner start"
    data = dict()

    for line in code.replace("\\\n", "").split("\n"):
        if start_string in line:
            if kernel_name is None or correct_kernel(kernel_name, line):
                if is_cxx(langs.language):
                    name = line.strip().split(" ")[3]
                    tmp_string = line.strip().split(" ")[4:]
                elif is_fortran(langs.language):
                    name = line.strip().split(" ")[2]
                    tmp_string = line.strip().split(" ")[3:]
                data[name] = dict()
                for param in tmp_string:
                    if len(param) == 0:
                        continue
                    p_name = param.split("(")[0]
                    param = param.replace(p_name, "", 1)
                    param = param[1:-1]
                    p_type = param.split(":")[0]
                    try:
                        p_size = param.split(":")[1]
                    except IndexError:
                        p_size = 0
                    data[name][p_name] = [p_type, p_size]

    return data


def extract_preprocessor(code: str) -> list:
    """Extract include and define statements from code"""
    preprocessor = list()

    for line in code.replace("\\\n", "").split("\n"):
        if "#define" in line or "#include" in line:
            preprocessor.append(line)

    return preprocessor


def generate_directive_function(
    preprocessor: list,
    signature: str,
    body: str,
    langs: Code,
    data: dict = None,
    initialization: str = "",
    deinitialization: str = "",
    user_dimensions: dict = None,
) -> str:
    """Generate tunable function for one directive"""

    if is_cxx(langs.language):
        code = cpp_template
        body = start_timing_cxx(body)
        if data is not None:
            body = wrap_data(body + "\n", langs, data, preprocessor, user_dimensions)
        body = end_timing_cxx(body)
    elif is_fortran(langs.language):
        code = f90_template
        body = wrap_timing(body, langs.language)
        if data is not None:
            body = wrap_data(body + "\n", langs, data, preprocessor, user_dimensions)
        name = signature.split(" ")[1].split("(")[0]
        code = code.replace("<!?NAME?!>", name)
    code = code.replace("<!?PREPROCESSOR?!>", "\n".join(preprocessor))
    # if present, add user specific dimensions as defines
    if user_dimensions is not None:
        user_defines = ""
        for key, value in user_dimensions.items():
            user_defines += f"#define {key} {value}\n"
        code = code.replace("<!?USER_DEFINES?!>", user_defines)
    else:
        code = code.replace("<!?USER_DEFINES?!>", "")
    code = code.replace("<!?SIGNATURE?!>", signature)
    code = code.replace("<!?INITIALIZATION?!>", initialization)
    code = code.replace("<!?DEINITIALIZATION?!>", deinitialization)
    if data is not None:
        body = add_present_openacc(body, langs, data, preprocessor, user_dimensions)
    code = code.replace("<!?BODY?!>", body)

    return code


def allocate_array(p_type: str, size: int) -> np.ndarray:
    """Allocate a Numpy array"""
    max_int = 1024
    array = None
    if p_type == "float*":
        array = np.random.rand(size).astype(np.float32)
    elif p_type == "double*":
        array = np.random.rand(size).astype(np.float64)
    elif p_type == "int*":
        array = np.random.randint(max_int, size=size)
    else:
        # The parameter is an array of user defined types
        array = np.random.rand(size).astype(np.byte)
    return array


def allocate_scalar(p_type: str, size: int) -> np.number:
    """Allocate a Numpy scalar"""
    scalar = None
    if p_type == "float":
        scalar = np.float32(size)
    elif p_type == "double":
        scalar = np.float64(size)
    elif p_type == "int":
        scalar = np.int32(size)
    else:
        # The parameter is some user defined type
        scalar = np.byte(size)
    return scalar


def allocate_signature_memory(data: dict, preprocessor: list = None, user_dimensions: dict = None) -> list:
    """Allocates the data needed by a kernel and returns the arguments array"""
    args = []

    for parameter in data.keys():
        p_type = data[parameter][0]
        size = parse_size(data[parameter][1], preprocessor, user_dimensions)
        if "*" in p_type:
            args.append(allocate_array(p_type, size.get()))
        else:
            args.append(allocate_scalar(p_type, size.get()))

    return args


def add_new_line(line: str) -> str:
    """Adds the new line character to the end of the line if not present"""
    if line.rfind("\n") != len(line) - 1:
        return line + "\n"
    return line


def add_present_openacc(
    code: str, langs: Code, data: dict, preprocessor: list = None, user_dimensions: dict = None
) -> str:
    """Add the present clause to OpenACC directive"""
    new_body = ""
    for line in code.replace("\\\n", "").split("\n"):
        if not line_contains_openacc_parallel_directive(line, langs.language):
            new_body += line
        else:
            # The line contains an OpenACC directive
            if openacc_directive_contains_data_clause(line):
                # The OpenACC directive manages memory, do not interfere
                return code
            else:
                new_line = line.replace("\n", "")
                present_clause = ""
                for name in data.keys():
                    if "*" in data[name][0]:
                        size = parse_size(data[name][1], preprocessor=preprocessor, dimensions=user_dimensions)
                        if is_cxx(langs.language):
                            present_clause += add_present_openacc_cxx(name, size)
                        elif is_fortran(langs.language):
                            present_clause += add_present_openacc_fortran(name, size)
                new_body += new_line + present_clause.rstrip() + "\n"
        new_body = add_new_line(new_body)
    return new_body


def add_present_openacc_cxx(name: str, size: ArraySize) -> str:
    """Create present clause for C++ OpenACC directive"""
    return f" present({name}[:{size.get()}]) "


def add_present_openacc_fortran(name: str, size: ArraySize) -> str:
    """Create present clause for Fortran OpenACC directive"""
    if len(size) == 1:
        return f" present({name}(:{size.get()})) "
    else:
        md_size = fortran_md_size(size)
        return f" present({name}({','.join(md_size)})) "


def process_directives(langs: Code, source: str, user_dimensions: dict = None) -> Tuple[dict, dict]:
    """Helper functions to process all the directives in the code and create tunable functions"""
    kernel_strings = dict()
    kernel_args = dict()
    preprocessor = extract_preprocessor(source)
    signatures = extract_directive_signature(source, langs)
    bodies = extract_directive_code(source, langs)
    data = extract_directive_data(source, langs)
    init = extract_initialization_code(source, langs)
    deinit = extract_deinitialization_code(source, langs)
    for kernel in signatures.keys():
        kernel_strings[kernel] = generate_directive_function(
            preprocessor, signatures[kernel], bodies[kernel], langs, data[kernel], init, deinit, user_dimensions
        )
        kernel_args[kernel] = allocate_signature_memory(data[kernel], preprocessor, user_dimensions)
    return (kernel_strings, kernel_args)
