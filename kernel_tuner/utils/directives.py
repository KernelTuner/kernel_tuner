import numpy as np


def correct_kernel(kernel_name: str, line: str) -> bool:
    return f" {kernel_name} " in line or (
        kernel_name in line and len(line.partition(kernel_name)[2]) == 0
    )


def cpp_or_f90(code: str) -> tuple:
    return "#pragma acc" in code, "!$acc" in code


def extract_code(start: str, stop: str, code: str, kernel_name: str = None) -> dict:
    """Extract an arbitrary section of code"""
    found_section = False
    sections = dict()
    tmp_string = list()
    name = ""
    init_found = 0
    cpp, f90 = cpp_or_f90(code)

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
                        if cpp:
                            name = line.strip().split(" ")[3]
                        elif f90:
                            name = line.strip().split(" ")[2]
                    except IndexError:
                        name = f"init_{init_found}"
                        init_found += 1

    return sections


def extract_directive_code(code: str, kernel_name: str = None) -> dict:
    """Extract explicitly marked directive sections from code"""
    cpp, f90 = cpp_or_f90(code)

    if cpp:
        start_string = "#pragma tuner start"
        end_string = "#pragma tuner stop"
    elif f90:
        start_string = "!$tuner start"
        end_string = "!$tuner stop"

    return extract_code(start_string, end_string, code, kernel_name)


def extract_initialization_code(code: str) -> str:
    """Extract the initialization section from code"""
    cpp, f90 = cpp_or_f90(code)

    if cpp:
        start_string = "#pragma tuner initialize"
        end_string = "#pragma tuner stop"
    elif f90:
        start_string = "!$tuner initialize"
        end_string = "!$tuner stop"

    init_code = extract_code(start_string, end_string, code)
    if len(init_code) >= 1:
        return "\n".join(init_code.values())
    else:
        return ""


def extract_directive_signature(code: str, kernel_name: str = None) -> dict:
    """Extract the user defined signature for directive sections"""
    cpp, f90 = cpp_or_f90(code)

    if cpp:
        start_string = "#pragma tuner start"
    elif f90:
        start_string = "!$tuner start"
    signatures = dict()

    for line in code.replace("\\\n", "").split("\n"):
        if start_string in line:
            if kernel_name is None or correct_kernel(kernel_name, line):
                tmp_string = line.strip().split(" ")
                if cpp:
                    name = tmp_string[3]
                    tmp_string = tmp_string[4:]
                elif f90:
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
                    if cpp:
                        params.append(f"{p_type} {p_name}")
                    elif f90:
                        params.append(p_name)
                if cpp:
                    signatures[name] = f"float {name}({', '.join(params)})"
                elif f90:
                    signatures[
                        name
                    ] = f"function {name}({', '.join(params)}) result(timing)\nuse iso_c_binding\n"
                    params = list()
                    for param in tmp_string:
                        if len(param) == 0:
                            continue
                        p_name = param.split("(")[0]
                        param = param.replace(p_name, "", 1)
                        p_type = param[1:-1]
                        p_size = p_type.split(":")[1]
                        p_type = p_type.split(":")[0]
                        if "float*" in p_type:
                            params.append(
                                f"real (c_float), dimension({p_size}) :: {p_name}"
                            )
                        elif "double*" in p_type:
                            params.append(
                                f"real (c_double), dimension({p_size}) :: {p_name}"
                            )
                        elif "int*" in p_type:
                            params.append(
                                f"integer (c_int), dimension({p_size}) :: {p_name}"
                            )
                        elif "float" in p_type:
                            params.append(f"real (c_float), value :: {p_name}")
                        elif "double" in p_type:
                            params.append(f"real (c_double), value :: {p_name}")
                        elif "int" in p_type:
                            params.append(f"integer (c_int), value :: {p_name}")
                    signatures[name] += "\n".join(params) + "\n"

    return signatures


def extract_directive_data(code: str, kernel_name: str = None) -> dict:
    """Extract the data used in the directive section"""
    cpp, f90 = cpp_or_f90(code)

    if cpp:
        start_string = "#pragma tuner start"
    elif f90:
        start_string = "!$tuner start"
    data = dict()

    for line in code.replace("\\\n", "").split("\n"):
        if start_string in line:
            if kernel_name is None or correct_kernel(kernel_name, line):
                if cpp:
                    name = line.strip().split(" ")[3]
                    tmp_string = line.strip().split(" ")[4:]
                elif f90:
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


def wrap_timing(code: str) -> str:
    """Wrap timing code around the provided code"""
    cpp, f90 = cpp_or_f90(code)

    if cpp:
        start = "auto kt_timing_start = std::chrono::steady_clock::now();"
        end = "auto kt_timing_end = std::chrono::steady_clock::now();"
        timing = "std::chrono::duration<float, std::milli> elapsed_time = kt_timing_end - kt_timing_start;"
        ret = "return elapsed_time.count();"
    elif f90:
        start = "integer (c_int) :: kt_timing_start\nreal (c_float) :: kt_rate\ninteger (c_int) :: kt_timing_end\nreal (c_float) :: timing\ncall system_clock(kt_timing_start, kt_rate)"
        end = "call system_clock(kt_timing_end)"
        timing = "timing = (real(kt_timing_end - kt_timing_start) / real(kt_rate)) * 1e3"
        ret = ""

    return "\n".join([start, code, end, timing, ret])


def generate_directive_function(
    preprocessor: str, signature: str, body: str, initialization: str = ""
) -> str:
    """Generate tunable function for one directive"""
    cpp, f90 = cpp_or_f90(body)

    code = "\n".join(preprocessor)
    if cpp and "#include <chrono>" not in preprocessor:
        code += "\n#include <chrono>\n"
    if cpp:
        code += 'extern "C" ' + signature + "{\n"
    elif f90:
        code += "\nmodule kt\nuse iso_c_binding\ncontains\n"
        code += "\n" + signature
    if len(initialization) > 1:
        code += initialization + "\n"
    code += wrap_timing(body)
    if cpp:
        code += "\n}"
    elif f90:
        name = signature.split(" ")[1].split("(")[0]
        code += f"\nend function {name}\nend module kt\n"

    return code


def allocate_signature_memory(data: dict, preprocessor: list = None) -> list:
    """Allocates the data needed by a kernel and returns the arguments array"""
    args = []
    max_int = 1024

    for parameter in data.keys():
        p_type = data[parameter][0]
        size = data[parameter][1]
        if type(size) is not int:
            try:
                # Try to convert the size to an integer
                size = int(size)
            except ValueError:
                # If size cannot be natively converted to string, we try to derive it from the preprocessor
                for line in preprocessor:
                    if f"#define {size}" in line:
                        try:
                            size = int(line.split(" ")[2])
                            break
                        except ValueError:
                            continue
        if "*" in p_type:
            # The parameter is an array
            if p_type == "float*":
                args.append(np.random.rand(size).astype(np.float32))
            elif p_type == "double*":
                args.append(np.random.rand(size).astype(np.float64))
            elif p_type == "int*":
                args.append(np.random.randint(max_int, size=size))
        else:
            # The parameter is a scalar
            if p_type == "float":
                args.append(np.float32(size))
            elif p_type == "double":
                args.append(np.float64(size))
            elif p_type == "int":
                args.append(np.int32(size))

    return args
