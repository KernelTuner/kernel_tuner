import numpy as np

def extract_code(start: str, stop: str, code: str, kernel_name: str = None) -> dict:
    """Extract an arbitrary section of code"""
    found_section = False
    sections = dict()
    tmp_string = list()
    name = ""

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
                if kernel_name is None or f" {kernel_name} " in line:
                    found_section = True
                    name = line.strip().split(" ")[3]

    return sections


def extract_directive_code(code: str, kernel_name: str = None) -> dict:
    """Extract explicitly marked directive sections from code"""
    cpp = False
    f90 = False
    if "#pragma acc" in code:
        cpp = True
    elif "!$acc" in code:
        f90 = True

    if cpp:
        start_string = "#pragma tuner start"
        end_string = "#pragma tuner stop"
    elif f90:
        start_string = "!$tuner start"
        end_string = "!$tuner stop"

    return extract_code(start_string, end_string, code, kernel_name)


def extract_initialization_code(code: str) -> str:
    """Extract the initialization section from code"""
    cpp = False
    f90 = False
    if "#pragma acc" in code:
        cpp = True
    elif "!$acc" in code:
        f90 = True

    if cpp:
        start_string = "#pragma tuner initialize"
        end_string = "#pragma tuner stop"
    elif f90:
        start_string = "!$tuner initialize"
        end_string = "!$tuner stop"

    function = extract_code(start_string, end_string, code)
    if len(function) == 1:
        _, value = function.popitem()
        return value
    else:
        return ""


def extract_directive_signature(code: str, kernel_name: str = None) -> dict:
    """Extract the user defined signature for directive sections"""
    cpp = False
    f90 = False
    if "#pragma acc" in code:
        cpp = True
    elif "!$acc" in code:
        f90 = True

    if cpp:
        start_string = "#pragma tuner start"
    elif f90:
        start_string = "!$tuner start"
    signatures = dict()

    for line in code.replace("\\\n", "").split("\n"):
        if start_string in line:
            if kernel_name is None or f" {kernel_name} " in line:
                tmp_string = line.strip().split(" ")
                name = tmp_string[3]
                tmp_string = tmp_string[4:]
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
                    signatures[name] = f"function {name}({', '.join(params)})"

    return signatures


def extract_directive_data(code: str, kernel_name: str = None) -> dict:
    """Extract the data used in the directive section"""
    cpp = False
    f90 = False
    if "#pragma acc" in code:
        cpp = True
    elif "!$acc" in code:
        f90 = True

    if cpp:
        start_string = "#pragma tuner start"
    elif f90:
        start_string = "!$tuner start"
    data = dict()

    for line in code.replace("\\\n", "").split("\n"):
        if start_string in line:
            if kernel_name is None or f" {kernel_name} " in line:
                name = line.strip().split(" ")[3]
                data[name] = dict()
                tmp_string = line.strip().split(" ")[4:]
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
    cpp = False
    f90 = False
    if "#pragma acc" in code:
        cpp = True
    elif "!$acc" in code:
        f90 = True

    if cpp:
        start = "auto start = std::chrono::steady_clock::now();"
        end = "auto end = std::chrono::steady_clock::now();"
        timing = "std::chrono::duration<float, std::milli> elapsed_time = end - start;"
        ret = "return elapsed_time.count();"
    elif f90:
        start = "integer,intent(out) start\nreal,intent(out) rate\ninteger,intent(out) end\ncall system_clock(start, rate)"
        end = "call system_clock(end)"
        timing = "timing = (real(end - start) / real(rate)) * 1e3"
        ret = ""

    return "\n".join([start, code, end, timing, ret])


def generate_directive_function(
    preprocessor: str, signature: str, body: str, initialization: str = ""
) -> str:
    """Generate tunable function for one directive"""
    cpp = False
    f90 = False
    if "#pragma acc" in body:
        cpp = True
    elif "!$acc" in body:
        f90 = True

    code = "\n".join(preprocessor)
    if cpp and "#include <chrono>" not in preprocessor:
        code += "\n#include <chrono>\n"
    if cpp:
        code += 'extern "C" ' + signature + "{\n"
    elif f90:
        code += signature + " result(timing)\n"
    if len(initialization) > 1:
        code += initialization + "\n"
    code += wrap_timing(body) + "\n}"
    if cpp:
        code += "\n}"
    elif f90:
        name = signature.split(" ")[1]
        code += f"\nend function {name}\n"

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
