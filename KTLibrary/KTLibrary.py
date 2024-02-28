from .CacheFiles.files import read_cache_file

class KTLibrary:
    def __init__(self):
        self.cache_file = None
    
    def read_file(self, file_path: str):
        self.cache_file = read_cache_file(self, file_path)
        print(type(self.cache_file))
        print(type(self.cache_file["device_name"]))
        # TODO: Check if cache file is in the right format, validate with json schema
    
    def print_info(self):
        device_name = self.cache_file["device_name"]
        kernel_name = self.cache_file["kernel_name"]
        problem_size = self.cache_file["problem_size"]
        print(device_name)
        print(kernel_name)
        print(problem_size)
        tune_parameters = self.cache_file["tune_params_keys"]
        for x in tune_parameters:
            print(x)
