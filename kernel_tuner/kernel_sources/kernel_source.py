import inspect
import kernel_tuner.util as util

from abc import abstractmethod

from kernel_tuner.kernel_sources.model.prepared_kernel_source_data import PreparedKernelSourceData




class KernelSource:
    def __new__(cls, kernel_name, kernel_sources, lang, defines=None):
        """Factory behavior"""
        if cls is KernelSource:  
            if inspect.isfunction(kernel_sources) and (lang and lang.upper() == "TRITON"): # TODO should this be isfunction?
                from kernel_tuner.kernel_sources.kernel_source_fn import KernelSourceFn
                print("CREATING KSFN")
                return KernelSourceFn(kernel_name, kernel_sources, lang, defines)
            else:
                from kernel_tuner.kernel_sources.kernel_source_str import KernelSourceStr
                print("CREATING KSSTR")
                return KernelSourceStr(kernel_name, kernel_sources, lang, defines)
        # otherwise, normal subclass init
        return super().__new__(cls)

    def __init__(self, kernel_name, kernel_sources, lang, defines=None):
        if not isinstance(kernel_sources, list):
            kernel_sources = [kernel_sources]
        self.kernel_sources = kernel_sources
        self.kernel_name = kernel_name
        self.defines = defines

        if lang is None:
            if callable(self.kernel_sources[0]):
                raise TypeError("Please specify language when using a code generator function")
            kernel_string = self.get_kernel_string(0)
            self.lang = util.detect_language(kernel_string)
        else:
            self.lang = lang

    @abstractmethod
    def prepare_kernel_instance(self, kernel_options, params, grid, threads) -> PreparedKernelSourceData:
        raise NotImplementedError("create_kernel_instance not implemented")

    @abstractmethod
    def check_argument_lists(self, kernel_name, arguments):
        raise NotImplementedError("check_argument_lists not implemented")



'''
class KernelSource:
    def __init__(self, kernel_name, kernel_sources, lang, defines=None):
        if not isinstance(kernel_sources, list):
            kernel_sources = [kernel_sources]

        self.kernel_sources = kernel_sources
        self.kernel_name = kernel_name
        self.defines = defines

        if lang is None:
            if callable(self.kernel_sources[0]):
                raise TypeError("Please specify language when using a code generator function")
            kernel_string = self.get_kernel_string(0)
            self.lang = util.detect_language(kernel_string)
        else:
            self.lang = lang

    @abstractmethod
    def prepare_kernel_instance(self, kernel_options, params, grid, threads) -> PreparedKernelSourceData:
        raise NotImplementedError("create_kernel_instance not implemented")

    @abstractmethod
    def check_argument_lists(self, kernel_name, arguments):
        raise NotImplementedError("check_argument_lists not implemented")

'''