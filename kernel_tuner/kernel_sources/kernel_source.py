import kernel_tuner.util as util

from abc import abstractmethod

from kernel_tuner.language import Language
from kernel_tuner.kernel_sources.model.prepared_kernel_source_data import PreparedKernelSourceData


class KernelSourceFactory(type):
    '''
    Factory to dynamically determine if a KernelSource should be a KernelSourceStr or a KernelSourceFn.
    if lang=generic_python, KernelSourceFn will be used. In all other cases, an instance of KernelSourceStr 
    is created.
    
    We create a kernelsouce by calling KernelSource(...). The call method for KernelSource is replaced by 
    the call method in KernelSourceFactory, because KernelSource uses that class as metaclass. Inside the 
    call method, a call to create either KernelSourceStr or KernelSourceFn is performed. This triggers the 
    __init__ call of the corresponding subclass. In both subclasses, we call super().__init__, which triggers 
    the __init__ call of the KernelSource class to initalize some common variables.
    '''
    def __call__(cls, kernel_name, kernel_sources, lang, defines=None, call_function=None, decorator=None):
        if lang == None:
            language = None 
        else:
            try:
                language = Language(lang.upper())
            except ValueError:
                raise TypeError(f"Supported languages are {[l.value for l in Language]}")
            
        # Determine if we need to create a KernelSourceStr or a KernelSourceFn
        if (language and language == Language.GENERIC_PYTHON): 
            ks_str = False 
        else:
            ks_str = True
        
        from kernel_tuner.kernel_sources.kernel_source_str import KernelSourceStr
        from kernel_tuner.kernel_sources.kernel_source_fn import KernelSourceFn
        if cls is KernelSource:  
            if ks_str:
                return KernelSourceStr(kernel_name, kernel_sources, lang, defines)
            else:
                return KernelSourceFn(kernel_name, kernel_sources, lang, defines, call_function, decorator)

        # Else, normal behaviour for subclasses
        if ks_str:
            return super().__call__(kernel_name, kernel_sources, lang, defines)
        else:
            return super().__call__(kernel_name, kernel_sources, lang, defines, call_function, decorator)
    


class KernelSource(metaclass=KernelSourceFactory):

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
            language = util.detect_language(kernel_string)
            self.lang = Language(language.upper())
        else:
            try:
                self.lang = Language(lang.upper())
            except ValueError:
                raise TypeError(f"Supported languages are {[l.value for l in Language]}")
            

    @abstractmethod
    def prepare_kernel_instance(self, kernel_options, params, grid, threads) -> PreparedKernelSourceData:
        raise NotImplementedError("create_kernel_instance not implemented")

    @abstractmethod
    def check_argument_lists(self, kernel_name, arguments):
        raise NotImplementedError("check_argument_lists not implemented")


