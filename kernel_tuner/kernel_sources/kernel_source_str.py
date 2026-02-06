import kernel_tuner.util as util
import logging

from kernel_tuner.kernel_sources.kernel_source import KernelSource
from kernel_tuner.core import wrap_templated_kernel
from kernel_tuner.kernel_sources.model.prepared_kernel_source_data import PreparedKernelSourceData
from kernel_tuner.language import Language


class KernelSourceStr(KernelSource):
    """Class that holds the kernel sources.

    There is a primary kernel source for string-based kernels., which can be either a source string,
    a filename (indicating a file containing the kernel source code),
    or a callable (generating the kernel source code).
    There can additionally be (one or multiple) secondary kernel sources, which
    must be filenames.
    """

    def __init__(self, kernel_name, kernel_sources, lang, defines=None):
        super().__init__(kernel_name, kernel_sources, lang, defines)
    
    def prepare_kernel_instance(self, kernel_options, params, grid, threads):
        name, kernel_string, temp_files = self.prepare_list_of_files(
            kernel_name=kernel_options.kernel_name,
            params=params,
            grid=grid,
            threads=threads,
            block_size_names=kernel_options.block_size_names,
        )

        lang_is_cuda = self.lang in [Language.CUDA, Language.NVCUDA]

        if lang_is_cuda and "<" in name and ">" in name:
            kernel_string, name = wrap_templated_kernel(kernel_string, name)

        return PreparedKernelSourceData(
            temp_files=temp_files,
            kernel_name=name,
            kernel_str=kernel_string,
            kernel_fn=None,
        )



    def get_kernel_string(self, index=0, params=None):
        """Retrieve the kernel source with the given index and return as a string.

        See util.get_kernel_string() for details.

        :param index: Index of the kernel source in the list of sources.
        :type index: int

        :param params: Dictionary containing the tunable parameters for this specific
            kernel instance, only needed when kernel_source is a generator.
        :type param: dict

        :returns: A string containing the kernel code.
        :rtype: string
        """
        logging.debug("get_kernel_string called")

        if hasattr(self, 'lang') and self.lang == Language.HYPERTUNER:
            return ""

        kernel_source = self.kernel_sources[index]
        return util.get_kernel_string(kernel_source, params)

    def prepare_list_of_files(
        self, kernel_name, params, grid, threads, block_size_names
    ):
        """Prepare the kernel string along with any additional files.

        The first file in the list is allowed to include or read in the others
        The files beyond the first are considered additional files that may also contain tunable parameters

        For each file beyond the first this function creates a temporary file with
        preprocessors statements inserted. Occurrences of the original filenames in the
        first file are replaced with their temporary counterparts.

        :param kernel_name: A string specifying the kernel name.
        :type kernel_name: string

        :param params: A dictionary with the tunable parameters for this particular
            instance.
        :type params: dict()

        :param grid: The grid dimensions for this instance. The grid dimensions are
            also inserted into the code as if they are tunable parameters for
            convenience.
        :type grid: tuple()

        :param threads: The thread block dimensions for this instance. The thread block are
            also inserted into the code as if they are tunable parameters for
            convenience.
        :type threads: tuple()

        :param block_size_names: A list of strings that denote the names
            for the thread block dimensions.
        :type block_size_names: list(string)

        """
        temp_files = dict()

        if self.lang == Language.HYPERTUNER:
            return tuple(["", "", temp_files])

        for i, f in enumerate(self.kernel_sources):
            if i > 0 and not util.looks_like_a_filename(f):
                raise ValueError("When passing multiple kernel sources, the secondary entries must be filenames")

            ks = self.get_kernel_string(i, params)
            # add preprocessor statements
            n, ks = util.prepare_kernel_string(
                kernel_name,
                ks,
                params,
                grid,
                threads,
                block_size_names,
                self.lang,
                self.defines,
            )

            if i == 0:
                # primary kernel source
                name = n
                kernel_string = ks
                continue

            # save secondary kernel sources to temporary files

            # generate temp filename with the same extension
            temp_file = util.get_temp_filename(suffix="." + f.split(".")[-1])
            temp_files[f] = temp_file
            util.write_file(temp_file, ks)
            # replace occurrences of the additional file's name in the first kernel_string with the name of the temp file
            kernel_string = kernel_string.replace(f, temp_file)

        return name, kernel_string, temp_files

    def get_user_suffix(self, index=0):
        """Get the suffix of the kernel filename, if the user specified one. Return None otherwise."""
        if util.looks_like_a_filename(self.kernel_sources[index]) and ("." in self.kernel_sources[index]):
            return "." + self.kernel_sources[index].split(".")[-1]
        return None

    def get_suffix(self, index=0):
        """Return a suitable suffix for a kernel filename.

        This uses the user-specified suffix if available, or one based on the
        lang/backend otherwise.
        """
        # TODO: Consider delegating this to the backend
        suffix = self.get_user_suffix(index)
        if suffix is not None:
            return suffix

        _suffixes = {Language.CUDA: ".cu", Language.OPENCL: ".cl", Language.C: ".c"}
        try:
            return _suffixes[self.lang]
        except KeyError:
            return ".c"

    def check_argument_lists(self, kernel_name, arguments):
        """Check if the kernel arguments have the correct types.

        This is done by calling util.check_argument_list on each kernel string.
        """
        for i, f in enumerate(self.kernel_sources):
            if not callable(f):
                util.check_argument_list(kernel_name, self.get_kernel_string(i), arguments)
            else:
                logging.debug("Checking of arguments list not supported yet for code generators.")