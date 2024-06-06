from kernel_tuner.observers import PrologueObserver

try:
    import nvmetrics
except (ImportError):
    nvmetrics = None

class NCUObserver(PrologueObserver):
    """``NCUObserver`` measures performance counters.

        The exact performance counters supported differ per GPU, some examples:

         * "dram__bytes.sum",                                     # Counter         byte            # of bytes accessed in DRAM
         * "dram__bytes_read.sum",                                # Counter         byte            # of bytes read from DRAM
         * "dram__bytes_write.sum",                               # Counter         byte            # of bytes written to DRAM
         * "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum", # Counter         inst            # of FADD thread instructions executed where all predicates were true
         * "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum", # Counter         inst            # of FFMA thread instructions executed where all predicates were true
         * "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum", # Counter         inst            # of FMUL thread instructions executed where all predicates were true

        :param metrics: The metrics to observe. This should be a list of strings.
                        You can use ``ncu --query-metrics`` to get a list of valid metrics.
        :type metrics: list[str]

    """

    def __init__(self, metrics=None, device=0):
        if not nvmetrics:
            raise ImportError("could not import nvmetrics")

        self.metrics = metrics
        self.device = device
        self.results = dict()

    def before_start(self):
        nvmetrics.measureMetricsStart(self.metrics, self.device)

    def after_finish(self):
        self.results = nvmetrics.measureMetricsStop()

    def get_results(self):
        return dict(zip(self.metrics, self.results))
