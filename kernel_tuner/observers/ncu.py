from kernel_tuner.observers import PrologueObserver

try:
    import nvmetrics
except (ImportError):
    nvmetrics = None
    pass

class NCUObserver(PrologueObserver):
    """``NCUObserver`` measures performance counters.

    """

    def __init__(self, metrics=None):
        """Create a new ``NCUObserver``.

        :param metrics: The metrics to observe. This should be a list of strings.
                        You can use ``ncu --query-metrics`` to get a list of valid metrics.
        """

        if not nvmetrics:
            print("NCUObserver is not available.")

        self.metrics = metrics
        self.results = dict()

    def before_start(self):
        if nvmetrics:
            nvmetrics.measureMetricsStart(self.metrics)

    def after_finish(self):
        if nvmetrics:
            self.results = nvmetrics.measureMetricsStop()

    def get_results(self):
        return dict(zip(self.metrics, self.results))
