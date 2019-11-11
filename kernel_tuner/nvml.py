try:
    import pynvml
except ImportError:
    pynvml = None

class nvml(object):
    """Class that gathers the NVML functionality for one device"""

    def __init__(self, id=0):
        """Create object to control device using NVML"""

        pynvml.nvmlInit()
        self.dev = pynvml.nvmlDeviceGetHandleByIndex(id)

        try:
            self._pwr_limit = pynvml.nvmlDeviceGetPowerManagementLimit(self.dev)
            self.pwr_constraints = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(self.dev)
        except pynvml.NVMLError_NotSupported:
            self._pwr_limit = None
            self.pwr_constraints = [1, 0] # inverted range to make all range checks fail

        try:
            self._persistence_mode = pynvml.nvmlDeviceGetPersistenceMode(self.dev)
        except pynvml.NVMLError_NotSupported:
            self._persistence_mode = None

        try:
            self._auto_boost = pynvml.nvmlDeviceGetAutoBoostedClocksEnabled(self.dev)[0]  # returns [isEnabled, isDefaultEnabled]
        except pynvml.NVMLError_NotSupported:
            self._auto_boost = None

        try:
            self.gr_clock_default = pynvml.nvmlDeviceGetDefaultApplicationsClock(self.dev, pynvml.NVML_CLOCK_GRAPHICS)
            self.sm_clock_default = pynvml.nvmlDeviceGetDefaultApplicationsClock(self.dev, pynvml.NVML_CLOCK_SM)
            self.mem_clock_default = pynvml.nvmlDeviceGetDefaultApplicationsClock(self.dev, pynvml.NVML_CLOCK_MEM)

            self.supported_mem_clocks = pynvml.nvmlDeviceGetSupportedMemoryClocks(self.dev)

            #gather the supported gr clocks for each supported mem clock into a dict
            self.supported_gr_clocks = dict()
            for mem_clock in self.supported_mem_clocks:
                supported_gr_clocks = pynvml.nvmlDeviceGetSupportedGraphicsClocks(self.dev, mem_clock)
                self.supported_gr_clocks[mem_clock] = supported_gr_clocks
        except pynvml.NVMLError_NotSupported:
            self.gr_clock_default = None
            self.sm_clock_default = None
            self.mem_clock_default = None
            self.supported_mem_clocks = []
            self.supported_gr_clocks = dict()

    @property
    def pwr_state(self):
        """Get the Device current Power State"""
        return pynvml.nvmlDeviceGetPowerState(self.dev)

    @property
    def pwr_limit(self):
        """Control the power limit (may require permission), check pwr_constraints for the allowed range"""
        return self._pwr_limit

    @pwr_limit.setter
    def pwr_limit(self, new_limit):
        if not self.pwr_constraints[0] <= new_limit <= self.pwr_constraints[1]:
            raise ValueError("Power limit out of range")
        pynvml.nvmlDeviceSetPowerManagementLimit(self.dev, new_limit)
        self._pwr_limit = pynvml.nvmlDeviceGetPowerManagementLimit(self.dev)

    @property
    def persistence_mode(self):
        """Control persistence mode (may require permission), 0 for disabled, 1 for enabled"""
        return self._persistence_mode

    @persistence_mode.setter
    def persistence_mode(self, new_mode):
        if not new_mode in [0, 1]:
            raise ValueError("Illegal value for persistence mode, should be either 0 or 1")
        pynvml.nvmlDeviceSetPersistenceMode(self.dev, new_mode)
        self._persistence_mode = pynvml.nvmlDeviceGetPersistenceMode(self.dev)

    def set_clocks(self, mem_clock, gr_clock):
        """Set the memory and graphics clock for this device (may require permission)"""
        if not mem_clock in self.supported_mem_clocks:
            raise ValueError("Illegal value for memmory clock")
        if not gr_clock in self.supported_gr_clocks[mem_clock]:
            raise ValueError("Graphics clock incompatible with memory clock")
        pynvml.nvmlDeviceSetApplicationsClocks(self.dev, mem_clock, gr_clock)
        self._gr_clock = pynvml.nvmlDeviceGetApplicationsClock(self.dev, pynvml.NVML_CLOCK_GRAPHICS)
        self._sm_clock = pynvml.nvmlDeviceGetApplicationsClock(self.dev, pynvml.NVML_CLOCK_SM)
        self._mem_clock = pynvml.nvmlDeviceGetApplicationsClock(self.dev, pynvml.NVML_CLOCK_MEM)

    @property
    def gr_clock(self):
        """Control the graphics clock (may require permission), only values compatible with the memory clock can be set directly"""
        return pynvml.nvmlDeviceGetApplicationsClock(self.dev, pynvml.NVML_CLOCK_GRAPHICS)

    @gr_clock.setter
    def gr_clock(self, new_clock):
        self.set_clocks(self.mem_clock, new_clock)

    @property
    def mem_clock(self):
        """Control the graphics clock (may require permission), only values compatible with the graphics clock can be set directly"""
        return pynvml.nvmlDeviceGetApplicationsClock(self.dev, pynvml.NVML_CLOCK_MEM)

    @mem_clock.setter
    def mem_clock(self, new_clock):
        self.set_clocks(new_clock, self.gr_clock)

    @property
    def auto_boost(self):
        """Control the auto boost setting (may require permission), 0 for disable, 1 for enabled"""
        return self._auto_boost

    @auto_boost.setter
    def auto_boost(self, setting):
        #might need to use pynvml.NVML_FEATURE_DISABLED or pynvml.NVML_FEATURE_ENABLED instead of 0 or 1
        if not setting in [0, 1]:
            raise ValueError("Illegal value for auto boost enabled, should be either 0 or 1")
        pynvml.nvmlDeviceSetAutoBoostedClocksEnabled(self.dev, setting)
        self._auto_boost = pynvml.nvmlDeviceGetAutoBoostedClocksEnabled(self.dev)[0]

    def pwr_usage(self):
        """Return current power usage in milliwatts"""
        return pynvml.nvmlDeviceGetPowerUsage(self.dev)
