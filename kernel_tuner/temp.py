


class NVMLObserver(BenchmarkObserver):
    """ Observer that measures time using CUDA events during benchmarking """
    def __init__(self, observables=None, device=0):
        self.observables = observables
        self.nvml = nvml(device)
        #verify that the observales are valid and that
        #our NVML+device combination is capable of measuring them
        #...

    def before_start(self):
        self.power_readings = []

    def after_start(self):
        self.t0 = time.time()

    def during(self):
        #first check if we're interested in power
        self.power_readings.append([time.time()-self.t0, self.nvml.pwr_usage()])

    def after_finish(self):
        #pre and postfix to start at 0 and end at kernel end
        if power_readings:
            power_readings = [[0.0, power_readings[0][1]]] + power_readings
            power_readings = power_readings + [[execution_time / 1000.0, power_readings[-1][1]]]
            result["power"].append(power_readings) #time in s, power usage in milliwatts

            #compute energy consumption as area under curve
            x = [d[0] for d in power_readings]
            y = [d[1]/1000.0 for d in power_readings] #convert to Watt
            energy.append(np.trapz(y,x)) #in Joule


    def get_results(self):
            results = {"time": np.average(self.times), "times": self.times.copy()}
            self.times = []
            return results



