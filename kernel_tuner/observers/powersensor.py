import time
import numpy as np

from kernel_tuner.observers.observer import BenchmarkObserver

# check if powersensor is installed
try:
    import powersensor
except ImportError:
    powersensor = None

DUMP_FILE = "temp_psdumpfile"

class PowerSensorObserver(BenchmarkObserver):
    """Observer that an external PowerSensor2 device to accurately measure power

    Requires PowerSensor2 hardware and powersensor Python bindings.

    :param observables: A list of string, containing any of "ps_energy" or "ps_power".
        To measure energy in Joules or power consumption in Watt.
        If not passed "ps_energy" is used to report energy consumption of kernels in Joules.
    :type observables: list

    :param device: A string with the path to the PowerSensor2 device, default "/dev/ttyACM0".
    :type device: string

    """

    def __init__(self, observables=None, device=None):
        if not powersensor:
            raise ImportError("could not import powersensor")

        supported = ["ps_energy", "ps_power", "ps_readings"]
        for obs in observables:
            if not obs in supported:
                raise ValueError(f"Observable {obs} not in supported: {supported}")
        self.observables = observables or ["ps_energy"]

        device = device or "/dev/ttyACM0"
        self.ps = powersensor.PowerSensor(device)

        self.iteration = 0

    def wait_for_dump_file(self, file_name):
        marker_found = False
        sleep = 0
        while (not marker_found) and sleep < 1000:
            with open(file_name, 'r') as fh:
                file = fh.read().split('\n')
                marker_found = "M Y" in file
            if not marker_found:
                sleep += 50
                time.sleep(0.05)

    def before_start(self):
        # check if end marker has been written for previous measurement
        # if not, wait a bit
        if self.iteration > 0:
            self.wait_for_dump_file(DUMP_FILE + str(self.iteration-1))

        self.ps.dump(DUMP_FILE + str(self.iteration))

    def after_start(self):
        self.ps.mark('X')

    def after_finish(self):
        self.ps.mark('Y')
        # stop dumping
        # self.ps.dump("") # if we stop dumping immediately the end marker is missing
        self.iteration += 1

    def collect_data_from_dump_file(self, iteration):

        with open(DUMP_FILE + str(iteration), 'r') as fh:
            file_str = fh.read()

        marker_found = False
        times = []
        powers = []
        readings = []

        lines = file_str.split('\n')
        for i, line in enumerate(lines):
            # skip first line or empty lines
            if i==0 or len(line) < 1:
                continue
            # collect data between markers
            elif line[0] == "S" and marker_found:
                line_info = line.split(" ")
                times.append(float(line_info[1]))
                powers.append(float(line_info[-1]))
                readings.append([float(num) for num in line_info[1:]])
            # look for markers
            elif line[0] == "M":
                marker_found = not marker_found

        if len(times) < 2:
            print("Something went wrong ...")
            print(f"{iteration=}")
            print(f"{marker_found=}")
            print(f"{times=}")


        # integrate to get energy, divide energy by time in seconds to get power
        energy = np.trapz(powers, times) # Joule
        power = energy / (times[-1] - times[0]) # Watts

        return power, energy, readings

    def get_results(self):
        iterations = self.iteration

        # wait for end marker in last dump file and stop dumping
        self.wait_for_dump_file(DUMP_FILE + str(self.iteration-1))
        self.ps.dump("")

        powers = []
        energies = []
        reads = []

        for i in range(iterations):
            power, energy, readings = self.collect_data_from_dump_file(i)
            powers.append(power)
            energies.append(energy)
            reads.append(readings)

        results = {}
        if "ps_energy" in self.observables:
            results["ps_energy"] = np.average(energies)
        if "ps_power" in self.observables:
            results["ps_power"] = np.average(powers)
        if "ps_readings" in self.observables:
            results["ps_readings"] = reads

        if results["ps_power"] > 300:
            exit(1)

        # reset iteration
        self.iteration = 0

        return results
