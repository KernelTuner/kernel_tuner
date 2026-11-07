# NVML functionality for the JuliaNVMLObserver, using the NVML wrapper of CUDA.jl.
# Functions raise `NVML.NVMLError` on failure; `error_code` lets Python tell e.g. "not supported" from "no permission".
# Units follow NVML: power in milliwatt, energy in millijoule, clocks in MHz, temperature in degrees Celsius.
module KernelTunerNVML

using CUDA: CUDA, NVML

# NVML return code of an exception, or -1 if it is not an NVML error
error_code(err) = err isa NVML.NVMLError ? Int(err.code) : -1


## device lookup

# the device (not a MIG instance, as power and clocks are controlled per device) that runs the given CUDA device,
# found by UUID, as CUDA and NVML can number the devices differently (e.g. with CUDA_VISIBLE_DEVICES)
device_of(cuda_device::CUDA.CuDevice) = NVML.Device(CUDA.parent_uuid(cuda_device))

device_by_index(index::Integer) = NVML.Device(index)

device_by_uuid(uuid::AbstractString) = NVML.Device(Base.UUID(replace(uuid, r"^(GPU|MIG)-" => "")))

function device_by_pci_bus(pci_bus::AbstractString)
    handle = Ref{NVML.nvmlDevice_t}()
    NVML.nvmlDeviceGetHandleByPciBusId_v2(pci_bus, handle)
    index = Ref{Cuint}()
    NVML.nvmlDeviceGetIndex(handle[], index)
    return NVML.Device(Int(index[]))  # NVML.Device has no constructor that takes a handle
end

index(dev::NVML.Device) = NVML.index(dev)

uuid(dev::NVML.Device) = "GPU-$(NVML.uuid(dev))"

function pci_bus(dev::NVML.Device)
    info = Ref{NVML.nvmlPciInfo_t}()
    NVML.nvmlDeviceGetPciInfo_v3(dev, info)
    chars = collect(info[].busId)
    return String(UInt8.(chars[1:something(findfirst(iszero, chars), length(chars) + 1) - 1]))
end


## power and energy

function power_limit(dev::NVML.Device)
    limit = Ref{Cuint}()
    NVML.nvmlDeviceGetPowerManagementLimit(dev, limit)
    return Int(limit[])
end

function power_limit_constraints(dev::NVML.Device)
    min_limit, max_limit = Ref{Cuint}(), Ref{Cuint}()
    NVML.nvmlDeviceGetPowerManagementLimitConstraints(dev, min_limit, max_limit)
    return [Int(min_limit[]), Int(max_limit[])]
end

set_power_limit!(dev::NVML.Device, limit::Integer) = NVML.nvmlDeviceSetPowerManagementLimit(dev, limit)

# current power, measured instantly where supported, as the regular power usage is averaged over about a second on
# recent GPUs, which is much longer than the run of a kernel
# reused for every call, so that reading the power during a measurement does not allocate and trigger garbage collection
const power_request = [NVML.nvmlFieldValue_t(ntuple(_ -> 0x00, sizeof(NVML.nvmlFieldValue_t)))]

function power(dev::NVML.Device)
    values = power_request
    GC.@preserve values begin
        ptr = pointer(values)
        values[1] = NVML.nvmlFieldValue_t(ntuple(_ -> 0x00, sizeof(NVML.nvmlFieldValue_t)))
        unsafe_store!(ptr.fieldId, NVML.NVML_FI_DEV_POWER_INSTANT)
        NVML.nvmlDeviceGetFieldValues(dev, 1, ptr)
        if unsafe_load(ptr.nvmlReturn) == NVML.NVML_SUCCESS
            return Int(unsafe_load(ptr.value).uiVal)
        end
    end
    # instant power is not supported, fall back to the regular power usage
    usage = Ref{Cuint}()
    NVML.nvmlDeviceGetPowerUsage(dev, usage)
    return Int(usage[])
end

# total energy used since the driver was loaded
function energy(dev::NVML.Device)
    total = Ref{Culonglong}()
    NVML.nvmlDeviceGetTotalEnergyConsumption(dev, total)
    return Int(total[])
end

temperature(dev::NVML.Device) = NVML.temperature(dev)

# performance state, from 0 (maximum performance) to 15 (minimum performance), or 32 if unknown
function power_state(dev::NVML.Device)
    state = Ref{NVML.nvmlPstates_t}()
    NVML.nvmlDeviceGetPowerState(dev, state)
    return Int(state[])
end


## device settings

function persistence_mode(dev::NVML.Device)
    mode = Ref{NVML.nvmlEnableState_t}()
    NVML.nvmlDeviceGetPersistenceMode(dev, mode)
    return Int(mode[])
end

set_persistence_mode!(dev::NVML.Device, mode::Integer) =
    NVML.nvmlDeviceSetPersistenceMode(dev, NVML.nvmlEnableState_t(mode))

function auto_boost(dev::NVML.Device)
    enabled, default_enabled = Ref{NVML.nvmlEnableState_t}(), Ref{NVML.nvmlEnableState_t}()
    NVML.nvmlDeviceGetAutoBoostedClocksEnabled(dev, enabled, default_enabled)
    return Int(enabled[])
end

set_auto_boost!(dev::NVML.Device, setting::Integer) =
    NVML.nvmlDeviceSetAutoBoostedClocksEnabled(dev, NVML.nvmlEnableState_t(setting))


## clocks, `kind` is "graphics" or "mem"

clock_type(kind::AbstractString) = kind == "graphics" ? NVML.NVML_CLOCK_GRAPHICS :
                                   kind == "mem" ? NVML.NVML_CLOCK_MEM :
                                   throw(ArgumentError("unknown clock kind: $kind"))

function clock_query(f, dev::NVML.Device, kind::AbstractString)
    mhz = Ref{Cuint}()
    f(dev, clock_type(kind), mhz)
    return Int(mhz[])
end

clock(dev::NVML.Device, kind) = clock_query(NVML.nvmlDeviceGetClockInfo, dev, kind)
applications_clock(dev::NVML.Device, kind) = clock_query(NVML.nvmlDeviceGetApplicationsClock, dev, kind)
default_applications_clock(dev::NVML.Device, kind) = clock_query(NVML.nvmlDeviceGetDefaultApplicationsClock, dev, kind)

# supported clocks, from high to low
supported_memory_clocks(dev::NVML.Device) = NVML.supported_memory_clocks(dev)
supported_graphics_clocks(dev::NVML.Device, mem_clock::Integer) = NVML.supported_graphics_clocks(dev, mem_clock)

set_applications_clocks!(dev::NVML.Device, mem_clock::Integer, gr_clock::Integer) =
    NVML.nvmlDeviceSetApplicationsClocks(dev, mem_clock, gr_clock)

function set_locked_clocks!(dev::NVML.Device, mem_clock::Integer, gr_clock::Integer)
    NVML.nvmlDeviceSetGpuLockedClocks(dev, gr_clock, gr_clock)
    NVML.nvmlDeviceSetMemoryLockedClocks(dev, mem_clock, mem_clock)
end

function reset_locked_clocks!(dev::NVML.Device)
    NVML.nvmlDeviceResetGpuLockedClocks(dev)
    NVML.nvmlDeviceResetMemoryLockedClocks(dev)
end

end
