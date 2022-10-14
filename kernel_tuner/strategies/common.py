from collections import OrderedDict

_docstring_template = """ Find the best performing kernel configuration in the parameter space

    This $NAME$ strategy supports the following strategy_options:

$STRAT_OPT$

    :params runner: A runner from kernel_tuner.runners
    :type runner: kernel_tuner.runner

    :param kernel_options: A dictionary with all options for the kernel.
    :type kernel_options: kernel_tuner.interface.Options

    :param device_options: A dictionary with all options for the device
        on which the kernel should be tuned.
    :type device_options: kernel_tuner.interface.Options

    :param tuning_options: A dictionary with all options regarding the tuning
        process.
    :type tuning_options: kernel_tuner.interface.Options

    :returns: A list of dictionaries for executed kernel configurations and their
        execution times. And a dictionary that contains information
        about the hardware/software environment on which the tuning took place.
    :rtype: list(dict()), dict()

    """

def get_strategy_docstring(name, strategy_options):
    """ Generate docstring for a 'tune' method of a strategy """
    return _docstring_template.replace("$NAME$", name).replace("$STRAT_OPT$", make_strategy_options_doc(name, strategy_options))

def make_strategy_options_doc(strategy_name, strategy_options):
    """ Generate documentation for the supported strategy options and their defaults """
    doc = ""
    for opt, val in strategy_options.items():
        doc += f"     * {opt}: {val[0]}, default {str(val[1])}. \n"
    doc += "\n"
    return doc

def get_options(strategy_options, options):
    """ Get the strategy-specific options or their defaults from user-supplied strategy_options """
    accepted = list(options.keys()) + ["max_fevals", "time_limit"]
    for key in strategy_options:
        if key not in accepted:
            raise ValueError(f"Unrecognized option {key} in strategy_options")
    assert isinstance(options, OrderedDict)
    return [strategy_options.get(opt, default) for opt, (_, default) in options.items()]


