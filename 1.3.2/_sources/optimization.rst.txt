.. _optimizations:

Optimization strategies
-----------------------

Kernel Tuner supports many optimization strategies that accelerate the auto-tuning search process. By default, Kernel Tuner 
uses 'brute force' tuning, which means that Kernel Tuner will try all possible combinations of all values of all tunable 
parameters. Even with simple kernels this form of tuning can become prohibitively slow and a waste of time and energy.

To enable optimization strategies in Kernel Tuner, you simply have to supply the name of the strategy you'd like to use using 
the ``strategy=`` optional argument of ``tune_kernel()``. Kernel Tuner currently supports the following strategies:

 * "basinhopping" Basin Hopping
 * "bayes_opt" Bayesian Optimization
 * "brute_force" (default) iterates through the entire search space
 * "dual annealing" dual annealing
 * "diff_evo" differential evolution
 * "firefly_algorithm" firefly algorithm strategy
 * "genetic_algorithm" a genetic algorithm optimization
 * "greedy_ils" greedy randomized iterative local search
 * "greedy_mls" greedy randomized multi-start local search
 * "minimize" uses a local minimization algorithm
 * "mls" best-improvement multi-start local search
 * "ordered_greedy_mls" multi-start local search that uses a fixed order
 * "pso" particle swarm optimization
 * "random_sample" takes a random sample of the search space
 * "simulated_annealing" simulated annealing strategy

Most strategies have some mechanism built in to detect when to stop tuning, which may be controlled through specific 
parameters that can be passed to the strategies using the ``strategy_options=`` optional argument of ``tune_kernel()``. You 
can also override whatever internal stop criterion the strategy uses, and set either a time limit in seconds (using ``time_limit=``) or a maximum 
number of unique function evaluations (using ``max_fevals=``).

To give an example, one could simply add these two arguments to any code calling ``tune_kernel()``:

.. code-block:: python

    results, env = tune_kernel("vector_add", kernel_string, size, args, tune_params,
                               strategy="random_sample",
                               strategy_options=dict(max_fevals=5))


A 'unique function evaluation' corresponds to the first time that Kernel Tuner tries to compile and benchmark a parameter 
configuration that has been selected by the optimization strategy. If you are continuing from a previous tuning session using 
cache files, serving a value from the cache for the first time in the run also counts as a function evaluation for the strategy.
Only unique function evaluations are counted, so the second time a parameter configuration is selected by the strategy it is served from the 
cache, but not counted as a unique function evaluation.

All optimization algorithms, except for brute_force, random_sample, and bayes_opt, allow the user to specify an initial guess or 
starting point for the optimization, called ``x0``. This can be passed to the strategy using the ``strategy_options=`` dictionary with ``"x0"`` as key and
a list of values for each parameter in tune_params to note the starting point. For example, for a kernel that has parameters ``block_size_x`` (64, 128, 256)
and ``tile_size_x`` (1,2,3), one could pass ``strategy_options=dict(x0=[128,2])`` to ``tune_kernel()`` to make sure the strategy starts from
the configuration with ``block_size_x=128, tile_size_x=2``. The order in the ``x0`` list should match the order in the tunable parameters dictionary.

Below all the strategies are listed with their strategy-specific options that can be passed in a dictionary to the ``strategy_options=`` argument
of ``tune_kernel()``.


kernel_tuner.strategies.basinhopping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: kernel_tuner.strategies.basinhopping
    :members:

kernel_tuner.strategies.bayes_opt
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: kernel_tuner.strategies.bayes_opt
    :members:

kernel_tuner.strategies.brute_force
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: kernel_tuner.strategies.brute_force
    :members:

kernel_tuner.strategies.diff_evo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: kernel_tuner.strategies.diff_evo
    :members:

kernel_tuner.strategies.dual_annealing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: kernel_tuner.strategies.dual_annealing
    :members:

kernel_tuner.strategies.firefly_algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: kernel_tuner.strategies.firefly_algorithm
    :members:

kernel_tuner.strategies.genetic_algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: kernel_tuner.strategies.genetic_algorithm
    :members:

kernel_tuner.strategies.greedy_ils
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: kernel_tuner.strategies.greedy_ils
    :members:

kernel_tuner.strategies.greedy_mls
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: kernel_tuner.strategies.greedy_mls
    :members:

kernel_tuner.strategies.minimize
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: kernel_tuner.strategies.minimize
    :members:

kernel_tuner.strategies.mls
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: kernel_tuner.strategies.mls
    :members:

kernel_tuner.strategies.ordered_greedy_mls
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: kernel_tuner.strategies.ordered_greedy_mls
    :members:

kernel_tuner.strategies.pso
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: kernel_tuner.strategies.pso
    :members:

kernel_tuner.strategies.random_sample
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: kernel_tuner.strategies.random_sample
    :members:

kernel_tuner.strategies.simulated_annealing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: kernel_tuner.strategies.simulated_annealing
    :members:


