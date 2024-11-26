"""Bayesian Optimization implementation using BO Torch and transfer learning with RGPE."""

try:
    import torch
    from botorch.acquisition import LogExpectedImprovement
    from botorch.optim.optimize import optimize_acqf_discrete
    from torch import Tensor
    bayes_opt_present = True
except ImportError:
    bayes_opt_present = False

from math import ceil, sqrt

import numpy as np

from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies.bayes_opt_BOTorch import BayesianOptimization
from kernel_tuner.util import StopCriterionReached


def tune(searchspace: Searchspace, runner, tuning_options):
    """The entry function for tuning a searchspace using this algorithm."""
    max_fevals = tuning_options.strategy_options.get("max_fevals", 100)
    bo = BayesianOptimizationTransfer(searchspace, runner, tuning_options)
    return bo.run(max_fevals)

class BayesianOptimizationTransfer(BayesianOptimization):
    """Bayesian Optimization class with transfer learning."""

    def __init__(self, searchspace: Searchspace, runner, tuning_options):
        super().__init__(searchspace, runner, tuning_options)

        # set up the data and model for each transfer learning base task
        self.searchspaces_transfer_learning: list[Searchspace] = []
        self.inputs_transfer_learning: list[Tensor] = []
        self.outcomes_transfer_learning: list[Tensor] = []
        for tl_cache in tuning_options.transfer_learning_caches:
            print(f"Importing transfer learning for {tl_cache["kernel_name"]}-{tl_cache['device_name']}")
            # construct the searchspace for this task
            tensor_kwargs = searchspace.tensor_kwargs
            tl_searchspace = Searchspace(None, None, None, from_cache=tl_cache)
            tl_searchspace.initialize_tensorspace(**tensor_kwargs)
            self.searchspaces_transfer_learning.append(tl_searchspace)

            # get the inputs and outcomes for this task
            inputs = []
            outcomes = []
            for c in tl_cache["cache"].values():
                result = c[tuning_options.objective]
                if self.is_valid_result(result):
                    config = tuple(c[p] for p in tl_searchspace.tune_params.keys())
                    inputs.append(tl_searchspace.param_config_to_tensor(config))
                    if not self.maximize:
                        result = -result
                    outcomes.append(result)
            tl_inputs = torch.stack(inputs).to(tl_searchspace.tensor_device)
            tl_outcomes = torch.tensor(outcomes, **tensor_kwargs).unsqueeze(-1)
            assert tl_inputs.shape[0] == tl_outcomes.shape[0]
            self.inputs_transfer_learning.append(tl_inputs)
            self.outcomes_transfer_learning.append(tl_outcomes)
    
    def run(self, max_fevals: int, max_batch_size=2048):
        """Run the Bayesian Optimization loop for at most `max_fevals`."""
        try:
            if not self.initial_sample_taken:
                self.initial_sample()
            model, mll = self.get_model_and_likelihood(self.searchspace, self.train_X, self.train_Y, self.train_Yvar)
            fevals_left = max_fevals - self.initial_sample_size

            # create array to gradually reduce number of optimization spaces as fewer fevals are left
            tensorspace_size = self.searchspace_tensors.size(0)
            reserve_final_loops = min(3, fevals_left)   # reserve some loops at the end that are never split
            fevals_left -= reserve_final_loops
            num_loops = min(max(round(sqrt(fevals_left*2)), 3), fevals_left)  # set the number of loops for the array
            avg_optimization_spaces = max(round(sqrt(tensorspace_size / max_batch_size)), 1)  # set the average number of optimization spaces
            numspace = np.geomspace(start=avg_optimization_spaces, stop=0.1, num=num_loops)
            nums_optimization_spaces = np.clip(np.round(numspace * (fevals_left / numspace.sum())), a_min=1, a_max=None)
            # if there's a discrepency, add or subtract the difference from the first number
            if np.sum(nums_optimization_spaces) != fevals_left:
                nums_optimization_spaces[0] += fevals_left - np.sum(nums_optimization_spaces)
            nums_optimization_spaces = np.concatenate([nums_optimization_spaces, np.full(reserve_final_loops, 1)])
            fevals_left += reserve_final_loops

            # get the number of top configurations to select from transfer learning caches
            num_tl_caches = len(self.outcomes_transfer_learning)
            use_top_configs_until_loop = np.argmax(nums_optimization_spaces < num_tl_caches+1)  # stop after we have fewer num_optimization_spaces than caches - because with more caches the ACQF will not be used as much
            num_top_configs = sum([ceil(n/(num_tl_caches+1)) for n in nums_optimization_spaces[:use_top_configs_until_loop]])

            # select the top configurations for each of the transfer learning caches
            top_configs = [list() for _ in self.outcomes_transfer_learning]
            for tl_index, tl_outcomes in enumerate(self.outcomes_transfer_learning):
                top_configs[tl_index] = self.inputs_transfer_learning[tl_outcomes.argmax()[:num_top_configs]] # TODO check if correct

            # # if there are duplicate configurations, move them up and make sure there are only unique configs
            # duplicate configurations are inserted at the num_configs-(sum of indices)th index
            # # TODO

            # Bayesian optimization loop
            for loop_i, num_optimization_spaces in enumerate(nums_optimization_spaces):
                num_optimization_spaces = round(min(num_optimization_spaces, fevals_left))

                # calculate how many of the optimization spaces to optimize using GP
                optimize_with_GP = max(round(num_optimization_spaces/(num_tl_caches+1)), 1)

                # divide the optimization space into random chuncks
                tensorspace_size = self.searchspace_tensors.size(0)
                if num_optimization_spaces <= 1:
                    optimization_spaces = [self.searchspace_tensors]
                else:
                    # shuffle the searchspace
                    shuffled_indices = torch.randperm(tensorspace_size)
                    tensorspace = self.searchspace_tensors[shuffled_indices]
                    optimization_spaces = tensorspace.split(ceil(tensorspace_size / num_optimization_spaces))

                # fit on a Gaussian Process model
                mll = self.fit(mll)

                # define the acquisition functions
                acqf = LogExpectedImprovement(model=model, best_f=self.train_Y.max(), maximize=True)
                
                # optimize acquisition function to find the next evaluation point
                for i, optimization_space in enumerate(optimization_spaces):
                    candidate, _ = optimize_acqf_discrete(
                        acqf, 
                        q=1, 
                        choices=optimization_space,
                        max_batch_size=max_batch_size
                    )
                    
                    # evaluate the new candidate
                    self.evaluate_configs(candidate)
                    fevals_left -= 1

                # reinitialize the models so they are ready for fitting on next iteration
                if loop_i < len(nums_optimization_spaces) - 1:
                    model, mll = self.get_model_and_likelihood(self.searchspace, self.train_X, self.train_Y, self.train_Yvar, state_dict=model.state_dict())
        except StopCriterionReached as e:
            if self.tuning_options.verbose:
                print(e)

        return self.cost_func.results
