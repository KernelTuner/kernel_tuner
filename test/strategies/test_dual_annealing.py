import numpy as np

import kernel_tuner



def test_dual_annealing_gives_warning(recwarn):
    """ Uses pytest built-in fixture recwarn to record any warnings this code might throw. """

    kernel_tuner.tune_kernel(
        kernel_name="foo",
        kernel_source="void foo(int) { }",
        problem_size=1,
        arguments=[np.int32(0)],
        tune_params={"block_size_x": range(1, 8)},
        strategy="dual_annealing",
        restrictions="block_size_x < 4",
    )

    for warning in recwarn:
        # Print the actual Warning object / message string
        print(warning.message)

        # Print the category (e.g., <class 'UserWarning'>)
        print(warning.category)

        # Print file and line where it was raised
        print(f"Raised at {warning.filename}:{warning.lineno}")

    # Test that no warnings are thrown
    assert len(recwarn) == 0

