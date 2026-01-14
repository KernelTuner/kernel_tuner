"""Module for parsing and evaluating restrictions."""

from inspect import getsource
from types import FunctionType
from typing import Union
import ast
import logging
import numpy as np
import re
import textwrap

from constraint import (
    AllDifferentConstraint,
    AllEqualConstraint,
    Constraint,
    ExactSumConstraint,
    FunctionConstraint,
    InSetConstraint,
    MaxProdConstraint,
    MaxSumConstraint,
    MinProdConstraint,
    MinSumConstraint,
    NotInSetConstraint,
    SomeInSetConstraint,
    SomeNotInSetConstraint,
)


def check_restriction(restrict, params: dict) -> bool:
    """Check whether a configuration meets a search space restriction."""
    # if it's a python-constraint, convert to function and execute
    if isinstance(restrict, Constraint):
        restrict = convert_constraint_restriction(restrict)
        return restrict(list(params.values()))
    # if it's a string, fill in the parameters and evaluate
    elif isinstance(restrict, str):
        return eval(replace_param_occurrences(restrict, params))
    # if it's a function, call it
    elif callable(restrict):
        return restrict(**params)
    # if it's a tuple, use only the parameters in the second argument to call the restriction
    elif (
        isinstance(restrict, tuple)
        and len(restrict) in (2, 3)
        and callable(restrict[0])
        and isinstance(restrict[1], (list, tuple))
    ):
        # unpack the tuple
        if len(restrict) == 2:
            restrict, selected_params = restrict
        else:
            restrict, selected_params, source = restrict
        # look up the selected parameters and their value
        selected_params = dict((key, params[key]) for key in selected_params)
        # call the restriction
        if isinstance(restrict, Constraint):
            restrict = convert_constraint_restriction(restrict)
            return restrict(list(selected_params.values()))
        else:
            return restrict(**selected_params)
    # otherwise, raise an error
    else:
        raise ValueError(f"Unknown restriction type {type(restrict)} ({restrict})")


def check_restrictions(restrictions, params: dict, verbose: bool) -> bool:
    """Check whether a configuration meets the search space restrictions."""
    if callable(restrictions):
        valid = restrictions(params)
        if not valid and verbose is True:
            print(f"skipping config {get_instance_string(params)}, reason: config fails restriction")
        return valid
    valid = True
    for restrict in restrictions:
        # Check the type of each restriction and validate accordingly. Re-implement as a switch when Python >= 3.10.
        try:
            valid = check_restriction(restrict, params)
            if not valid:
                break
        except ZeroDivisionError:
            logging.debug(f"Restriction {restrict} with configuration {get_instance_string(params)} divides by zero.")
    if not valid and verbose is True:
        print(f"skipping config {get_instance_string(params)}, reason: config fails restriction {restrict}")
    return valid


def convert_constraint_restriction(restrict: Constraint):
    """Convert the python-constraint to a function for backwards compatibility."""
    if isinstance(restrict, FunctionConstraint):

        def f_restrict(p):
            return restrict._func(*p)

    elif isinstance(restrict, AllDifferentConstraint):

        def f_restrict(p):
            return len(set(p)) == len(p)

    elif isinstance(restrict, AllEqualConstraint):

        def f_restrict(p):
            return all(x == p[0] for x in p)

    elif isinstance(restrict, MaxProdConstraint):

        def f_restrict(p):
            return np.prod(p) <= restrict._maxprod

    elif isinstance(restrict, MinProdConstraint):

        def f_restrict(p):
            return np.prod(p) >= restrict._minprod

    elif isinstance(restrict, MaxSumConstraint):

        def f_restrict(p):
            return sum(p) <= restrict._maxsum

    elif isinstance(restrict, ExactSumConstraint):

        def f_restrict(p):
            return sum(p) == restrict._exactsum

    elif isinstance(restrict, MinSumConstraint):

        def f_restrict(p):
            return sum(p) >= restrict._minsum

    elif isinstance(restrict, (InSetConstraint, NotInSetConstraint, SomeInSetConstraint, SomeNotInSetConstraint)):
        raise NotImplementedError(
            f"Restriction of the type {type(restrict)} is explicitly not supported in backwards compatibility mode, because the behaviour is too complex. Please rewrite this constraint to a function to use it with this algorithm."
        )
    else:
        raise TypeError(f"Unrecognized restriction {restrict}")
    return f_restrict


def parse_restrictions(
    restrictions: list[str], tune_params: dict, monolithic=False, format=None
) -> list[tuple[Union[Constraint, str], list[str]]]:
    """Parses restrictions from a list of strings into compilable functions and constraints, or a single compilable function (if monolithic is True). Returns a list of tuples of (strings or constraints) and parameters."""
    # rewrite the restrictions so variables are singled out
    regex_match_variable = r"([a-zA-Z_$][a-zA-Z_$0-9]*)"

    def replace_params(match_object):
        key = match_object.group(1)
        if key in tune_params and format != "pyatf":
            param = str(key)
            return "params[params_index['" + param + "']]"
        else:
            return key

    def replace_params_split(match_object):
        # careful: has side-effect of adding to set `params_used`
        key = match_object.group(1)
        if key in tune_params:
            param = str(key)
            params_used.add(param)
            return param
        else:
            return key

    # remove functionally duplicate restrictions (preserves order and whitespace)
    if all(isinstance(r, str) for r in restrictions):
        # clean the restriction strings to functional equivalence
        restrictions_cleaned = [r.replace(" ", "") for r in restrictions]
        restrictions_cleaned_unique = list(dict.fromkeys(restrictions_cleaned))  # dict preserves order
        # get the indices of the unique restrictions, use these to build a new list of restrictions
        restrictions_unique_indices = [restrictions_cleaned.index(r) for r in restrictions_cleaned_unique]
        restrictions = [restrictions[i] for i in restrictions_unique_indices]

    # create the parsed restrictions
    if not monolithic:
        # split into functions that only take their relevant parameters
        parsed_restrictions = list()
        for res in restrictions:
            params_used: set[str] = set()
            parsed_restriction = re.sub(regex_match_variable, replace_params_split, res).strip()
            params_used = list(params_used)
            finalized_constraint = None
            # we must turn it into a general function
            if format is not None and format.lower() == "pyatf":
                finalized_constraint = parsed_restriction
            else:
                finalized_constraint = f"def r({', '.join(params_used)}): return {parsed_restriction} \n"
            parsed_restrictions.append((finalized_constraint, params_used))

        # if pyATF, restrictions that are set on the same parameter must be combined into one
        if format is not None and format.lower() == "pyatf":
            res_dict = dict()
            registered_params = list()
            registered_restrictions = list()
            parsed_restrictions_pyatf = list()
            for param in tune_params.keys():
                registered_params.append(param)
                for index, (res, params) in enumerate(parsed_restrictions):
                    if index in registered_restrictions:
                        continue
                    if all(p in registered_params for p in params):
                        if param not in res_dict:
                            res_dict[param] = (list(), list())
                        res_dict[param][0].append(res)
                        res_dict[param][1].extend(params)
                        registered_restrictions.append(index)
            # combine multiple restrictions into one
            for res_tuple in res_dict.values():
                res, params_used = res_tuple
                params_used = list(
                    dict.fromkeys(params_used)
                )  # param_used should only contain unique, dict preserves order
                parsed_restrictions_pyatf.append(
                    (f"def r({', '.join(params_used)}): return ({') and ('.join(res)}) \n", params_used)
                )
            parsed_restrictions = parsed_restrictions_pyatf
    else:
        # create one monolithic function
        parsed_restrictions = ") and (".join(
            [re.sub(regex_match_variable, replace_params, res) for res in restrictions]
        )

        # tidy up the code by removing the last suffix and unnecessary spaces
        parsed_restrictions = "(" + parsed_restrictions.strip() + ")"
        parsed_restrictions = " ".join(parsed_restrictions.split())

        # provide a mapping of the parameter names to the index in the tuple received
        params_index = dict(zip(tune_params.keys(), range(len(tune_params.keys()))))

        if format == "pyatf":
            parsed_restrictions = [
                (
                    f"def restrictions({', '.join(params_index.keys())}): return {parsed_restrictions} \n",
                    list(tune_params.keys()),
                )
            ]
        else:
            parsed_restrictions = [
                (
                    f"def restrictions(*params): params_index = {params_index}; return {parsed_restrictions} \n",
                    list(tune_params.keys()),
                )
            ]

    return parsed_restrictions


def get_all_lambda_asts(func):
    """Extracts the AST nodes of all lambda functions defined on the same line as func.

    Args:
        func: A lambda function object.

    Returns:
        A list of all ast.Lambda node objects on the line where func is defined.

    Raises:
        ValueError: If the source can't be retrieved or no lambda is found.
    """
    res = []
    try:
        source = getsource(func)
        source = textwrap.dedent(source).strip()
        parsed = ast.parse(source)

        # Find the Lambda node
        for node in ast.walk(parsed):
            if isinstance(node, ast.Lambda):
                res.append(node)
        if not res:
            raise ValueError(f"No lambda node found in the source {source}.")
    except SyntaxError:
        """ Ignore syntax errors on the lambda """
        return res
    except OSError:
        raise ValueError("Could not retrieve source. Is this defined interactively or dynamically?")
    return res


class InvalidLambdaError(Exception):
    def __str__(self):
        return "lambda could not be parsed by Kernel Tuner"


class ConstraintLambdaTransformer(ast.NodeTransformer):
    """Replaces any `NAME['string']` subscript with just `'string'`, if `NAME`
    matches the lambda argument name.
    """
    def __init__(self, dict_arg_name):
        self.dict_arg_name = dict_arg_name

    def visit_Name(self, node):
        # If we find a Name node that is not part of a Subscript expression, then
        # we throw an exception. This happens when a lambda contains a captured
        # variable or calls a function. In these cases, we cannot transform the
        # lambda into a string so we just exit the ast transformer.
        raise InvalidLambdaError()

    def visit_Subscript(self, node):
        # We only replace subscript expressions of the form <dict_arg_name>['some_string']
        if (isinstance(node.value, ast.Name)
                and node.value.id == self.dict_arg_name
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)):
            # Replace `dict_arg_name['some_key']` with the string used as key
            return ast.Name(node.slice.value)
        return self.generic_visit(node)


def unparse_constraint_lambda(lambda_ast):
    """Parse the lambda function to replace accesses to tunable parameter dict
    Returns string body of the rewritten lambda function
    """
    args = lambda_ast.args

    # Kernel Tuner only allows constraint lambdas with a single argument
    if len(args.args) != 1:
        raise InvalidLambdaError()

    first_arg = args.args[0].arg

    # Create transformer that replaces accesses to tunable parameter dict
    # with simply the name of the tunable parameter
    transformer = ConstraintLambdaTransformer(first_arg)
    new_lambda_ast = transformer.visit(lambda_ast)

    return ast.unparse(new_lambda_ast.body).strip()


def convert_constraint_lambdas(restrictions):
    """Extract and convert all constraint lambdas from the restrictions"""
    res = []
    for c in restrictions:
        if isinstance(c, (str, Constraint)):
            res.append(c)
        elif callable(c):
            try:
                lambda_asts = get_all_lambda_asts(c)
                res += [unparse_constraint_lambda(lambda_ast) for lambda_ast in lambda_asts]
            except (InvalidLambdaError, ValueError):
                res.append(c)   # it's just a plain function, not a lambda


    result = list(set(res))
    if not len(result) == len(restrictions):
        raise ValueError("An error occured when parsing restrictions. If you mix lambdas and string-based restrictions, please define the lambda first.")

    return result


def compile_restrictions(
    restrictions: list, tune_params: dict, monolithic=False, format=None
) -> list[tuple[Union[str, FunctionType], list[str], Union[str, None]]]:
    """Parses restrictions from a list of strings into a list of strings or Functions and parameters used and source, or a single Function if monolithic is true."""
    restrictions = convert_constraint_lambdas(restrictions)

    # filter the restrictions to get only the strings
    restrictions_str, restrictions_ignore = [], []
    for r in restrictions:
        if isinstance(r, str):
            restrictions_str.append(r)
        else:
            restrictions_ignore.append(r)

    if len(restrictions_str) == 0:
        return restrictions_ignore

    # parse the strings
    parsed_restrictions = parse_restrictions(restrictions_str, tune_params, monolithic=monolithic, format=format)

    # compile the parsed restrictions into a function
    compiled_restrictions: list[tuple] = list()
    for restriction, params_used in parsed_restrictions:
        if isinstance(restriction, str):
            # if it's a string, parse it to a function
            code_object = compile(restriction, "<string>", "exec")
            func = FunctionType(code_object.co_consts[0], globals())
            compiled_restrictions.append((func, params_used, restriction))
        elif isinstance(restriction, Constraint):
            # otherwise it already is a Constraint, pass it directly
            compiled_restrictions.append((restriction, params_used, None))
        else:
            raise ValueError(f"Restriction {restriction} is neither a string or Constraint {type(restriction)}")

    # return the restrictions and used parameters
    if len(restrictions_ignore) == 0:
        return compiled_restrictions

    # use the required parameters or add an empty tuple for unknown parameters of ignored restrictions
    noncompiled_restrictions = []
    for r in restrictions_ignore:
        if isinstance(r, tuple) and len(r) == 2 and isinstance(r[1], (list, tuple)):
            restriction, params_used = r
            noncompiled_restrictions.append((restriction, params_used, restriction))
        else:
            noncompiled_restrictions.append((r, [], r))
    return noncompiled_restrictions + compiled_restrictions


def parse_restrictions_pysmt(restrictions: list, tune_params: dict, symbols: dict):
    """Parses restrictions from a list of strings into PySMT compatible restrictions."""
    from pysmt.shortcuts import (
        GE,
        GT,
        LE,
        LT,
        And,
        Bool,
        Div,
        Equals,
        Int,
        Minus,
        Or,
        Plus,
        Pow,
        Real,
        String,
        Times,
    )

    regex_match_variable = r"([a-zA-Z_$][a-zA-Z_$0-9]*)"

    boolean_comparison_mapping = {
        "==": Equals,
        "<": LT,
        "<=": LE,
        ">=": GE,
        ">": GT,
        "&&": And,
        "||": Or,
    }

    operators_mapping = {"+": Plus, "-": Minus, "*": Times, "/": Div, "^": Pow}

    constant_init_mapping = {
        "int": Int,
        "float": Real,
        "str": String,
        "bool": Bool,
    }

    def replace_params(match_object):
        key = match_object.group(1)
        if key in tune_params:
            return 'params["' + key + '"]'
        else:
            return key

    # rewrite the restrictions so variables are singled out
    parsed = [re.sub(regex_match_variable, replace_params, res) for res in restrictions]
    # ensure no duplicates are in the list
    parsed = list(set(parsed))
    # replace ' or ' and ' and ' with ' || ' and ' && '
    parsed = list(r.replace(" or ", " || ").replace(" and ", " && ") for r in parsed)

    # compile each restriction by replacing parameters and operators with their PySMT equivalent
    compiled_restrictions = list()
    for parsed_restriction in parsed:
        words = parsed_restriction.split(" ")

        # make a forward pass over all the words to organize and substitute
        add_next_var_or_constant = False
        var_or_constant_backlog = list()
        operator_backlog = list()
        operator_backlog_left_right = list()
        boolean_backlog = list()
        for word in words:
            if word.startswith("params["):
                # if variable
                varname = word.replace('params["', "").replace('"]', "")
                var = symbols[varname]
                var_or_constant_backlog.append(var)
            elif word in boolean_comparison_mapping:
                # if comparator
                boolean_backlog.append(boolean_comparison_mapping[word])
                continue
            elif word in operators_mapping:
                # if operator
                operator_backlog.append(operators_mapping[word])
                add_next_var_or_constant = True
                continue
            else:
                # if constant: evaluate to check if it is an integer, float, etc. If not, treat it as a string.
                try:
                    constant = ast.literal_eval(word)
                except ValueError:
                    constant = word
                # convert from Python type to PySMT equivalent
                type_instance = constant_init_mapping[type(constant).__name__]
                var_or_constant_backlog.append(type_instance(constant))
            if add_next_var_or_constant:
                right, left = var_or_constant_backlog.pop(-1), var_or_constant_backlog.pop(-1)
                operator_backlog_left_right.append((left, right, len(var_or_constant_backlog)))
                add_next_var_or_constant = False
                # reserve an empty spot for the combined operation to preserve the order
                var_or_constant_backlog.append(None)

        # for each of the operators, instantiate them with variables or constants
        for i, operator in enumerate(operator_backlog):
            # merges the first two symbols in the backlog into one
            left, right, new_index = operator_backlog_left_right[i]
            assert (
                var_or_constant_backlog[new_index] is None
            )  # make sure that this is a reserved spot to avoid changing the order
            var_or_constant_backlog[new_index] = operator(left, right)

        # for each of the booleans, instantiate them with variables or constants
        compiled = list()
        assert len(boolean_backlog) <= 1, "Max. one boolean operator per restriction."
        for boolean in boolean_backlog:
            left, right = var_or_constant_backlog.pop(0), var_or_constant_backlog.pop(0)
            compiled.append(boolean(left, right))

        # add the restriction to the list of restrictions
        compiled_restrictions.append(compiled[0])

    return And(compiled_restrictions)
