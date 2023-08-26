import collections
from collections import OrderedDict
import copy
import pandas as pd
import numpy as np
from datetime import date, timedelta


pd.options.display.float_format = '{:.8f}'.format


def _generate_output_error_msg(fn_name, fn_inputs, fn_outputs, fn_expected_outputs):
    formatted_inputs = []
    formatted_outputs = [
        f'INPUT {input_name}:\n{str(input_value)}\n'
        for input_name, input_value in fn_inputs.items()
    ]
    formatted_outputs.extend(
        f'OUTPUT {output_name}:\n{str(output_value)}\n'
        for output_name, output_value in fn_outputs.items()
    )
    formatted_expected_outputs = [
        f'EXPECTED OUTPUT FOR {expected_output_name}:\n{str(expected_output_value)}\n'
        for expected_output_name, expected_output_value in fn_expected_outputs.items()
    ]
    return 'Wrong value for {}.\n' \
           '{}\n' \
           '{}\n' \
           '{}' \
        .format(
            fn_name,
            '\n'.join(formatted_inputs),
            '\n'.join(formatted_outputs),
            '\n'.join(formatted_expected_outputs))


def _is_equal(x, y):
    is_equal = False

    if isinstance(x, pd.DataFrame) or isinstance(y, pd.Series):
        is_equal = x.equals(y)
    elif isinstance(x, np.ndarray):
        is_equal = np.array_equal(x, y)
    elif isinstance(x, list):
        if len(x) == len(y):
            for x_item, y_item in zip(x, y):
                if not _is_equal(x_item, y_item):
                    break
            else:
                is_equal = True
    else:
        is_equal = x == y

    return is_equal


def project_test(func):
    def func_wrapper(*args):
        result = func(*args)
        print('Tests Passed')
        return result

    return func_wrapper


def generate_random_tickers(n_tickers=None):
    min_ticker_len = 3
    max_ticker_len = 5
    tickers = []

    if not n_tickers:
        n_tickers = np.random.randint(8, 14)

    ticker_symbol_random = np.random.randint(ord('A'), ord('Z')+1, (n_tickers, max_ticker_len))
    ticker_symbol_lengths = np.random.randint(min_ticker_len, max_ticker_len, n_tickers)
    for ticker_symbol_rand, ticker_symbol_length in zip(ticker_symbol_random, ticker_symbol_lengths):
        ticker_symbol = ''.join([chr(c_id) for c_id in ticker_symbol_rand[:ticker_symbol_length]])
        tickers.append(ticker_symbol)

    return tickers


def generate_random_dates(n_days=None):
    if not n_days:
        n_days = np.random.randint(14, 20)

    start_year = np.random.randint(1999, 2017)
    start_month = np.random.randint(1, 12)
    start_day = np.random.randint(1, 29)
    start_date = date(start_year, start_month, start_day)

    return [start_date + timedelta(days=i) for i in range(n_days)]


def assert_structure(received_obj, expected_obj, obj_name):
    assert isinstance(
        received_obj, type(expected_obj)
    ), f'Wrong type for output {obj_name}. Got {type(received_obj)}, expected {type(expected_obj)}'

    if hasattr(expected_obj, 'shape'):
        assert (
            received_obj.shape == expected_obj.shape
        ), f'Wrong shape for output {obj_name}. Got {received_obj.shape}, expected {expected_obj.shape}'
    elif hasattr(expected_obj, '__len__'):
        assert len(received_obj) == len(
            expected_obj
        ), f'Wrong len for output {obj_name}. Got {len(received_obj)}, expected {len(expected_obj)}'

    if type(expected_obj) == pd.DataFrame:
        assert set(received_obj.columns) == set(
            expected_obj.columns
        ), f'Incorrect columns for output {obj_name}\nCOLUMNS:          {sorted(received_obj.columns)}\nEXPECTED COLUMNS: {sorted(expected_obj.columns)}'

        # This is to catch a case where __equal__ says it's equal between different types
        assert {type(i) for i in received_obj.columns} == {
            type(i) for i in expected_obj.columns
        }, f'Incorrect types in columns for output {obj_name}\nCOLUMNS:          {sorted(received_obj.columns)}\nEXPECTED COLUMNS: {sorted(expected_obj.columns)}'

        for column in expected_obj.columns:
            assert (
                received_obj[column].dtype == expected_obj[column].dtype
            ), f'Incorrect type for output {obj_name}, column {column}\nType:          {received_obj[column].dtype}\nEXPECTED Type: {expected_obj[column].dtype}'

    if type(expected_obj) in {pd.DataFrame, pd.Series}:
        assert set(received_obj.index) == set(
            expected_obj.index
        ), f'Incorrect indices for output {obj_name}\nINDICES:          {sorted(received_obj.index)}\nEXPECTED INDICES: {sorted(expected_obj.index)}'

        # This is to catch a case where __equal__ says it's equal between different types
        assert {type(i) for i in received_obj.index} == {
            type(i) for i in expected_obj.index
        }, f'Incorrect types in indices for output {obj_name}\nINDICES:          {sorted(received_obj.index)}\nEXPECTED INDICES: {sorted(expected_obj.index)}'


def does_data_match(obj_a, obj_b):
    if type(obj_a) == pd.DataFrame:
        # Sort Columns
        obj_b = obj_b.sort_index(1)
        obj_a = obj_a.sort_index(1)

    if type(obj_a) in {pd.DataFrame, pd.Series}:
        # Sort Indices
        obj_b = obj_b.sort_index()
        obj_a = obj_a.sort_index()
    try:
        data_is_close = np.isclose(obj_b, obj_a, equal_nan=True)
    except TypeError:
        data_is_close = obj_b == obj_a
    else:
        if isinstance(obj_a, collections.Iterable):
            data_is_close = data_is_close.all()

    return data_is_close


def assert_output(fn, fn_inputs, fn_expected_outputs, check_parameter_changes=True):
    assert type(fn_expected_outputs) == OrderedDict

    if check_parameter_changes:
        fn_inputs_passed_in = copy.deepcopy(fn_inputs)
    else:
        fn_inputs_passed_in = fn_inputs

    fn_raw_out = fn(**fn_inputs_passed_in)

    # Check if inputs have changed
    if check_parameter_changes:
        for input_name, input_value in fn_inputs.items():
            passed_in_unchanged = _is_equal(input_value, fn_inputs_passed_in[input_name])

            assert (
                passed_in_unchanged
            ), f"""Input parameter "{input_name}" has been modified inside the function. The function shouldn\'t modify the function parameters."""

    fn_outputs = OrderedDict()
    if len(fn_expected_outputs) == 1:
        fn_outputs[list(fn_expected_outputs)[0]] = fn_raw_out
    elif len(fn_expected_outputs) > 1:
        assert (
            type(fn_raw_out) == tuple
        ), f'Expecting function to return tuple, got type {type(fn_raw_out)}'
        assert len(fn_raw_out) == len(
            fn_expected_outputs
        ), f'Expected {len(fn_expected_outputs)} outputs in tuple, only found {len(fn_raw_out)} outputs'
        for key_i, output_key in enumerate(fn_expected_outputs.keys()):
            fn_outputs[output_key] = fn_raw_out[key_i]

    err_message = _generate_output_error_msg(
        fn.__name__,
        fn_inputs,
        fn_outputs,
        fn_expected_outputs)

    for fn_out, (out_name, expected_out) in zip(fn_outputs.values(), fn_expected_outputs.items()):
        assert_structure(fn_out, expected_out, out_name)
        correct_data = does_data_match(expected_out, fn_out)

        assert correct_data, err_message
