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


def assert_output(fn, fn_inputs, fn_expected_outputs):
    assert type(fn_expected_outputs) == OrderedDict

    fn_outputs = OrderedDict()
    fn_inputs_passed_in = copy.deepcopy(fn_inputs)
    fn_raw_out = fn(**fn_inputs_passed_in)

    # Check if inputs have changed
    for input_name, input_value in fn_inputs.items():
        passed_in_unchanged = _is_equal(input_value, fn_inputs_passed_in[input_name])

        assert (
            passed_in_unchanged
        ), f"""Input parameter "{input_name}" has been modified inside the function. The function shouldn\'t modify the function parameters."""

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
        assert isinstance(
            fn_out, type(expected_out)
        ), f'Wrong type for output {out_name}. Got {type(fn_out)}, expected {type(expected_out)}'

        if hasattr(expected_out, 'shape'):
            assert (
                fn_out.shape == expected_out.shape
            ), f'Wrong shape for output {out_name}. Got {fn_out.shape}, expected {expected_out.shape}'
        elif hasattr(expected_out, '__len__'):
            assert len(fn_out) == len(
                expected_out
            ), f'Wrong len for output {out_name}. Got {len(fn_out)}, expected {len(expected_out)}'

        if type(expected_out) == pd.DataFrame:
            assert set(fn_out.columns) == set(
                expected_out.columns
            ), f'Incorrect columns for output {out_name}\nCOLUMNS:          {sorted(fn_out.columns)}\nEXPECTED COLUMNS: {sorted(expected_out.columns)}'

            for column in expected_out.columns:
                assert (
                    fn_out[column].dtype == expected_out[column].dtype
                ), f'Incorrect type for output {out_name}, column {column}\nType:          {fn_out[column].dtype}\nEXPECTED Type: {expected_out[column].dtype}'

            # Sort Columns
            fn_out = fn_out.sort_index(1)
            expected_out = expected_out.sort_index(1)

        if type(expected_out) in {pd.DataFrame, pd.Series}:
            assert set(fn_out.index) == set(
                expected_out.index
            ), f'Incorrect indices for output {out_name}\nINDICES:          {sorted(fn_out.index)}\nEXPECTED INDICES: {sorted(expected_out.index)}'

            # Sort Indices
            fn_out = fn_out.sort_index()
            expected_out = expected_out.sort_index()

        try:
            out_is_close = np.isclose(fn_out, expected_out, equal_nan=True)
        except TypeError:
            out_is_close = fn_out == expected_out
        else:
            if isinstance(expected_out, collections.Iterable):
                out_is_close = out_is_close.all()

        assert out_is_close, err_message
