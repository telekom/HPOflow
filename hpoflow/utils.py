# Copyright (c) 2021 Philip May
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Util functionality and tools."""


import logging
import warnings


_logger = logging.getLogger(__name__)


def func_no_exception_caller(func, *args, **kwargs):
    """Delegate the function call and log exceptions.

    This function catches all exceptions and just logs them.
    """
    try:
        func(*args, **kwargs)
    except Exception as e:
        error_msg = "Exception raised calling {}! With args: {} kwargs: {} exception: {}".format(
            func.__name__, args, kwargs, e
        )
        _logger.error(error_msg, exc_info=True)
        warnings.warn(error_msg, RuntimeWarning)
