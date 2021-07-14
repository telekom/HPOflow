# Copyright (c) 2021 Philip May
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

import pytest

from hpoflow.utils import func_no_exception_caller


def test_func_no_exception_caller(caplog):
    def func_with_exception(x, y, z=8):
        raise Exception("Test exception.")

    with pytest.warns(RuntimeWarning):
        func_no_exception_caller(func_with_exception, 3, z=9, y="hello")

    assert "Exception raised calling func_with_exception!" in caplog.text
    assert "With args: (3,) kwargs: {'z': 9, 'y': 'hello'}" in caplog.text
    assert "exception: Test exception." in caplog.text
