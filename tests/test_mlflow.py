# Copyright (c) 2021 Philip May
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

from hpoflow import normalize_mlflow_entry_name, normalize_mlflow_entry_names_in_dict


def test_normalize_mlflow_entry_name():
    entry_name = "AaZz10_-. /ÄäÖöÜüß#(!)}=%§"
    result = normalize_mlflow_entry_name(entry_name)
    assert result == "AaZz10_-. /AeaeOeoeUeuess________"


def test_normalize_mlflow_entry_names_in_dict():
    dct = {
        "AaZz10_-. /ÄäÖöÜüß#(!)}=%§": 6,
        "somethingÄ": 8,
        "ok key": 9,
    }
    result = normalize_mlflow_entry_names_in_dict(dct)

    assert "AaZz10_-. /AeaeOeoeUeuess________" in result
    assert "somethingAe" in result
    assert "ok key" in result
    assert len(result) == 3
