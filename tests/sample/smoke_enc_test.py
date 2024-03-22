# -*- coding: utf-8 -*-
import pytest

from deepctr_torch.sample.smoke_enc import SMOTEENC


def test_smokeenc():
    # if version.parse(torch.__version__) >= version.parse("1.1.0") and len(dnn_hidden_units)==0:#todo check version
    #     return
    sen = SMOTEENC(categorical_features=[1])
    assert 1 == 1


if __name__ == "__main__":
    pass
