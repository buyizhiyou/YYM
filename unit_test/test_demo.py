#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test_demo.py
@Time    :   2022/11/08 13:58:18
@Author  :   shiqing 
@Version :   Cinnamoroll V1
'''
import numbers

import numpy as np
import pytest

import sys
sys.path.append("../")

from utils.metircs import mutual_info


class TestDemo():

    def test_mutual_info(self):
        probs = np.random.rand(3,4,10)
        mi = mutual_info(probs)
        print(mi)

    def test_entropy(self):
        assert 1==2


if __name__ == '__main__':
    pytest.main(["test_demo.py::TestDemo::test_mutual_info","-v","-s"])