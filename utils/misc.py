#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   misc.py
@Time    :   2023/11/07 14:37:01
@Author  :   shiqing 
@Version :   Cinnamoroll V1
'''

class argsdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__