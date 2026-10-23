#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROS parameter validation helpers.

The ``validate_*`` functions are pure and reusable; the ``*_param`` /
``finite_float`` style helpers additionally read from the ROS parameter
server, preserving the behaviour of the original monolithic script.
"""

import math

import numpy as np
import rospy


def validate_finite_float(name, value, minimum=None, maximum=None,
                          strict_min=False):
    """Validate a single numeric parameter value and return it as float."""
    if isinstance(value, bool):
        raise ValueError(name + "不能是布尔值")
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(name + "必须填写实测数字，不能保留MEASURE") from error
    if not math.isfinite(value):
        raise ValueError(name + " 必须是有限数字")
    if minimum is not None:
        invalid = value <= minimum if strict_min else value < minimum
        if invalid:
            raise ValueError("{} 小于允许下限".format(name))
    if maximum is not None and value > maximum:
        raise ValueError("{} 大于允许上限".format(name))
    return value


def validate_numeric_list(name, value, count):
    """Validate a list/tuple of ``count`` finite numbers."""
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError("{} 必须包含{}个数字".format(name, count))
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as error:
        raise ValueError(name + "必须填写实测数字列表") from error
    if any(isinstance(item, bool) for item in value) or not np.isfinite(result).all():
        raise ValueError(name + " 包含非有限数值或布尔值")
    return result


def finite_float(name, default=None, minimum=None, maximum=None,
                 strict_min=False):
    """Read ``name`` from the parameter server and validate it."""
    value = rospy.get_param(name) if default is None else rospy.get_param(name, default)
    return validate_finite_float(name, value, minimum=minimum,
                                 maximum=maximum, strict_min=strict_min)


def numeric_list(name, value, count):
    """Validate a parameter value that must contain ``count`` numbers."""
    return validate_numeric_list(name, value, count)


def optional_numeric_list(name, count):
    """Read an optional numeric-list parameter; ``[]`` means None."""
    value = rospy.get_param(name, [])
    if value == []:
        return None
    return validate_numeric_list(name, value, count)
