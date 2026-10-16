"""Differentiable observables (cf. ood_ld observables.py):
    phi_fmt(o) = log Σ_{a∈A_o} p(a|x_o)                 willingness to answer in the required form
    phi_ans(o) = log p(a*|x_o) − log Σ_{a≠a*} p(a|x_o)   correct vs wrong inside the option set
"""
from __future__ import annotations


def phi_fmt(logprobs, option_ids):
    raise NotImplementedError


def phi_ans(logprobs, option_ids, answer_idx: int):
    raise NotImplementedError
