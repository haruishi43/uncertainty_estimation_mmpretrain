#!/usr/bin/env python3

import torch


def nig_nll(gamma, v, alpha, beta, y):
    """Negative Log-Likelihood Loss for Normal Inverse Gamma.

    https://arxiv.org/abs/1910.02600
    """
    two_beta_lambda = 2 * beta * (1 + v)
    t1 = 0.5 * (torch.pi / v).log()
    t2 = alpha * two_beta_lambda.log()
    t3 = (alpha + 0.5) * (v * (y - gamma) ** 2 + two_beta_lambda).log()
    t4 = alpha.lgamma()
    t5 = (alpha + 0.5).lgamma()
    nll = t1 - t2 + t3 + t4 - t5
    return nll.mean()


def nig_reg(gamma, v, alpha, _beta, y):
    """Regularization for Normal Inverse Gamma."""
    reg = (y - gamma).abs() * (2 * v + alpha)
    return reg.mean()


def evidential_regression(dist_params, y, lamb=1.0):
    return nig_nll(*dist_params, y) + lamb * nig_reg(*dist_params, y)
