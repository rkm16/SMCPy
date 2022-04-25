import numpy as np
import pytest

from smcpy.hierarch_log_likelihoods import ApproxHierarch
from smcpy.log_likelihoods import BaseLogLike


def test_import_approx_hierarch():
    like = ApproxHierarch(1, 2, 3)

    assert like._model == 1
    assert like._data == 2
    assert like._args == 3
    assert isinstance(like, BaseLogLike)


def test_approx_hierarch_call(mocker):
    n_particles = 15
    nre1 = 3
    nre2 = 3
    nre3 = 5

    conditionals = [np.tile(np.arange(1, nre1 + 1), (n_particles, 1)),
                    np.tile(np.arange(1, nre2 + 1), (n_particles, 1)),
                    np.tile(np.arange(1, nre3 + 1), (n_particles, 1))]
    model = mocker.Mock(side_effect=[np.log(conditionals[0]),
                                     np.log(conditionals[1]),
                                     np.log(conditionals[2])])
    data_log_priors = [np.log([5] * nre1),
                       np.log([7] * nre2),
                       np.log([9] * nre3)]
    data = [np.array([[1, 2]] * nre1),
            np.array([[2, 3]] * nre2),
            np.array([[3, 4]] * nre3)]
    model_class = mocker.Mock(return_value=model)
    marginal_log_likes = np.log([1, 2, 3])
    args = (marginal_log_likes, data_log_priors)
    inputs = np.ones((n_particles, 5))

    expected_like = np.product([1 * 6 / 5, 2 * 6 / 7, 3 * 15 / 9])
    expected_log_like = np.log(expected_like)
    expected_log_likes = np.tile(expected_log_like, (n_particles, 1))

    like = ApproxHierarch(model_class, data, args)
    log_likes = like(inputs)

    model_class.assert_called_once_with(inputs)
    for i, d in enumerate(data):
        call = model.call_args_list[i][0][0]
        np.testing.assert_array_equal(call, d)

    np.testing.assert_array_equal(log_likes, expected_log_likes)
