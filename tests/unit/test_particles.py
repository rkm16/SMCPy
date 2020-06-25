import pytest
import numpy as np

from copy import copy

from smcpy.smc.particles import Particles

@pytest.fixture
def dummy_params():
    return {'a': [1] * 10, 'b': [2] * 10}


@pytest.fixture
def dummy_param_array():
    return np.array([[1, 2]] * 10)


@pytest.fixture
def dummy_log_likes():
    return [0.1] * 10


@pytest.fixture
def dummy_log_weights():
    return [0.2] * 10


@pytest.fixture
def particles(dummy_params, dummy_log_likes, dummy_log_weights):
    return Particles(dummy_params, dummy_log_likes, dummy_log_weights)


def test_set_particles(particles, dummy_params, dummy_log_likes,
                       dummy_log_weights, dummy_param_array):

    params = dummy_param_array
    log_likes = np.array(dummy_log_likes).reshape(-1, 1)
    normalized_weights = np.array([0.1] * 10).reshape(-1, 1)
    normalized_log_weights = np.log(normalized_weights)

    assert particles.param_names == ('a', 'b')
    assert particles.num_particles == 10
    np.testing.assert_array_equal(particles.params, params)
    np.testing.assert_array_equal(particles.log_likes, log_likes)
    np.testing.assert_array_equal(particles.log_weights, normalized_log_weights)
    np.testing.assert_array_almost_equal(particles.weights, normalized_weights)


def test_params_value_error():
    params = {'a': [1, 2], 'c': [2], 'b': 4}
    with pytest.raises(ValueError):
        Particles(params, None, None)


def test_params_type_error():
    with pytest.raises(TypeError):
        Particles([], None, None)


@pytest.mark.parametrize('log_likes', (4, [], [1], np.ones(3), np.ones(5)))
def test_log_likes_value_errors(log_likes):
    params = {'a': [5] * 4}
    with pytest.raises(ValueError):
        Particles(params, log_likes, None)


@pytest.mark.parametrize('log_weights', (4, [], [1], np.ones(3), np.ones(5)))
def test_log_weights_value_errors(log_weights):
    params = {'a': [5] * 4}
    log_likes = [5] * 4
    with pytest.raises(ValueError):
        Particles(params, log_likes, log_weights)


def test_particles_copy(particles):
    particles_copy = particles.copy()
    assert particles_copy is not particles
    assert isinstance(particles_copy, Particles)


def test_compute_ess(particles, dummy_log_weights):
    expected_norm_log_weights = np.array([0.1] * 10)
    expected_ess = 1 / np.sum(expected_norm_log_weights ** 2)
    assert particles.compute_ess() == pytest.approx(expected_ess)


@pytest.mark.parametrize('params, weights',
                         (({'a': [1, 2], 'b': [2, 3]}, [1, 1]),
                          ({'a': [1, 5/3], 'b': [4, 2]}, [1, 3])))
def test_compute_mean(params, weights):
    log_likes = np.ones(2)
    expected_means = {'a': 1.5, 'b': 2.5}

    particles = Particles(params, log_likes, np.log(weights))

    assert particles.compute_mean() == expected_means


@pytest.mark.parametrize('params, weights, expected_var',
        (({'a': [1, 2], 'b': [2, 3]}, [1, 1], {'a': 0.5, 'b': 0.5}),
         ({'a': [1, 5/3], 'b': [4, 2]}, [1, 3], {'a': 2/9, 'b':2.})))
def test_compute_variance(params, weights, expected_var):
    log_likes = np.ones(2)

    particles = Particles(params, log_likes, np.log(weights))

    assert particles.compute_variance() == pytest.approx(expected_var)


@pytest.mark.parametrize('params, weights, expected_var',
        (({'a': [1, 2], 'b': [2, 3]}, [1, 1], {'a': 0.5, 'b': 0.5}),
         ({'a': [1, 5/3], 'b': [4, 2]}, [1, 3], {'a': 2/9, 'b':2.})))
def test_get_std_dev(params, weights, expected_var):
    log_likes = np.ones(2)

    particles = Particles(params, log_likes, np.log(weights))

    expected_var['a'] = np.sqrt(expected_var['a'])
    expected_var['b'] = np.sqrt(expected_var['b'])
    assert particles.compute_std_dev() == pytest.approx(expected_var)

def test_compute_covariance(smc_step, particle, mocker):
    # need to prob use this example
    p1 = copy(particle)
    p1.params = {'a': 1.1, 'b': 2.2}
    p2 = copy(particle)
    p2.params = {'a': 1.0, 'b': 2.1}
    p3 = copy(particle)
    p3.params = {'a': 0.8, 'b': 1.9}

    smc_step.particles = [p1, p2, p3]
    mocker.patch.object(smc_step, 'normalize_step_log_weights',
                        return_value=np.array([0.1, 0.7, 0.2]))
    mocker.patch.object(smc_step, 'get_mean',
                        return_value={'a': 0.97, 'b': 2.06})

    scale = 1 / (1 - np.sum(np.array([0.1, 0.7, 0.2]) ** 2))
    expected_cov = np.array([[0.0081, 0.0081], [0.0081, 0.0082]]) * scale
    np.testing.assert_array_almost_equal(smc_step.get_covariance(),
                                         expected_cov)


#@pytest.mark.filterwarnings('ignore: current step')
#def test_covariance_not_positive_definite_is_eye(smc_step, particle, mocker):
#    particle.params = {'a': 1.1, 'b': 2.2}
#    smc_step.particles = [particle] * 3
#    mocker.patch.object(smc_step, 'normalize_step_log_weights',
#                        return_value=np.array([0.1, 0.7, 0.2]))
#    mocker.patch.object(smc_step, 'get_mean', return_value={'a': 1, 'b': 2})
#    mocker.patch.object(smc_step, '_is_positive_definite', return_value=False)
#    np.testing.assert_array_equal(smc_step.get_covariance(), np.eye(2))
#
#


#

#def test_get_params(smc_step, particle):
#    particle.params = {'a': 1}
#    smc_step.particles = [particle] * 3
#    np.testing.assert_array_equal(smc_step.get_params('a'), np.array([1] * 3))
#
#
#def test_get_param_dicts(smc_step, particle):
#    particle.params = {'a': 1, 'b': 2}
#    smc_step.particles = [particle] * 3
#    assert smc_step.get_param_dicts() == [{'a': 1, 'b': 2}] * 3
#
#
#def test_resample(smc_step, particle, mocker):
#
#    p1 = copy(particle)
#    p1.params = {'a': 1}
#    p1.log_weight = 0.1
#    p1.log_like = 0.1
#    p2 = copy(particle)
#    p2.params = {'a': 2}
#    p2.log_weight = 0.1
#    p2.log_like = 0.1
#    p3 = copy(particle)
#    p3.params = {'a': 3}
#    p3.log_weight = 0.1
#    p3.log_like = 0.1
#
#    smc_step.particles = [p1, p2, p3]
#    mocker.patch.object(smc_step, 'normalize_step_log_weights')
#    mocker.patch.object(smc_step, 'get_log_weights',
#                        return_value=np.log([.1, .5, .4]))
#    mocker.patch('numpy.random.uniform', return_value=np.array([1, 0.6, 0.12]))
#    smc_step.resample()
#
#    particles = smc_step.particles
#
#    assert all([particles[0].params == p3.params,
#                particles[1].params == p3.params,
#                particles[2].params == p2.params])
#    assert sum([np.exp(p.log_weight) for p in particles]) == pytest.approx(1)
#
#
#@pytest.mark.parametrize('ess_threshold,call_expected',[(0.1, False),
#                         (0.49, False), (0.51, True), (0.9, True)])
#def test_resample_if_needed(smc_step, ess_threshold, call_expected, mocker):
#    mocker.patch.object(smc_step, 'compute_ess', return_value=0.5)
#    mocker.patch.object(smc_step, 'resample')
#    smc_step.resample_if_needed(ess_threshold)
#    assert smc_step.resample.called is call_expected
#
#
#def test_update_log_weights(smc_step, particle, mocker):
#    p1 = copy(particle)
#    p1.params = {'a': 1}
#    p1.log_weight = 1
#    p1.log_like = 0.1
#    p2 = copy(particle)
#    p2.params = {'a': 2}
#    p2.log_weight = 1
#    p2.log_like = 0.2
#    smc_step.particles = [p1, p2]
#
#    mocker.patch.object(smc_step, 'normalize_step_log_weights')
#
#    smc_step.update_log_weights(delta_phi=0.1)
#
#    smc_step.normalize_step_log_weights.assert_called()
#    assert smc_step.particles[0].log_weight == 1.01
#    assert smc_step.particles[1].log_weight == 1.02
