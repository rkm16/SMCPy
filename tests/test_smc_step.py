import pytest
import numpy as np
from smcpy.particles.smc_step import SMCStep
from smcpy.particles.particle import Particle


@pytest.fixture
def particle_list():
    particle = Particle({'a': 1, 'b': 2}, 0.2, -0.2)
    return 5 * [particle]


@pytest.fixture
def step_tester():
    return SMCStep()


@pytest.fixture
def filled_step(step_tester, particle_list):
    step_tester.add_particle(particle_list)
    return step_tester


def test_type_error_when_particle_not_list(step_tester):
    with pytest.raises(TypeError):
        step_tester.add_particle("Bad param type")


def test_type_error_not_particle_class(step_tester):
    with pytest.raises(TypeError):
        step_tester.add_particle([1, 2, 3])


def test_private_variable_creation(step_tester, particle_list):
    step_tester.add_particle(particle_list)
    assert step_tester._particles == particle_list


def test_get_likes(filled_step):
    assert filled_step.get_likes()[0] == pytest.approx(0.818730753078)


def test_get_log_likes(filled_step):
    assert filled_step.get_log_likes()[0] == -0.2


def test_get_mean(filled_step):
    assert filled_step.get_mean()['a'] == 1.0


def test_get_particles(filled_step, particle_list):
    assert filled_step.get_particles() == particle_list


def test_get_weights(filled_step):
    assert filled_step.get_weights()[0] == 0.2


def test_calcuate_covariance(filled_step):
    filled_step.calculate_covariance() == np.array([[0, 0], [0, 0]])


def test_compute_ess(filled_step):
    assert filled_step.compute_ess() == pytest.approx(5.0)
