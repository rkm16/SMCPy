import numpy as np

from .log_likelihoods import BaseLogLike
from .utils.log_sum import logsum


class ApproxHierarch(BaseLogLike):

    def __init__(self, model, data, args):
        super().__init__(model, data, args)

    def __call__(self, inputs):
        self._override_model_wrapper()
        log_like = np.full((inputs.shape[0], len(self._data)), -np.inf)
        for i, d in enumerate(self._data):
            log_conditionals = self._get_output(d)
            log_priors = self._args[1][i]
            mll = self._args[0][i]
            log_like[:, i] = mll + logsum(log_conditionals - log_priors)
        return log_like.sum(axis=1).reshape(-1, 1)

    def _override_model_wraper(self):
        model = self._model(inputs)
        self.set_model_wrapper(lambda dummy, x: model(x))

    @staticmethod
    def logsum(Z):
        '''
        Assumes summing over columns.
        '''
        Z = -np.sort(-np.array(Z), axis=1) # descending over columns
        Z0 = Z[:, [0]]
        Z_shifted = Z[:, 1:] - Z0
        return Z0.flatten() + np.log(1 + np.sum(np.exp(Z_shifted), axis=1))
