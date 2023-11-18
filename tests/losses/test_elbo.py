import unittest

import numpy as np
import torch
from src.losses.dmm_elbo import DMMContinuousELBO, DMMBinaryELBO


class SimpleModel:
    def __init__(self):
        batch_size = 32
        n_time_steps = 10
        latent_dim = 1
        obs_dim = 1

        def emission_model(*args, **kwargs):
            # mean=-0.5, log_var=log(0.5)
            return [
                torch.zeros((batch_size, n_time_steps, latent_dim)) - 0.5,
                torch.log(torch.zeros((batch_size, n_time_steps, obs_dim)) + 0.5),
            ]

        def transition_model(*args, **kwargs):
            # mean=0.5, log_var=log(2)
            return [
                torch.zeros((batch_size, n_time_steps, latent_dim)) + 0.5,
                torch.log(torch.zeros((batch_size, n_time_steps, latent_dim)) + 2),
            ]

        def inference_model(*args, **kwargs):
            # zero mean, unit variance (zero log variance)
            return [
                torch.zeros((batch_size, n_time_steps, latent_dim)),
                torch.zeros((batch_size, n_time_steps, latent_dim)),
            ], None

        self.emission_model = emission_model
        self.transition_model = transition_model
        self.inference_model = inference_model


class TestDMMELBO(unittest.TestCase):
    def test_cont_elbo(self):
        annealing_params = {"enabled": False, "warm_up": 0, "n_epochs_for_full": 100}
        elbo = DMMContinuousELBO(annealing_params)

        observations = torch.zeros((32, 10, 1))
        data = [observations]
        model = SimpleModel()

        # Test loss
        loss, to_log = elbo(model, data, 0)
        self.assertAlmostEquals(
            to_log["log_p observation loss"], 0.82236492633, delta=1e-3
        )
        self.assertAlmostEquals(to_log["KL loss"], 0.14316623, delta=1e-3)

        self.assertAlmostEquals(loss, 0.82236492633 + 0.14316623, delta=1e-3)

    def test_cont_elbo_rmse_eval(self):
        annealing_params = {"enabled": False, "warm_up": 0, "n_epochs_for_full": 100}
        elbo = DMMContinuousELBO(annealing_params, rmse_eval_latent=False)

        observations = torch.zeros((32, 10, 1))
        data = [observations]
        model = SimpleModel()

        # TODO: Test rmse window eval

    def test_binary_elbo_loss(self):
        annealing_params = {"enabled": False, "warm_up": 0, "n_epochs_for_full": 100}
        elbo = DMMBinaryELBO(annealing_params)

        observations = torch.zeros((32, 10, 1))
        observations[:, 5:] = 1
        data = [observations]
        model = SimpleModel()
        model.emission_model = (
            lambda *args, **kwargs: torch.zeros_like(observations) + 0.5
        )

        # Test loss
        loss, to_log = elbo(model, data, 0)
        self.assertAlmostEquals(to_log["log_p observation loss"], 0.693147, delta=1e-3)
        self.assertAlmostEquals(to_log["KL loss"], 0.14316623, delta=1e-3)

        self.assertAlmostEquals(loss, 0.693147 + 0.14316623, delta=1e-3)


if __name__ == "__main__":
    unittest.main()
