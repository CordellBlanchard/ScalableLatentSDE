import unittest
from src.datasets.synthetic import linear_gssm, nonlinear_gssm


class TestSyntheticDataset(unittest.TestCase):
    def test_linear_gssm_1(self):
        dataloaders = linear_gssm(
            n_train_samples=100,
            n_val_samples=10,
            n_test_samples=10,
            n_time_steps=20,
            batch_size=10,
            return_latents=False,
            return_times=False,
        )

        self.assertEqual(len(dataloaders["train"]), 10)
        self.assertEqual(len(dataloaders["val"]), 1)
        self.assertEqual(len(dataloaders["test"]), 1)

        data = next(iter(dataloaders["train"]))
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0].shape, (10, 20, 1))

    def test_linear_gssm_2(self):
        dataloaders = linear_gssm(
            n_train_samples=10000,
            n_val_samples=10000,
            n_test_samples=10000,
            n_time_steps=200,
            batch_size=10000,
            return_latents=True,
            return_times=False,
        )

        self.assertEqual(len(dataloaders["train"]), 1)
        self.assertEqual(len(dataloaders["val"]), 1)
        self.assertEqual(len(dataloaders["test"]), 1)

        data = next(iter(dataloaders["train"]))
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0].shape, (10000, 200, 1))
        self.assertEqual(data[1].shape, (10000, 200, 1))

    def test_linear_gssm_3(self):
        dataloaders = linear_gssm(
            n_train_samples=100,
            n_val_samples=100,
            n_test_samples=100,
            n_time_steps=200,
            batch_size=100,
            return_latents=True,
            return_times=True,
        )

        self.assertEqual(len(dataloaders["train"]), 1)
        self.assertEqual(len(dataloaders["val"]), 1)
        self.assertEqual(len(dataloaders["test"]), 1)

        data = next(iter(dataloaders["train"]))
        self.assertEqual(len(data), 3)
        self.assertEqual(data[0].shape, (100, 200, 1))
        self.assertEqual(data[1].shape, (100, 200, 1))
        self.assertEqual(data[2].shape, (100, 200))


if __name__ == "__main__":
    unittest.main()
