import numpy as np
from typing import List, Dict
from sklearn.metrics import mean_squared_error


class AutoRegressionIntegreatedRMSE:
    """
    RMSE for AutoRegressive Integrated (ARI) model.

    Parameters:
    - model (AutoRegressiveIntegrated): The trained ARI model

    Methods:
    - calculate_rmse(true_values, predicted_values): Calculates the Root Mean Squared Error (RMSE) for the predicted values.
    - rolling_window_rmse(eval_window_shifts, n_eval_windows): Rolling window evaluation of RMSE.

    """

    def __init__(self):
        """
        Initialize the AutoRegressionIntegrated object.

        Parameters:
        - model (AutoRegressiveIntegrated): The trained ARI model
        """
        pass
    
    def calculate_rmse(self, true_values, predicted_values):
        """
        Calculates the Root Mean Squared Error (RMSE) for the predicted values.

        Parameters:
        - true_values (np.array): The true values.
        - predicted_values (np.array): The predicted values.

        Returns:
        - rmse (float): The Root Mean Squared Error.
        """
        mse = mean_squared_error(true_values, predicted_values)
        rmse = np.sqrt(mse)
        return rmse

    def rolling_window_eval(self, model: any, data: np.array, eval_window_shifts: List[int], n_eval_windows: int, diff_order: np.array, coef: float) -> Dict[int, List[float]]:
        """
        Rolling window evaluation of RMSE for Pandas-based VAR models.

        Parameters:
        - model (any): ARI model
        - data (np.Array): np.array containing the test dataset
        - eval_window_shifts (List[int]): List of shifts to use for evaluation.
        - n_eval_windows (int): Number of times to evaluate (each time shifts by 1).
        - diff_order (np.array): array containing the diff order of each variable from training data
        - coef (float): coefficient found from fitting model to training data

        Returns:
        - Dict[int, List[float]]: Dictionary of RMSE for each shift.
        """
        max_window_shift = max(eval_window_shifts)
        rolling_window_rmse = {i: [] for i in eval_window_shifts}

        for n in range(n_eval_windows):
            n_observations = data.shape[1] - n - max_window_shift
            if n_observations <= 0:
                print("Warning: during RMSE rolling window evaluation, n_observations <= 0")
                continue
                
            observations = data[:, :n_observations, 0]
            predicted_values = model.predict_future(test_data=observations, steps=max_window_shift, diff_order=diff_order, coef=coef)
            for shift in eval_window_shifts:
                shift_obs = data[:, n_observations + shift - 1, 0]
                shift_preds = predicted_values[:, shift - 1]
                shift_rmse = self.calculate_rmse(shift_obs, shift_preds)
                rolling_window_rmse[shift].append(shift_rmse)

        return rolling_window_rmse