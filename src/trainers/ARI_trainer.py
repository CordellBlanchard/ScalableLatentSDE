import pandas as pd
import numpy as np
from tqdm import tqdm
from statsmodels.tsa.stattools import adfuller

class AutoRegressionIntegratedTrainer:
  """
  Trainer for AutoRegressive Integrated (ARI) model.

  Parameters
  ----------
  model: nn.Module
      Model to train
  dataloaders: Dict[str, DataLoader]
      Dictionary of dataloaders for train, val, and test sets
  loss: nn.Module
      Loss function to use
      forward should take as input: model output, data, and epoch
  logger: Logger
      Used to log metrics during training
  trainer_params: Dict[str, Any]
      Dictionary of parameters for the trainer
      Should contain max_diff, significance_level, lag, eval_window_shift, n_eval_windows, diff_order
  """

  def __init__(self, model, dataloaders, loss, logger, trainer_params):
    """
    Initialize the AutoRegressionIntegrated object.

    Parameters:
    - data (pd.DataFrame): The time series data with columns representing different variables.
    - max_diff (int): The maximum number of differences to attempt to make the data stationary.
    - significance_level (float): The significance level for the Augmented Dickey-Fuller test.
    - lag (int): The lag order to use. Default is 2.
    """

    self.model = model # ARI model
    self.dataloaders = dataloaders # data
    self.loss = loss # rmse loss
    self.logger = logger # wandb logger
    self.params = trainer_params # contains max_diff, siginicane level, lag, etc.

    self.coef = None
    self._useVAR = False # if data is more than 1 dimensional use data. (have not added feature that yet)


  def _make_stationary(self, data):
    """
    Takes differences of the dataset until it becomes stationary.
    
    Parameters:
        - data (np.array): training dataset.
    """
    differenced_data = data.copy()
    for index in tqdm(range(data.shape[0]), desc="Making Data Stationary"):
      for order in self.params["diff_order"]: # this will be for multidimensions
          for _ in range(order):
            # Take the difference
            differenced_data[index, :, 0] = pd.Series(differenced_data[index,:, 0]).diff().bfill().values
    
    return differenced_data

  def run(self):
    """
    Fits the VAR or AR model with the selected lag order.

    Returns:
    - diff_order (pd.Series): array containing the diff order of each variable from training data
    - coef (float): coefficient found from fitting model to training data
    """
    all_data = [] # shape is (n_samples, n_time_steps, number_of_dimensions)
    for data in self.dataloaders['train']:
        if isinstance(data, list):
            all_data.append(data[0].numpy())
        else:
            all_data.append(data.numpy())

    
    # check if number_of_dimensions is greater than 1? to use VAR

    # make data stationary
    all_data = np.concatenate(all_data, axis=0)
    differenced_data = self._make_stationary(all_data)
    
    # Prepare data for AR model
    # we will need to create the lag values for AR
    # and the predictions of those lag values
    lag_vals, next_vals = [], []
    for index in range(differenced_data.shape[0]):
        sample = differenced_data[index, :, 0]
        for i in range(self.params["lag"], len(sample)):
            lag_vals.append(sample[i-self.params["lag"]:i])
            next_vals.append(sample[i])

    # use least squares to fit a coeffient to best predict all samples in dimension
    lag_vals, next_vals = np.array(lag_vals), np.array(next_vals)
    self.coef = np.linalg.lstsq(lag_vals, next_vals, rcond=None)[0]
    to_log = self.run_eval(self.dataloaders["test"])
    self.logger.log_test(to_log)

    # save the coefficient
    np.save(self.params["save_path"], self.coef)
  
  def run_eval(self, dataloader):
    """
    Run evaluation loop

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        Dataloader to use for evaluation

    Returns
    -------
    """

    all_data = [] # shape is (n_samples, n_time_steps, number_of_dimensions)
    for data in dataloader:
        if isinstance(data, list):
            all_data.append(data[0].numpy())
        else:
            all_data.append(data.numpy())

    all_data = np.concatenate(all_data, axis=0)

    # Moving window RMSE
    rolling_window_rmse = self.loss.rolling_window_eval(
        self.model,
        all_data,
        self.params["eval_window_shifts"],
        self.params["n_eval_windows"],
        self.params["diff_order"],
        self.coef
    )

    for key in rolling_window_rmse:
        rolling_window_rmse[key] = sum(rolling_window_rmse[key]) / len(
            rolling_window_rmse[key]
        )

    eval_metrics = {}
    eval_metrics["rolling_window"] = rolling_window_rmse
    return eval_metrics
     