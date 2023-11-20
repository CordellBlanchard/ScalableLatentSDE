import pandas as pd
import numpy as np


class AutoRegressionIntegrated:
  """
  AutoRegressive Integrated (ARI) model for one dimensional data.

  Parameters:
  - diff_order (np.array): array containing the diff order of each variable from training data
  - coef (float): coefficient found from fitting model to training data

  Methods:
  - predict_future(test_data, steps): Makes predictions using the fitted VAR model.
  - inverse_difference(predictions): Transforms predicted data back to its original form if diff order is greater than 0.
  """

  def __init__(self, lag=2):
    """
    Initialize the AutoRegressionIntegrated object.

    Parameters:
    - lag (int): The lag order to use. Default is 2.
    """

    self.lag = lag

  def _inverse_difference(self, predictions, history, diff_order):
      """
      Transforms predicted data back to its original form if diff order is greater than 0.

      Parameters:
      - predictions (pd.DataFrame): The predicted values.
      - history (pd.Dataframe): the data that predictions were predicting before differencing
      - diff_order (np.array): numpy array containing the diff order of each variable from training data


      Returns:
      - inverted_predictions (pd.DataFrame): The transformed predictions.
      """
      inverted_predictions = predictions.copy()

      for row in range(predictions.shape[0]):
          for order in diff_order: # for multidimnesions
            if order > 0:
                last_non_missing = history[row, -1]
                # Apply cumulative sum for each difference order
                for _ in range(order):
                    inverted_predictions[row, :] = inverted_predictions[row, :].cumsum() + last_non_missing

      return inverted_predictions


  def predict_future(self, test_data, steps, diff_order, coef):
    """
    Make predictions using the fitted VAR model.

    Parameters:
    - test_data (pd.DataFrame): The test data used for making predictions (needs to have same indices as training data).
    - steps (int): The number of steps to forecast.
    - diff_order (np.array): numpy array containing the diff order of each variable from training data
    - coef (float): coefficient found from fitting model to training data


    Returns:
    - inverse_predictions_df (pd.DataFrame): The predicted values.
    """

    # difference data to same order as training data:
    differenced_test_data = test_data.copy()
    for row in range(differenced_test_data.shape[0]):
        for order in diff_order:
            for _ in range(order):
                differenced_test_data[row, :] = pd.Series(differenced_test_data[row, :]).diff().bfill().values
    
    differenced_dataset = differenced_test_data.tolist()

    predictions = []
    # predict t amount of steps
    for t in range(steps):
        predictions_for_step = []
        for index in range(differenced_test_data.shape[0]):
            sample = differenced_dataset[index][:]
            length = len(sample)
            lag_vals = sample[length - self.lag:length]
            prediction = coef.dot(lag_vals)  # Predict using dot product
            predictions_for_step.append(prediction)
            differenced_dataset[index][:].append(prediction)  # Append prediction to history for next step

        predictions.append(predictions_for_step)
    
    predictions_array = np.array(predictions).T # transpose to get it in the right format

    inverse_predictions = self._inverse_difference(predictions_array, test_data, diff_order)

    return inverse_predictions

