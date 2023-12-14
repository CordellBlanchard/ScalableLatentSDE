# Scalable Neural Latent Stochastic Differential Equations

## How to run
```
pip install -r requirements.txt
python src/main.py "configs/dmm/linear_synthetic.yaml"
```
### To use WandB for logging
```
wandb login
```
This will prompt you to enter you API key; you can find it on your account on wandb. Note: this only needs to be done once for each machine. To use the wandb logger, make sure the config has "WandbLogger" as the logger name.

## Extending
### Models
All models should be placed in the src/models directory, and they should be imported in src/models/__init__.py
To create a new model, you need to have the same interface as StateSpaceModel in src/models/base.py, extending that class is the recommended approach to adding a new model (see src/models/dmm.py for examples). Your model can have any arguments you need, you can then set these values in the config file's model_params. You can use the model by setting model_name in the config file to be the same as your class's name, be sure to include all arguments needed by your model in model_params in the config file.

### Losses
All losses should be placed in the src/losses directory, and they should be imported in src/losses/__init__.py
To create a new loss, you need to have need to create class with the following interface:
```
class newLoss(nn.Module):
    def __init__(self, your_params):
        super().__init__()
        # initialize your module
    
    def forward(
        self,
        latent_distribution: Tuple[torch.Tensor, torch.Tensor],
        emission_distribution: Tuple[torch.Tensor, torch.Tensor],
        transition_distribution: Tuple[torch.Tensor, torch.Tensor],
        observation_gt: torch.Tensor,
        epoch: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Forward pass for the loss

        Parameters
        ----------
        latent_distribution : Tuple[torch.Tensor, torch.Tensor]
            Mean and standard deviation of the latent distribution, q(z_t | z_{t-1}, x_{t:T})
            shape = (batch_size, time_steps, latent_dim)
        emission_distribution : Tuple[torch.Tensor, torch.Tensor]
            Mean and standard deviation of the emission distribution, p(x_t | z_t)
            shape = (batch_size, time_steps, obs_dim)
        transition_distribution : Tuple[torch.Tensor, torch.Tensor]
            Mean and standard deviation of the transition distribution, p(z_t | z_{t-1})
            shape = (batch_size, time_steps, latent_dim)
        observation_gt : torch.Tensor
            Ground truth observations, x_{1:T}
            shape = (batch_size, time_steps, obs_dim)
        epoch : int, by default 0
            Which epoch this loss is in, needed for annealing

        Returns
        -------
        torch.Tensor
            float value for the loss
        Dict[str, float]
            Dictionary of losses for each component of the ELBO
            Used for logging
        """
        # implementation here
    def rolling_window_eval(
        self,
        model: nn.Module,
        data: torch.Tensor,
        eval_window_shifts: List[int],
        n_eval_windows: int,
    ) -> Dict[int, List[float]]:
        """
        Rolling window evaluation of RMSE

        Parameters
        ----------
        model : nn.Module
            Model to use for prediction
        data : torch.Tensor
            Data to use for evaluation, observations, shape = (batch_size, time_steps, obs_dim)
        eval_window_shifts : List[int]
            List of shifts to use for evaluation
        n_eval_windows : int
            Number of times to evaluate (each time shifts by 1)

        Returns
        -------
        Dict[int, List[float]]
            Dictionary of RMSE for each shift
        """
        # implementation here, see src/losses/dmm_elbo.py for an example
```
The dictionary can have any float values needed to be logged during training, validation, and testing. Please ensure that there are not nested dictionaries, it should be a simple dictionary of keys and float values.

### Datasets
To create a new dataset, create a directory in the src/datasets folder. In that directory, you need to create a function that returns a dictionary with the following structure:
```
{
    "train": train_loader,
    "val": val_loader,
    "test": test_loader
}
```
The function can take any arguments you define, be sure to set these arguments in the config file under dataset_params. Also be sure to set dataset_name to the same name as your function. Finally, add an import for the function you created in src/datasets/__init__.py

### Loggers
All loggers should be added to the src/logger.py file. The logger_name should be set as the new logger's class name. Any parameters needed by the logger can be set in logger_params, which you can access in the init of the logger through config["logger_params"] as a dictionary. The class should implement the same interface as implemented by the PrintLogger in src/logger.py

### Trainers
All trainers should be added to the src/trainers directory and imported in src/trainers/__init__.py. The class should have the following initialization signature:
```
def __init__(
    self,
    model: nn.Module,
    dataloaders: Dict[str, DataLoader],
    loss: nn.Module,
    logger: Any,
    trainer_params: Dict[str, Any],
):
"""
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
    Should contain lr, n_epochs, val_freq, device, save_path, verbose
"""
```
Additionally, it should implement a method called run(). This is called only once.
