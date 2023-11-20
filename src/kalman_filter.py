import argparse
import yaml
from datasets import linear_gssm
from pykalman import KalmanFilter
import numpy as np

parser = argparse.ArgumentParser(description="Kalman Filter")
parser.add_argument("config", type=str, help="Path to the config yaml file")
parser.add_argument("--type", type=str, help="filter or smoother", required=False, default="filter")
parser.add_argument("--num_iters", type=int, help="Number of iterations", required=False, default=20)
args = parser.parse_args()

with open(args.config, "r", encoding="utf-8") as yaml_file:
        config = yaml.load(yaml_file, Loader=yaml.FullLoader)

rmses = []
logps = []

for i in range(args.num_iters):

    dataloaders = eval(config["dataset_name"])(**config["dataset_params"])
    test_loader = dataloaders['test']

    for i, (observations) in enumerate(test_loader):
        observations = np.array(observations)
        # print(observations.shape)

    X = observations.squeeze()
    # print(X.shape) # (512, 25)

    model = KalmanFilter(transition_matrices = np.array([1]),
                        observation_matrices = np.array([0.5]),
                        transition_covariance = np.array([10]),
                        observation_covariance = np.array([20]),
                        transition_offsets = np.array([0.05]),
                        observation_offsets = np.array([0]),
                        initial_state_mean = np.array([0]),
                        initial_state_covariance = np.array([10]))

    ll = []
    mus = np.zeros(X.shape)
    cov = np.zeros(X.shape)

    for i, patient in enumerate(X):
        if args.type == "filter":
            smoothed_state_means, smoothed_state_covariances = model.filter(patient)
        elif args.type == "smoother":
            smoothed_state_means, smoothed_state_covariances = model.smooth(patient)
        mus[i] = smoothed_state_means.ravel()
        cov[i] = smoothed_state_covariances.ravel()
        ll.append(-1*model.loglikelihood(patient))


    rmse = np.sqrt(np.mean((X - mus/2)**2))
    log_p = np.mean(ll)/25
    print("RMSE", rmse)
    print("Log p", log_p)
    rmses.append(rmse)
    logps.append(log_p)


print("RMSES", rmses)
print("RMSE Mean", np.mean(rmses))
print("RMSE Std", np.std(rmses))
print("LOGPS", logps)
print("LOGP Mean", np.mean(logps))
print("LOGP Std", np.std(logps))