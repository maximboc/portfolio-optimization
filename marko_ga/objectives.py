# objective_functions.py
from utils.markovitz_utils import sharpe_ratio
import numpy as np
def fitness_func(ga_instance, solution, solution_idx):
    return sharpe_ratio(solution, ga_instance.log_returns, ga_instance.cov_matrix, ga_instance.risk_free_rate)  - 1000000 * np.abs(np.sum(solution) - 1)

objective_functions = {
    "sharpe": fitness_func,
}
