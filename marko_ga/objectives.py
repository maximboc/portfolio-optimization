# objective_functions.py
from utils.markovitz_utils import negative_sharpe_ratio

objective_functions = {
    "sharpe": negative_sharpe_ratio,
}
