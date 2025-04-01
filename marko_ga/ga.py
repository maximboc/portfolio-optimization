from pygad import GA
import numpy as np
import hydra
from omegaconf import OmegaConf
import os
import sys
from datetime import datetime
from typing import List
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GA_Portfolio")

def on_parents_callback(ga_instance, selected_parents):
    """Log information about selected parents"""
    logger.info(f"Generation {ga_instance.generations_completed}: Selected {len(selected_parents)} parents")
    
    # Get fitness of parents
    parents_fitness = []
    for parent_idx in range(len(selected_parents)):
        parent_fitness = ga_instance.last_generation_fitness[ga_instance.last_generation_parents_indices[parent_idx]]
        parents_fitness.append(parent_fitness)
    
    logger.info(f"Parents fitness: min={min(parents_fitness):.6f}, max={max(parents_fitness):.6f}, avg={np.mean(parents_fitness):.6f}")
    logger.info(f"Parent indices: {ga_instance.last_generation_parents_indices}")
    
    # Optionally log more details about best parent
    best_parent_idx = np.argmax(parents_fitness)
    best_parent = selected_parents[best_parent_idx]
    logger.info(f"Best parent weights: {best_parent}")
    
    # Calculate portfolio metrics for best parent
    weights = best_parent
    portfolio_return = expected_return(weights, ga_instance.log_returns)
    portfolio_risk = standard_deviation(weights, ga_instance.cov_matrix)
    logger.info(f"Best parent - Return: {portfolio_return:.4f}, Risk: {portfolio_risk:.4f}")

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils.markovitz_utils import *
from utils.finance_utils import get_adj_close_from_stocks, get_risk_free_rate, get_maximum_risk

from marko_ga.selections import selection_operators
from marko_ga.mutations import mutation_operators
from marko_ga.objectives import objective_functions
from marko_ga.crossovers import crossover_operators
from marko_ga.initializations import initialization_functions

## Model
# Genome: [weights] such that sum weights = 1
# Init:
##   - Uniform (1/n)
##   - numpy.random.dirichlet
# selection:
##   - Best (given objective function)
##   - Top K
##   - Threshold
# crossover:
##   - Arithmetic Crossover: Weighted sum of 2 parents
##   - Convex combination: generalization with alea
# mutation:
##   - Directional Mutation
##   - Gaussian mutation with normalization

class MyCustomGA(GA):
    def __init__(self, fitness_func, cov_matrix, log_returns, **kwargs):
        super().__init__(fitness_func=fitness_func, **kwargs)
        self.log_returns = log_returns
        self.cov_matrix = cov_matrix
        self.risk_free_rate = get_risk_free_rate()

    def set_log_returns(self, log_returns):
        self.log_returns = log_returns

    def set_cov_matrix(self, cov_matrix):
        self.cov_matrix = cov_matrix

@dataclass
class DataConfig:
    start_date: str
    end_date: str
    stocks: List[str]

@dataclass
class Config:
    data: DataConfig

@hydra.main(config_path="config", config_name="simple.yaml", version_base="1.2")
def main(cfg: Config):
    #print(OmegaConf.to_yaml(cfg))
    adj_close_df = get_adj_close_from_stocks(cfg.data.stocks, cfg.data.start_date, cfg.data.end_date)
    print(adj_close_df)
    if adj_close_df.empty:
        print(
            f"ERROR : Stocks {cfg.data.stocks} not found in given range \n \
                with start date : {cfg.data.start_date} \n and End date : {cfg.data.end_date} "
        )
        print("Solution: []")

    log_returns = np.log(adj_close_df / adj_close_df.shift(1))
    log_returns = log_returns.dropna()
    cov_matrix = log_returns.cov() * 250.8875

    ga_model = MyCustomGA(
        num_generations=cfg.ga.num_generations,
        fitness_func=objective_functions[cfg.objective],

        parent_selection_type=selection_operators[cfg.selection],
        num_parents_mating=cfg.ga.num_parents_mating,

        crossover_type=crossover_operators[cfg.crossover],
        crossover_probability=cfg.ga.crossover_probability,

        mutation_type=mutation_operators[cfg.mutation],
        mutation_probability=cfg.ga.mutation_probability,

        initial_population=initialization_functions[cfg.init](cfg.ga.initial_population, len(cfg.data.stocks)),
        num_genes=len(cfg.data.stocks),
        random_seed=cfg.seed,
        cov_matrix=cov_matrix,
        log_returns=log_returns,
        on_parents=on_parents_callback,
    )
    ga_model.run()
    best_solution = ga_model.best_solution()
    ga_model.plot_genes()
    print(f"Best Solution: {best_solution}")

if __name__ == "__main__":
    main()

