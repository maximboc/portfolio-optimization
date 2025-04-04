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
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils.markovitz_utils import *
from utils.finance_utils import get_adj_close_from_stocks, get_maximum_risk

from marko_ga.selections import selection_operators
from marko_ga.mutations import mutation_operators
from marko_ga.objectives import objective_functions
from marko_ga.crossovers import crossover_operators
from marko_ga.initializations import initialization_functions

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Portfolio/GA")

best_fitness = -np.inf

def on_parents_callback(ga_instance, selected_parents) -> None:
    """Log information about selected parents"""
    logger.info(f"Generation {ga_instance.generations_completed}: Selected {len(selected_parents)} parents")

    parents_fitness = []
    for parent_idx in range(len(selected_parents)):
        parent_fitness = ga_instance.last_generation_fitness[ga_instance.last_generation_parents_indices[parent_idx]]
        parents_fitness.append(parent_fitness)
    
    logger.info(f"Parents fitness: min={min(parents_fitness):.6f}, max={max(parents_fitness):.6f}, avg={np.mean(parents_fitness):.6f}")
    logger.info(f"Parent indices: {ga_instance.last_generation_parents_indices}")

    best_parent_idx = np.argmax(parents_fitness)
    best_parent = selected_parents[best_parent_idx]
    logger.info(f"Best parent weights: {best_parent}")
    logger.info(f"np sum is {np.sum(best_parent)}")

    weights = best_parent
    portfolio_return = expected_return(weights, ga_instance.log_returns)
    portfolio_risk = standard_deviation(weights, ga_instance.cov_matrix)
    logger.info(f"Best parent - Return: {portfolio_return:.4f}, Risk: {portfolio_risk:.4f}")

def early_stopping(termination):
    def _early_stopping(ga_instance) -> str:
        global best_fitness
        if termination.minimal_generations > ga_instance.generations_completed:
            return ""
        
        # Check if best_solutions_fitness is populated before calling best_solution()
        if len(ga_instance.best_solutions_fitness) > 0:
            current_best_fitness = ga_instance.best_solutions_fitness[-1]
            
            if np.abs(current_best_fitness - best_fitness) < termination.epsilon:
                logger.warning(f"Early stopping at generation {ga_instance.generations_completed}")
                return "stop"
                
            best_fitness = max(best_fitness, current_best_fitness)
        
        return ""
    return _early_stopping


class MyCustomGA(GA):
    def __init__(self, fitness_func, cov_matrix, log_returns, risk_free_rate,  **kwargs):
        super().__init__(fitness_func=fitness_func, **kwargs)
        self.log_returns = log_returns
        self.cov_matrix = cov_matrix
        self.risk_free_rate = risk_free_rate

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
        risk_free_rate=cfg.data.risk_free_rate,
        on_parents=on_parents_callback,
        save_best_solutions=True,

        on_generation=early_stopping(cfg.ga.termination),
    )
    ga_model.run()
    best_solution = ga_model.best_solution()
    ga_model.plot_genes()
    print(f"Best Solution: {best_solution}")


def ga(config: str) -> dict:
    cfg = {}
    with hydra.initialize(config_path="config"):
        cfg = hydra.compose(config_name=config, overrides=[])

    def _ga(    stocks: list,
        start_date: datetime,
        end_date: datetime,
        bounds : list,
        risk_free_rate : int):
    
        adj_close_df = get_adj_close_from_stocks(stocks, start_date, end_date)
        log_returns = np.log(adj_close_df / adj_close_df.shift(1))
        log_returns = log_returns.dropna()
        cov_matrix = log_returns.cov() * 250.8875
        print("num_genes:" + str(len(stocks)))
        ga_model = MyCustomGA(
            num_generations=cfg.ga.num_generations,
            fitness_func=objective_functions[cfg.objective],

            parent_selection_type=selection_operators[cfg.selection],
            num_parents_mating=cfg.ga.num_parents_mating,

            crossover_type=crossover_operators[cfg.crossover],
            crossover_probability=cfg.ga.crossover_probability,

            mutation_type=mutation_operators[cfg.mutation],
            mutation_probability=cfg.ga.mutation_probability,

            initial_population=initialization_functions[cfg.init](cfg.ga.initial_population, len(stocks)),
            num_genes=len(stocks),
            random_seed=cfg.seed,
            risk_free_rate=risk_free_rate,
            cov_matrix=cov_matrix,
            log_returns=log_returns,
            on_parents=on_parents_callback,
            save_best_solutions=True,
            parallel_processing=5,
            on_generation=early_stopping(cfg.ga.termination),
        )
        start_time = time.time()        
        ga_model.run()
        end_time = time.time()
        
        best_solution = ga_model.best_solution()
        print(best_solution)
        return {
            "weights": best_solution[0],
            "fitness": ga_model.best_solutions_fitness[-1],
            "best_fitness": best_solution[1],
            "time": end_time - start_time,
        }
    return _ga
    

if __name__ == "__main__":

    main()
