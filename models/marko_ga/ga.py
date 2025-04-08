from pygad import GA
import numpy as np
import hydra
import os
import sys
from datetime import datetime
from typing import List
from dataclasses import dataclass
import logging
import time
import matplotlib.pyplot as plt
from pprint import pprint

from utils.markovitz_utils import *
from utils.finance_utils import get_adj_close_from_stocks, get_risk_free_rate

from models.marko_ga.logger import logger
from models.marko_ga.selections import selection_operators
from models.marko_ga.mutations import mutation_operators
from models.marko_ga.objectives import objective_functions
from models.marko_ga.crossovers import crossover_operators
from models.marko_ga.initializations import initialization_functions
from models.marko_ga.callbacks import on_parents_callback, combined_callback, early_stopping
from models.marko_ga.custom_ga import MyCustomGA, MyCustomGAPlot, MyCustomGABench

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
matplt = plt

@dataclass
class DataConfig:
    start_date: str
    end_date: str
    stocks: List[str]

@dataclass
class Config:
    data: DataConfig

@hydra.main(config_path="config", version_base="1.2")
def main(cfg: Config):
    adj_close_df = get_adj_close_from_stocks(cfg.data.stocks, cfg.data.start_date, cfg.data.end_date)
    logger.info(adj_close_df)
    if adj_close_df.empty:
        logger.error(
            f"ERROR : Stocks {cfg.data.stocks} not found in given range \n \
                with start date : {cfg.data.start_date} \n and End date : {cfg.data.end_date} "
        )
        logger.info("Solution: []")

    log_returns = np.log(adj_close_df / adj_close_df.shift(1))
    log_returns = log_returns.dropna()
    cov_matrix = log_returns.cov() * 250.8875

    if "king" in cfg and cfg.kind == "plot":
        ga_model = MyCustomGAPlot(
            num_generations=cfg.ga.num_generations,
            fitness_func=objective_functions[cfg.objective],
            initial_population=initialization_functions[cfg.init](cfg.ga.initial_population, len(cfg.data.stocks)),
            num_genes=len(cfg.data.stocks),
            random_seed=cfg.seed,

            # Selection
            parent_selection_type=selection_operators[cfg.selection],
            num_parents_mating=cfg.ga.num_parents_mating,
            # Crossover
            crossover_type=crossover_operators[cfg.crossover],
            crossover_probability=cfg.ga.crossover_probability,
            # Mutation
            mutation_type=mutation_operators[cfg.mutation],
            mutation_probability=cfg.ga.mutation_probability,
            # Callbacks
            on_generation=combined_callback,
            on_parents=on_parents_callback,
            # Other
            cov_matrix=cov_matrix,
            log_returns=log_returns,
            risk_free_rate=get_risk_free_rate(),
            parallel_processing=5,
        )
    elif "kind" in cfg and cfg.kind == "benchmark":
        ga_model = MyCustomGABench(
            num_generations=cfg.ga.num_generations,
            fitness_func=objective_functions[cfg.objective],
            initial_population=initialization_functions[cfg.init](cfg.ga.initial_population, len(cfg.data.stocks)),
            num_genes=len(cfg.data.stocks),
            random_seed=cfg.seed,

            # Selection
            parent_selection_type=selection_operators[cfg.selection],
            num_parents_mating=cfg.ga.num_parents_mating,
            # Crossover
            crossover_type=crossover_operators[cfg.crossover],
            crossover_probability=cfg.ga.crossover_probability,
            # Mutation
            mutation_type=mutation_operators[cfg.mutation],
            mutation_probability=cfg.ga.mutation_probability,
            # Callbacks
            on_generation=early_stopping(cfg.ga.termination, objective_functions[cfg.objective]),
            on_parents=on_parents_callback,
            # Other
            cov_matrix=cov_matrix,
            log_returns=log_returns,
            risk_free_rate=get_risk_free_rate(),
            parallel_processing=5,
        )
    else:
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
            risk_free_rate=cfg.data.risk_rate,
            cov_matrix=cov_matrix,
            log_returns=log_returns,
            on_parents=on_parents_callback,
            parallel_processing=5,
            on_generation=early_stopping(cfg.ga.termination, objective_functions[cfg.objective]),
        )
    ga_model.run()

    if "kind" in cfg and cfg.kind == "plot":
        plots_dir = os.path.join(os.getcwd(), "models", "marko_ga", "plots", "animation")
        os.makedirs(plots_dir, exist_ok=True)
        logger.info(f"Generation plots will be saved to: {plots_dir}")

        for gen_idx in range(len(ga_model.weights_history)):
            gen_plot_path = os.path.join(plots_dir, f"generation_{gen_idx:04d}.png")
            ga_model.plot_generation_weights(gen_idx, save_path=gen_plot_path)
            if gen_idx % 10 == 0:
                logger.info(f"Generated plot for generation {gen_idx}")
    elif "kind" in cfg and cfg.kind == "benchmark":
        pprint(ga_model.get_benchmark_metrics())
        ga_model.plot_fitness_history(cfg=cfg, save_path=os.path.join(os.getcwd(), "models", "marko_ga", "plots", "fitness_history.png"))

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
        logger.info(best_solution)
        return {
            "weights": best_solution[0],
            "fitness": ga_model.best_solutions_fitness[-1],
            "best_fitness": best_solution[1],
            "time": end_time - start_time,
        }
    return _ga

if __name__ == "__main__":
    main()
