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
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils.markovitz_utils import *
from utils.finance_utils import get_adj_close_from_stocks, get_maximum_risk, get_risk_free_rate

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

matplt = plt
@dataclass
class Config:
    data: DataConfig


def save_weights_callback(ga_instance):
    """Save the weights of the current population"""
    # Make a deep copy of the current population
    current_weights = np.copy(ga_instance.population)
    ga_instance.weights_history.append(current_weights)
    
    # You can also log some statistics if desired
    if ga_instance.generations_completed % 10 == 0:
        logger.info(f"Generation {ga_instance.generations_completed}: Saved population weights")
    
    # Chain with the early stopping callback
    return ""

class MyCustomGA_plot(GA):
    def __init__(self, fitness_func, cov_matrix, log_returns, risk_free_rate,  **kwargs):
        super().__init__(fitness_func=fitness_func, **kwargs)
        self.log_returns = log_returns
        self.cov_matrix = cov_matrix
        self.risk_free_rate = risk_free_rate
        self.weights_history = []

    def set_log_returns(self, log_returns):
        self.log_returns = log_returns

    def set_cov_matrix(self, cov_matrix):
        self.cov_matrix = cov_matrix

    def plot_generation_weights(self, gen_idx, save_path=None):
        """Plot the weights distribution for all individuals in a specific generation"""
        if not self.weights_history or gen_idx >= len(self.weights_history):
            logger.warning(f"Generation {gen_idx} data not available")
            return
            
        generation_weights = self.weights_history[gen_idx]
        num_individuals = len(generation_weights)
        num_genes = self.num_genes
        
        if num_genes == 2:
            # For 2-asset portfolios, create a single 2D scatter plot
            plt.figure(figsize=(10, 8))
            
            # Extract weights for the two assets
            asset1_weights = [individual[0] for individual in generation_weights]
            asset2_weights = [individual[1] for individual in generation_weights]
            
            # Create a scatter plot of all individuals
            plt.scatter(asset1_weights, asset2_weights, alpha=0.7, s=60)
            
            # Adding a line showing the constraint that weights sum to 1
            # (Optional - only if your weights are constrained to sum to 1)
            x = np.linspace(0, 1, 100)
            y = 1 - x
            plt.plot(x, y, 'r--', alpha=0.5, label='Weight Sum = 1')
            
            # Add annotations for some individuals (e.g., the best one)
            if hasattr(self, 'last_generation_fitness') and len(self.last_generation_fitness) > 0:
                best_idx = np.argmax(self.last_generation_fitness)
                best_x, best_y = asset1_weights[best_idx], asset2_weights[best_idx]
                plt.scatter([best_x], [best_y], color='red', s=100, label='Best Individual')
                plt.annotate(f'Best ({best_x:.2f}, {best_y:.2f})', 
                            (best_x, best_y), 
                            xytext=(10, 10), 
                            textcoords='offset points')
            
            plt.title(f'Generation {gen_idx} - Portfolio Weight Distribution')
            plt.xlabel('Asset 1 Weight')
            plt.ylabel('Asset 2 Weight')
            plt.grid(True)
            plt.legend()
            
            # Add a colorbar with fitness values if available
            if hasattr(self, 'last_generation_fitness') and len(self.last_generation_fitness) > 0:
                sc = plt.scatter(asset1_weights, asset2_weights, c=self.last_generation_fitness, 
                                cmap='viridis', alpha=0.7, s=60)
                plt.colorbar(sc, label='Fitness')
            
            plt.scatter([1] ,[0], color='purple', s=100, label='Optimal Weights')
            if save_path:
                plt.savefig(save_path)
            plt.close()
        else:
            # Implementation for more than 2 assets - simple histogram with best individual weights
            plt.figure(figsize=(15, 8))
            plt.title(f'Generation {gen_idx} - Best Individual Weights')
            
            # Find the best individual
            best_idx = 0
            if hasattr(self, 'last_generation_fitness') and len(self.last_generation_fitness) > 0:
                best_idx = np.argmax(self.last_generation_fitness)
            
            # Extract best weights
            best_weights = generation_weights[best_idx]
            
            # Create labels for x-axis
            #assets = [f'Asset {i+1}' for i in range(num_genes)]
            assets = ["META", "IBM", "AAPL", "MSFT","AMZN", "NVDA", "GOOGL", "TSLA", "NFLX", "BABA"]
            # Create bar chart (histogram) of best weights
            bars = plt.bar(assets, best_weights, color=[f'C{i}' for i in range(num_genes)])
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.4f}', ha='center', va='bottom')
            
            plt.ylim(0, 1.0)  # Fixed y-axis from 0 to 1
            plt.ylabel('Weight Value')
            plt.grid(True, alpha=0.3, axis='y')
            
            # Add text with fitness if available
            if hasattr(self, 'last_generation_fitness') and len(self.last_generation_fitness) > 0:
                best_fitness = self.last_generation_fitness[best_idx]
                plt.figtext(0.5, 0.01, f'Fitness: {best_fitness:.6f}', 
                           ha='center', bbox={'facecolor': 'lightgray', 'alpha': 0.5, 'pad': 5})
            
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Make room for the text at the bottom
            if save_path:
                plt.savefig(save_path)
            plt.close()
    
@hydra.main(config_path="config", config_name="test_for_plot.yaml", version_base="1.2")
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

    def combined_callback(ga_instance):
        save_weights_callback(ga_instance)
        return early_stopping(cfg.ga.termination)(ga_instance)

    ga_model = MyCustomGA_plot(
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
        #random_seed=cfg.seed,
        cov_matrix=cov_matrix,
        log_returns=log_returns,
        risk_free_rate=get_risk_free_rate(),
        on_parents=on_parents_callback,
        save_best_solutions=True,
        save_solutions=True,
        on_generation=combined_callback,
    )
    ga_model.run()
    best_solution = ga_model.best_solution()

    # Create a directory for storing generation plots
    plots_dir = os.path.join(os.getcwd(), "generation_plots")
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Generation plots will be saved to: {plots_dir}")

    # Plot weights for each generation
    for gen_idx in range(len(ga_model.weights_history)):
        gen_plot_path = os.path.join(plots_dir, f"generation_{gen_idx:04d}.png")
        ga_model.plot_generation_weights(gen_idx, save_path=gen_plot_path)
        if gen_idx % 10 == 0:
            print(f"Generated plot for generation {gen_idx}")


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
