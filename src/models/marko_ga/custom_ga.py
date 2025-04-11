from pygad import GA
import numpy as np
import matplotlib.pyplot as plt
from models.marko_ga.logger import logger
import time

class MyCustomGAPlot(GA):
    def __init__(self, fitness_func, cov_matrix, log_returns, risk_free_rate,  **kwargs):
        super().__init__(fitness_func=fitness_func, **kwargs)
        self.log_returns = log_returns
        self.cov_matrix = cov_matrix
        self.risk_free_rate = risk_free_rate
        self.best_fitness = -np.inf
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
            assets = ["Asset " + str(i+1) for i in range(num_genes)]
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
            
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            if save_path:
                plt.savefig(save_path)
            plt.close()


class MyCustomGABench(GA):
    def __init__(self, fitness_func, cov_matrix, log_returns, risk_free_rate, **kwargs):
        super().__init__(fitness_func=fitness_func, **kwargs)
        self.log_returns = log_returns
        self.cov_matrix = cov_matrix
        self.risk_free_rate = risk_free_rate
        self.weights_history = []
        self.last_generation_fitness = []
        self.best_fitness_history = []
        self._start_time = None
        self.best_fitness = -np.inf
        self._end_time = None

    def set_log_returns(self, log_returns):
        self.log_returns = log_returns

    def set_cov_matrix(self, cov_matrix):
        self.cov_matrix = cov_matrix

    def run(self):
        self._start_time = time.time()
        super().run()
        self._end_time = time.time()

    def get_benchmark_metrics(self):
        return {
            "time": self._end_time - self._start_time if self._end_time and self._start_time else None,
            "generations_run": len(self.best_solutions_fitness),
            "final_fitness": self.best_solutions_fitness[-1] if self.best_solutions_fitness else None,
            "best_solution": self.best_solution(),
        }

    def plot_fitness_history(self, cfg, save_path=None):
        if not self.best_solutions_fitness:
            logger.warning("No fitness history to plot.")
            return

        max_generations = self.num_generations if self.num_generations else len(self.best_solutions_fitness)
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.best_solutions_fitness, marker="o", label="Best Fitness")
        plt.title(f"Population: {cfg.ga.initial_population}, Crossover: {cfg.crossover}, Mutation: {cfg.mutation}, Init: {cfg.init}, Selection: {cfg.selection}")
        plt.xlabel("Generation")
        plt.ylabel("Best Fitness")
        plt.xlim(0, max_generations)
        plt.grid(True)

        if len(self.best_solutions_fitness) < max_generations:
            stopping_point = len(self.best_solutions_fitness)
            plt.axvline(stopping_point, color='red', linestyle='--', label='Early Stopping')
        plt.legend()
        if save_path:
            plt.savefig(save_path)
        plt.close()

class MyCustomGA(GA):
    def __init__(self, fitness_func, cov_matrix, log_returns, risk_free_rate,  **kwargs):
        super().__init__(fitness_func=fitness_func, **kwargs)
        self.log_returns = log_returns
        self.cov_matrix = cov_matrix
        self.risk_free_rate = risk_free_rate
        self.best_fitness = -np.inf

    def set_log_returns(self, log_returns):
        self.log_returns = log_returns

    def set_cov_matrix(self, cov_matrix):
        self.cov_matrix = cov_matrix
