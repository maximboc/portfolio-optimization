import numpy as np
from models.marko_ga.logger import logger
from utils.markovitz_utils import expected_return, standard_deviation

def save_weights_callback(ga_instance):
    """Save the weights of the current population"""
    current_weights = np.copy(ga_instance.population)
    ga_instance.weights_history.append(current_weights)
    
    if ga_instance.generations_completed % 10 == 0:
        logger.info(f"Generation {ga_instance.generations_completed}: Saved population weights")
    
    return ""

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

def early_stopping(termination, objective_function):
    """Early stopping callback for the genetic algorithm"""
    def _early_stopping(ga_instance) -> str:
        if termination.minimal_generations > ga_instance.generations_completed:
            return ""
        
        if len(ga_instance.best_solutions_fitness) > 0:
            best_weights = ga_instance.best_solution()[0]

            current_best_fitness = objective_function(ga_instance, best_weights, None)

            if ga_instance.best_fitness is None:
                ga_instance.best_fitness = -float("inf")

            if np.abs(current_best_fitness - ga_instance.best_fitness) < termination.epsilon:
                logger.warning(f"Early stopping at generation {ga_instance.generations_completed}")
                return "stop"

            ga_instance.best_fitness = max(ga_instance.best_fitness, current_best_fitness)

        return ""
    
    return _early_stopping

def combined_callback(ga_instance, termination):
    save_weights_callback(ga_instance)
    return early_stopping(termination)(ga_instance)
