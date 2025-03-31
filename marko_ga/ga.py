from pygad import GA
import numpy as np
import hydra
from omegaconf import OmegaConf

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



@hydra.main(config_path="config", config_name="config")
def main(cfg: OmegaConf):

    ga_model = pygad.GA(
        fitness_func=objective_functions[cfg.objective],
        selection_operator=selection_operators[cfg.selection],
        mutation_operator=mutation_operators[cfg.mutation],
        initialization_func=initialization_functions[cfg.init],
    )
    best_solution, best_fitness = ga_model.solve()
    ga_model.plot()

    print(f"Best Solution: {best_solution}")
    print(f"Best Fitness: {best_fitness}")

if __name__ == "__main__":
    main()

