from pygad import GA
import numpy as np
import hydra
from omegaconf import OmegaConf
import os
import sys

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
    def initialize_population(self):
        self.population = np.random.normal(loc=5, scale=2, size=(self.sol_per_pop, self.num_genes))

@hydra.main(config_path="config", config_name="simple.yml")
def main(cfg: OmegaConf):

    ga_model = GA(
        num_generations=cfg.ga.num_generations,
        fitness_func=objective_functions[cfg.objective],

        parent_selection_type=selection_operators[cfg.selection],
        num_parents_mating=cfg.ga.num_parents_mating,

        crossover_type=crossover_operators[cfg.crossover],
        crossover_probability=cfg.ga.crossover_probability,

        mutation_type=mutation_operators[cfg.mutation],
        mutation_probability=cfg.ga.mutation_probability,

        initial_population=initialization_functions[cfg.init](cfg.ga.initial_population, cfg.ga.num_genes),
        num_genes=cfg.ga.num_genes,
        random_seed=cfg.seed
    )
    best_solution, best_fitness = ga_model.solve()
    ga_model.plot()

    print(f"Best Solution: {best_solution}")
    print(f"Best Fitness: {best_fitness}")

if __name__ == "__main__":
    main()

