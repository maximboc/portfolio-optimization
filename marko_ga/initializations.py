import numpy as np

def uniform_initialization(num_solutions, num_genes):
    return np.random.rand(num_solutions, num_genes)

def dirichlet_initialization(num_solutions, num_genes, alpha=1.0):
    return np.random.dirichlet(np.ones(num_genes) * alpha, size=num_solutions)

initialization_functions = {
    "dirichelt": dirichlet_initialization,
    "uniform": uniform_initialization
}
