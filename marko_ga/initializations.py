import numpy as np

def uniform_initialization(num_solutions, num_genes):
    pop = []
    for _ in range(num_solutions):
        pop.append(np.random.uniform(1/np.ones(num_genes)))
    return np.array(pop)

def dirichlet_initialization(num_solutions, num_genes):
    pop = []
    for _ in range(num_solutions):
        pop.append(np.random.dirichlet(np.ones(num_genes)))
    return np.array(pop)

initialization_functions = {
    "dirichelt": dirichlet_initialization,
    "uniform": uniform_initialization
}
