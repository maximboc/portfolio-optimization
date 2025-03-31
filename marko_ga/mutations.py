import numpy as np

def mutatebitflip(population, pm):
    mutated_population = []
    for individual in population:
        if np.random.rand() < pm:
            mutation_point = np.random.randint(0, len(individual))
            individual[mutation_point] = 1 - individual[mutation_point]  # Bit-flip mutation
        mutated_population.append(individual)
    return np.array(mutated_population)

def mutate_directional(population, pm):
    mutated_population = []
    for individual in population:
        if np.random.rand() < pm:
            mutation_point = np.random.randint(0, len(individual))
            direction = np.random.choice([-1, 1])  # Randomly decide direction
            individual[mutation_point] += direction
            individual[mutation_point] = max(0, min(1, individual[mutation_point]))  # Ensure bounds
        mutated_population.append(individual)
    return np.array(mutated_population)

def mutate_gaussian(population, pm, sigma=0.2):
    mutated_population = []
    for individual in population:
        if np.random.rand() < pm:
            mutation_point = np.random.randint(0, len(individual))
            individual[mutation_point] += np.random.normal(0, sigma)
            individual[mutation_point] = max(0, min(1, individual[mutation_point]))
        mutated_population.append(individual)
    return np.array(mutated_population)

mutation_operators = {
    "bit_flip": mutatebitflip,
    "directional": mutate_directional,
    "gaussian": mutate_gaussian,
}
