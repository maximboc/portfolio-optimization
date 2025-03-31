import numpy as np

def mutate_directional(direction):
    def directional(offspring, ga_instance):
        mutated_offspring = offspring.copy()
        for idx, individual in enumerate(mutated_offspring):
            mutation_point = np.random.randint(0, len(individual))
            direction = np.random.choice([-1, 1]) 
            mutated_offspring[idx, mutation_point] += direction
            mutated_offspring[idx, mutation_point] = max(0, min(1, mutated_offspring[idx, mutation_point]))
        return mutated_offspring
    return directional

def mutate_gaussian(sigma):
    def gaussian_(offspring, ga_instance):
        mutated_offspring = offspring.copy()
        for idx, individual in enumerate(mutated_offspring):
                mutation_point = np.random.randint(0, len(individual))
                mutated_offspring[idx, mutation_point] += np.random.normal(0, sigma)
                mutated_offspring[idx, mutation_point] = max(0, min(1, mutated_offspring[idx, mutation_point]))
        return mutated_offspring
    return gaussian_

mutation_operators = {
    "directional": mutate_directional,
    "gaussian": mutate_gaussian,
}
