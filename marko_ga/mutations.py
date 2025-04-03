import numpy as np

def mutate_directional(alpha = 0.01):
    def _directional(offspring, ga_instance):
        for i in range(len(offspring)):
            direction = np.random.randn(*offspring[i].shape)
            direction -= np.mean(direction)
            offspring[i] += alpha * direction
            offspring[i] /= np.sum(offspring[i])

        return offspring
    return _directional

def mutate_gaussian(sigma):
    def _gaussian(offspring, ga_instance):
        mutated_offspring = offspring.copy()
        for idx, individual in enumerate(mutated_offspring):
            mutation_point = np.random.randint(0, len(individual))
            other_point = np.random.choice([i for i in range(len(individual)) if i != mutation_point])
            
            variation = np.random.normal(0, sigma)
            mutated_offspring[idx, mutation_point] += variation
            mutated_offspring[idx, other_point] -= variation

            mutated_offspring[idx, mutation_point] = max(0, min(1, mutated_offspring[idx, mutation_point]))
            mutated_offspring[idx, other_point] = max(0, min(1, mutated_offspring[idx, other_point]))
            
        return mutated_offspring
    
    return _gaussian

mutation_operators = {
    "directional": mutate_directional(),
    "directional_0.1": mutate_directional(0.1),
    "gaussian": mutate_gaussian(0.01),
    "gaussian_0.5": mutate_gaussian(0.05),
    "gaussian_1": mutate_gaussian(0.1),
}
