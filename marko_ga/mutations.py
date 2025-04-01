import numpy as np

def mutate_directional():
    def directional(offspring, ga_instance):
        mutated_offspring = offspring.copy()
        for idx, individual in enumerate(mutated_offspring):
            mutation_point = np.random.randint(0, len(individual))
            other_point = np.random.choice([i for i in range(len(individual)) if i != mutation_point])
            variation = np.random.uniform(0, max(mutated_offspring[idx, mutation_point], mutated_offspring[idx, other_point]))
            
            mutated_offspring[idx, mutation_point] += variation
            mutated_offspring[idx, other_point] -= variation

            mutated_offspring[idx, mutation_point] = max(0, min(1, mutated_offspring[idx, mutation_point]))
            mutated_offspring[idx, other_point] = max(0, min(1, mutated_offspring[idx, other_point]))
        
        return mutated_offspring
    
    return directional

def mutate_gaussian(sigma):
    def gaussian_(offspring, ga_instance):
        mutated_offspring = offspring.copy()
        for idx, individual in enumerate(mutated_offspring):
            mutation_point = np.random.randint(0, len(individual))
            other_point = np.random.choice([i for i in range(len(individual)) if i != mutation_point])
            
            variation = np.random.normal(0, sigma)
            mutated_offspring[idx, mutation_point] += variation
            mutated_offspring[idx, other_point] -= variation
            
            # Ensure values remain within valid range
            mutated_offspring[idx, mutation_point] = max(0, min(1, mutated_offspring[idx, mutation_point]))
            mutated_offspring[idx, other_point] = max(0, min(1, mutated_offspring[idx, other_point]))
            
        return mutated_offspring
    
    return gaussian_

mutation_operators = {
    "directional": mutate_directional(),
    "gaussian": mutate_gaussian(0.1),
    "gaussian_0.5": mutate_gaussian(0.5),
    "gaussian_1": mutate_gaussian(1),
}
