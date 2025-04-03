import numpy as np

def cross_arithmetic(parents, offspring_size, ga_instance):
    offspring = np.empty(offspring_size)
    idx = 0
    for i in range(0, len(parents), 2):
        if i + 1 < len(parents):
            parent1 = parents[i]
            parent2 = parents[i + 1]
            
            alpha = np.random.rand(parent1.shape[0])
            
            child1 = alpha * parent1 + (1 - alpha) * parent2
            child2 = (1 - alpha) * parent1 + alpha * parent2
            
            child1 /= np.sum(child1)
            child2 /= np.sum(child2)

            if idx < offspring_size[0]:
                offspring[idx] = child1
                idx += 1
            
            if idx < offspring_size[0]:
                offspring[idx] = child2
                idx += 1

    while idx < offspring_size[0]:
        offspring[idx] = offspring[idx % max(1, idx)]
        idx += 1
            
    return offspring


def cross_convex_combination(n = 3):
    def _cross_convex_combination(parents, offspring_size, ga_instance):
        offspring = np.empty(offspring_size)
        
        idx = 0

        for i in range(0, len(parents), n):
            sub_parents = parents[i:min(i+n, len(parents))]
            
            actual_n = len(sub_parents)
            
            for _ in range(actual_n):
                if idx < offspring_size[0]:
                    weights = np.random.dirichlet(np.ones(actual_n))
                    child = np.dot(weights, sub_parents)
                    
                    offspring[idx] = child
                    idx += 1
        while idx < offspring_size[0]:
            offspring[idx] = offspring[idx % max(1, idx)]
            idx += 1
                
        return offspring

    return _cross_convex_combination

crossover_operators = {
    "arithmetic": cross_arithmetic,
    "convex_combination": cross_convex_combination
}