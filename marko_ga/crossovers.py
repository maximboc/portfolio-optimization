import numpy as np

def cross_arithmetic(parents):
    offspring = []
    for i in range(0, len(parents), 2):
        parent1 = parents[i]
        parent2 = parents[i + 1]

        # Arithmetic crossover: Weighted sum of parents
        alpha = np.random.rand()  # Random weight
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2

        # Append the children to the offspring
        offspring.append(child1)
        offspring.append(child2)

    return np.array(offspring)

def cross_convex_combination(parents, n=3):
    n = max(n, len(parents))

    offspring = []
    for i in range(0, len(parents), n):
        sub_parents = np.array(parents[i:n]) # [[0.3,0.2,.3],[...]]
        for j in range(n):
            weights = np.random.diritchlet(n, alpha=1.0) #[0.2,0.3,.5]
            offspring.append((weights @ sub_parents).flatten())

    return np.array(offspring)

crossover_operators = {
    "arithmetic": cross_arithmetic,
    "convex_combination": cross_convex_combination
}
