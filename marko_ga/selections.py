import random
import numpy as np

def custom_tournament_selection(fitness, num_parents, ga_instance):
    selected_parents = []
    selected_indices = []
    for _ in range(num_parents):
        candidates = random.sample(range(len(fitness)), 2)
        f1, f2 = fitness[candidates[0]], fitness[candidates[1]]
        if f1 > f2:
            winner = candidates[0]
        else:
            winner = candidates[1]
        selected_parents.append(ga_instance.population[winner].copy())
        selected_indices.append(winner)
    return np.array(selected_parents), np.array(selected_indices)

def custom_top_k_selection(fitness, num_parents, ga_instance):
    fitness_sorted = sorted(range(len(fitness)), key=lambda k: fitness[k], reverse=True)
    parents = np.empty((num_parents, ga_instance.population.shape[1]))
    for parent_num in range(num_parents):
        parents[parent_num, :] = ga_instance.population[fitness_sorted[parent_num], :].copy()
    return parents, np.array(fitness_sorted[:num_parents])

def custom_best_selection(fitness, num_parents, ga_instance):
    best_index = np.argmax(fitness)
    best_parent = ga_instance.population[best_index].copy()
    selected_parents = np.array([best_parent] * num_parents)
    selected_indices = np.array([best_index] * num_parents)
    return selected_parents, selected_indices

def custom_threshold_selection(threshold):
    def custom_threshold_selection_(fitness, num_parents, ga_instance):
        selected_indices = [i for i, fit in enumerate(fitness) if fit > threshold]
        selected_indices = selected_indices[:num_parents]
        selected_parents = ga_instance.population[selected_indices].copy()
        return np.array(selected_parents), np.array(selected_indices)
    return custom_threshold_selection_

selection_operators = {
    "tournament": custom_tournament_selection,
    "top_k": custom_top_k_selection,
    "best": custom_best_selection,
    "threshold_10": custom_threshold_selection(10),
    "threshold_20": custom_threshold_selection(20)
}
