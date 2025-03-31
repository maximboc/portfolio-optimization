# selection_operators.py
import random
import numpy as np

def custom_tournament_selection(solutions, fitness):
    selected_parents = []
    for _ in range(len(solutions) // 2):
        candidates = random.sample(list(range(len(solutions))), 2)
        f1, f2 = fitness[candidates[0]], fitness[candidates[1]]
        if f1 < f2:
            selected_parents.append(solutions[candidates[0]])
        else:
            selected_parents.append(solutions[candidates[1]])
    return np.array(selected_parents)

def custom_top_k_selection(solutions, fitness, k=10):
    sorted_indices = np.argsort(fitness)[:k]
    return np.array(solutions)[sorted_indices]

def custom_best_selection(solutions, fitness):
    best_index = np.argmax(fitness)
    return np.array([solutions[best_index]])

def custom_threshold_selection(solutions, fitness, threshold=0.5):
    selected_parents = [solution for solution, fit in zip(solutions, fitness) if fit > threshold]
    return np.array(selected_parents)

selection_operators = {
    "tournament": custom_tournament_selection,
    "top_k": custom_top_k_selection,
    "best": custom_best_selection,
    "threshold": custom_threshold_selection
}
