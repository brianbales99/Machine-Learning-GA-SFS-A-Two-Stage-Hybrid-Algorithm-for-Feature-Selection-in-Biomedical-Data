import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
def load_data():
    filename = 'O_colon(in).csv'  # Update filename to the O_colon(in).csv dataset
    dataframe = pd.read_csv(filename, header=None)  # Load dataset without header
    array = dataframe.values
    X = array[:, :-1]  # Features (all columns except the last one)
    y = array[:, -1]  # Target variable (last column)
    return X, y

# Normalize the data
def normalize_data(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

# Main Genetic Algorithm without preselection process
def genetic_algorithm(X, y, num_generations, population_size, num_parents_mating, base_mutation_rate, n_splits=5):
    num_features = X.shape[1]

    # Initialize population randomly without preselection
    population = np.random.randint(2, size=(population_size, num_features))

    # Ensure each chromosome has at least 3 features selected
    for i in range(population_size):
        if np.sum(population[i, :]) < 3:
            selected_indices = np.random.choice(num_features, 3, replace=False)
            population[i, selected_indices] = 1

    best_chromosome = None
    best_score = 0

    # Helper function to evaluate population fitness with F1 score
    def evaluate_population_with_f1(population):
        scores = []
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for chromosome in population:
            if np.sum(chromosome) == 0:
                scores.append(0)
                continue

            X_selected = X[:, chromosome == 1]
            fold_scores = []
            for train_index, val_index in skf.split(X_selected, y):
                X_train_fold, X_val_fold = X_selected[train_index], X_selected[val_index]
                y_train_fold, y_val_fold = y[train_index], y[val_index]

                svm = SVC(probability=True)
                svm.fit(X_train_fold, y_train_fold)
                y_pred_fold = svm.predict(X_val_fold)
                fold_scores.append(f1_score(y_val_fold, y_pred_fold, average='weighted'))

            scores.append(np.mean(fold_scores))
        return np.array(scores)

    # Tournament Selection
    def tournament_selection(scores, population, num_parents_mating, tournament_size=3):
        parents = np.empty((num_parents_mating, num_features), dtype=int)
        for i in range(num_parents_mating):
            tournament_indices = np.random.choice(len(scores), tournament_size, replace=False)
            tournament_scores = scores[tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_scores)]
            parents[i, :] = population[winner_idx, :]
        return parents

    # Two-point Crossover
    def two_point_crossover(parents):
        offspring = np.empty((population_size - num_parents_mating, num_features))
        for k in range(offspring.shape[0]):
            parent1_idx = k % parents.shape[0]
            parent2_idx = (k + 1) % parents.shape[0]

            # Randomly select two crossover points
            point1 = np.random.randint(1, num_features - 1)
            point2 = np.random.randint(point1 + 1, num_features)

            # Create offspring by combining parts from both parents
            offspring[k, :point1] = parents[parent1_idx, :point1]
            offspring[k, point1:point2] = parents[parent2_idx, point1:point2]
            offspring[k, point2:] = parents[parent1_idx, point2:]
        return offspring

    # Adaptive Mutation Based on Fitness
    def adaptive_mutation(offspring, scores, base_mutation_rate):
        max_score = np.max(scores)
        for idx, chromosome in enumerate(offspring):
            # Calculate adaptive mutation rate for this chromosome
            adaptive_rate = base_mutation_rate * (1 - (scores[idx] / max_score)) if max_score > 0 else base_mutation_rate
            # Apply mutation with the adaptive rate
            for gene_idx in range(chromosome.shape[0]):
                if np.random.rand() < adaptive_rate:
                    chromosome[gene_idx] = 1 - chromosome[gene_idx]  # Flip bit
        return offspring

    # AUC evaluation helper for refining features
    def evaluate_auc_svm(X_selected, y):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs = []
        for train_index, test_index in skf.split(X_selected, y):
            X_train, X_test = X_selected[train_index], X_selected[test_index]
            y_train, y_test = y[train_index], y[test_index]
            svm = SVC(probability=True)
            svm.fit(X_train, y_train)
            y_probs = svm.predict_proba(X_test)[:, 1]
            aucs.append(roc_auc_score(y_test, y_probs))
        return np.mean(aucs)

    # Refinement by adding/removing features
    def refine_chromosome(chromosome):
        selected_features = np.where(chromosome == 1)[0]
        best_auc = evaluate_auc_svm(X[:, selected_features], y)

        # Add best features
        while True:
            candidate_features = np.where(chromosome == 0)[0]
            best_feature, new_auc = None, best_auc
            for candidate in candidate_features:
                X_candidate = np.column_stack((X[:, selected_features], X[:, candidate]))
                auc = evaluate_auc_svm(X_candidate, y)
                if auc > new_auc:
                    new_auc, best_feature = auc, candidate
            if best_feature is None:
                break
            chromosome[best_feature] = 1
            selected_features = np.append(selected_features, best_feature)
            best_auc = new_auc

        # Remove worst features
        while True:
            worst_feature, new_auc = None, best_auc
            for feature in selected_features:
                remaining_features = [f for f in selected_features if f != feature]
                auc = evaluate_auc_svm(X[:, remaining_features], y)
                if auc > new_auc:
                    new_auc, worst_feature = auc, feature
            if worst_feature is None:
                break
            chromosome[worst_feature] = 0
            selected_features = np.delete(selected_features, np.where(selected_features == worst_feature))
            best_auc = new_auc
        return chromosome

    # Main GA loop
    for generation in range(num_generations):
        scores = evaluate_population_with_f1(population)
        max_score = np.max(scores)
        if max_score > best_score:
            best_score = max_score
            best_chromosome = population[np.argmax(scores)]
        print(f"Generation {generation + 1}: Best Score = {best_score}")

        parents = tournament_selection(scores, population, num_parents_mating, tournament_size=3)
        offspring = two_point_crossover(parents)
        offspring = adaptive_mutation(offspring, scores, base_mutation_rate)
        for i in range(offspring.shape[0]):
            offspring[i, :] = refine_chromosome(offspring[i, :])

        population[:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = offspring

    return best_chromosome, best_score


# Main execution
X, y = load_data()
X = normalize_data(X)

# Parameters for Genetic Algorithm
num_generations = 15
population_size = 7
num_parents_mating = 7
base_mutation_rate = 0.05
n_splits = 5

best_chromosome, best_score = genetic_algorithm(X, y, num_generations, population_size, num_parents_mating,
                                                base_mutation_rate, n_splits=n_splits)

print(f"Best feature set: {np.where(best_chromosome == 1)[0]}")
print(f"Number of features in the final subset: {np.sum(best_chromosome)}")
print(f"Best F1 Score during GA: {best_score}")

# Final evaluation on the selected features using multiple metrics (SVM)
X_test_selected = X[:, best_chromosome == 1]
svm_final = SVC(probability=True)
svm_final.fit(X_test_selected, y)
test_predictions_svm = svm_final.predict(X_test_selected)

final_accuracy_svm = accuracy_score(y, test_predictions_svm)
final_balanced_accuracy_svm = balanced_accuracy_score(y, test_predictions_svm)
final_f1_svm = f1_score(y, test_predictions_svm, average='weighted')
final_precision_svm = precision_score(y, test_predictions_svm, average='weighted')
final_recall_svm = recall_score(y, test_predictions_svm, average='weighted')

print("\nSVM Results:")
print(f'Test Accuracy (SVM): {final_accuracy_svm:.4f}')
print(f'Test Balanced Accuracy (SVM): {final_balanced_accuracy_svm:.4f}')
print(f'Test F1 Score (SVM): {final_f1_svm:.4f}')
print(f'Test Precision (SVM): {final_precision_svm:.4f}')
print(f'Test Recall (SVM): {final_recall_svm:.4f}')

# Final evaluation on the selected features using multiple metrics (KNN)
knn_final = KNeighborsClassifier(n_neighbors=5)  # Add KNN with k=5
knn_final.fit(X_test_selected, y)
test_predictions_knn = knn_final.predict(X_test_selected)

final_accuracy_knn = accuracy_score(y, test_predictions_knn)
final_balanced_accuracy_knn = balanced_accuracy_score(y, test_predictions_knn)
final_f1_knn = f1_score(y, test_predictions_knn, average='weighted')
final_precision_knn = precision_score(y, test_predictions_knn, average='weighted')
final_recall_knn = recall_score(y, test_predictions_knn, average='weighted')

print("\nKNN Results:")
print(f'Test Accuracy (KNN): {final_accuracy_knn:.4f}')
print(f'Test Balanced Accuracy (KNN): {final_balanced_accuracy_knn:.4f}')
print(f'Test F1 Score (KNN): {final_f1_knn:.4f}')
print(f'Test Precision (KNN): {final_precision_knn:.4f}')
print(f'Test Recall (KNN): {final_recall_knn:.4f}')
