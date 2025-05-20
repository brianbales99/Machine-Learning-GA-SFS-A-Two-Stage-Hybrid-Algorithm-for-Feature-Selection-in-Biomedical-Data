import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, recall_score, precision_score, \
    f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.combine import SMOTEENN  # Import SMOTEENN for resampling


# Load dataset
def load_data():
    filename = 'O_Lung(in).csv'  # Update filename for the new dataset
    dataframe = pd.read_csv(filename, header=None)  # Read without a header

    # Separate features and target label
    X = dataframe.iloc[:, :-1].values  # All columns except the last one as features
    y = dataframe.iloc[:, -1].values  # Last column as labels

    # Encode string labels to integers (e.g., "Mesothelioma" to 0 or 1)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    return X, y


# Normalize data
def normalize_data(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)


# Apply SMOTEENN to balance the data
def apply_smoteenn(X, y):
    smote_enn = SMOTEENN(random_state=42)
    X_resampled, y_resampled = smote_enn.fit_resample(X, y)
    return X_resampled, y_resampled


# Calculate the evaluation measure M for feature pre-selection
def calculate_measure_M(X, y):
    return np.var(X, axis=0)  # Using variance as a simple filter measure


# Generalized evaluation function for cross-validated metrics with SMOTEENN
def evaluate_metrics(X, y, classifier, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs, accuracies, balanced_accuracies, recalls, precisions, f1_scores = [], [], [], [], [], []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Apply SMOTEENN to the training set
        X_train_resampled, y_train_resampled = apply_smoteenn(X_train, y_train)

        clf = classifier
        clf.fit(X_train_resampled, y_train_resampled)

        y_pred = clf.predict(X_test)

        # Calculate AUC if classifier supports predict_proba
        if hasattr(clf, "predict_proba"):
            y_probs = clf.predict_proba(X_test)[:, 1]
            aucs.append(roc_auc_score(y_test, y_probs))
        else:
            aucs.append(float("nan"))

        accuracies.append(accuracy_score(y_test, y_pred))
        balanced_accuracies.append(balanced_accuracy_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))

    # Return the mean of each metric
    return {
        'AUC': np.nanmean(aucs),
        'Accuracy': np.mean(accuracies),
        'Balanced Accuracy': np.mean(balanced_accuracies),
        'Recall': np.mean(recalls),
        'Precision': np.mean(precisions),
        'F1 Score': np.mean(f1_scores)
    }


# Main hybrid feature selection algorithm using SVM
def feature_selection_hybrid(X, y, m_ratio=0.2, n_ratio=0.1, n_splits=5):
    num_features = X.shape[1]
    m = int(m_ratio * num_features)
    n = int(n_ratio * num_features)

    best_auc = -1
    best_metrics = None
    selected_features = None

    while best_auc <= 0:
        initial_features = np.random.choice(num_features, m, replace=False)
        X_initial = X[:, initial_features]

        metrics = evaluate_metrics(X_initial, y, SVC(probability=True, random_state=42), n_splits)
        initial_auc = metrics['AUC']

        if initial_auc > evaluate_metrics(X, y, SVC(probability=True, random_state=42), n_splits)['AUC']:
            best_auc = initial_auc
            best_metrics = metrics
            selected_features = initial_features

    # Main loop: feature addition and removal steps
    while True:
        measure_M = calculate_measure_M(X, y)
        candidate_indices = np.argsort(measure_M)[-int(0.5 * len(measure_M)):]
        preselected_features = np.random.choice(candidate_indices, n, replace=False)
        candidates = [f for f in preselected_features if f not in selected_features]

        # Step b: Iteratively add features with SVM evaluation
        added = False
        while candidates:
            candidate_metrics = []
            for candidate in candidates:
                X_temp = X[:, np.append(selected_features, candidate)]
                candidate_metrics.append(evaluate_metrics(X_temp, y, SVC(probability=True, random_state=42), n_splits))

            best_candidate_index = np.argmax([metric['AUC'] for metric in candidate_metrics])
            best_candidate_metrics = candidate_metrics[best_candidate_index]

            if best_candidate_metrics['AUC'] > best_auc:
                best_auc = best_candidate_metrics['AUC']
                best_metrics = best_candidate_metrics
                selected_features = np.append(selected_features, candidates[best_candidate_index])
                candidates = [f for f in candidates if f != candidates[best_candidate_index]]
                added = True
            else:
                break

        # Step c: Iteratively remove the worst features with SVM evaluation
        removed = False
        while len(selected_features) > m:
            removal_metrics = []
            for feature in selected_features:
                temp_features = [f for f in selected_features if f != feature]
                X_temp = X[:, temp_features]
                removal_metrics.append(evaluate_metrics(X_temp, y, SVC(probability=True, random_state=42), n_splits))

            worst_feature_index = np.argmax([metric['AUC'] for metric in removal_metrics])
            worst_feature_metrics = removal_metrics[worst_feature_index]

            if worst_feature_metrics['AUC'] > best_auc:
                best_auc = worst_feature_metrics['AUC']
                best_metrics = worst_feature_metrics
                selected_features = np.delete(selected_features, worst_feature_index)
                removed = True
            else:
                break

        if not added and not removed:
            break

    return selected_features, best_metrics


# Execution
X, y = load_data()
X = normalize_data(X)

# Feature selection using SVM
selected_features, svm_metrics_during_selection = feature_selection_hybrid(X, y, m_ratio=0.2, n_ratio=0.1, n_splits=5)

# Final evaluation of the selected feature subset using both SVM and KNN
X_selected = X[:, selected_features]
svm_final_metrics = evaluate_metrics(X_selected, y, SVC(probability=True, random_state=42), n_splits=5)
knn_final_metrics = evaluate_metrics(X_selected, y, KNeighborsClassifier(), n_splits=5)

# Print results
print("Selected Features:", selected_features)
print("Total Number of Selected Features:", len(selected_features))
print("SVM Metrics During Selection:", svm_metrics_during_selection)
print("Final Evaluation Metrics with SVM:", svm_final_metrics)
print("Final Evaluation Metrics with KNN:", knn_final_metrics)
