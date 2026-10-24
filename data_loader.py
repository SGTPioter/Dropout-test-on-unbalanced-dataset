import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_preprocessed_data(dataset_type, data_home="./data", random_state=67, return_scaler=False):
    print(f"\n--- Przygotowanie danych: WARIANT '{dataset_type.upper()}' ---")
    covtype = fetch_covtype()
    features = covtype.data.astype(np.float32)
    labels = (covtype.target - 1).astype(np.int64).reshape(-1, 1)

    if dataset_type == 'edited':
        # Wariant z uciętymi minimalnymi klasami i ze sztucznie zbalansowanymi licznościami klas
        mask_keep = ~np.isin(labels.flatten(), [3, 4])
        features = features[mask_keep]
        labels = labels[mask_keep]

        mapping = {0: 0, 1: 1, 2: 2, 5: 3, 6: 4}
        labels = np.vectorize(mapping.get)(labels)

        unique, counts = np.unique(labels, return_counts=True)
        min_class_count = np.min(counts)
        max_allowed = min_class_count * 3

        indices_to_keep = []
        for cls in unique:
            cls_indices = np.where(labels.flatten() == cls)[0]
            if cls in [0, 1] and len(cls_indices) > max_allowed:
                np.random.seed(random_state)
                sampled_indices = np.random.choice(cls_indices, max_allowed, replace=False)
                indices_to_keep.extend(sampled_indices)
            else:
                indices_to_keep.extend(cls_indices)

        np.random.seed(random_state)
        np.random.shuffle(indices_to_keep)

        features = features[indices_to_keep]
        labels = labels[indices_to_keep]
        num_classes = 5
    else:
        # czyste dane pobrane z sklearn
        num_classes = 7

    X_train, X_hold, y_train, y_hold = train_test_split(
        features, labels, test_size=0.30, stratify=labels, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_hold, y_hold, test_size=0.50, stratify=y_hold, random_state=random_state
    )

    scaler = StandardScaler()
    X_train[:, :10] = scaler.fit_transform(X_train[:, :10])
    X_val[:, :10] = scaler.transform(X_val[:, :10])
    X_test[:, :10] = scaler.transform(X_test[:, :10])

    print(f"Rozmiary zbiorów - Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    if return_scaler:
        return X_train, y_train, X_val, y_val, X_test, y_test, num_classes, scaler
    return X_train, y_train, X_val, y_val, X_test, y_test, num_classes