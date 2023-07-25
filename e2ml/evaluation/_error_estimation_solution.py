import numpy as np

from sklearn.utils.validation import check_scalar, column_or_1d, check_random_state, check_consistent_length


def cross_validation(sample_indices, n_folds=5, random_state=None, y=None):
    """
    Performs a (stratified) k-fold cross-validation.

    Parameters
    ----------
    sample_indices : int
        Array of sample indices.
    n_folds : int, default=5
        Number of folds. Must be at least 2.
    random_state : int, RandomState instance or None, default=None
        `random_state` affects the ordering of the indices, which controls the randomness of each fold.

    Returns
    -------
    train : list
        Contains the training indices of each iteration, where train[i] represents iteration i.
    test : list
        Contains the test indices of each iteration, where test[i] represents iteration i.
    """
    # BEGIN SOLUTION

    # Check number of samples.
    sample_indices = column_or_1d(sample_indices, dtype=int).copy()

    # Check number of folds.
    n_folds = check_scalar(n_folds, "n_folds", target_type=int, min_val=2, max_val=len(sample_indices))

    # Check random state.
    random_state = check_random_state(random_state)

    # Check possible class labels.
    y = column_or_1d(y) if y is not None else np.zeros(len(sample_indices))
    check_consistent_length(sample_indices, y)
    classes, counts = np.unique(y, return_counts=True)
    classes = classes[np.argsort(-counts)]

    # Placeholder for fold indices.
    folds = [[] for _ in range(n_folds)]

    # Shuffle data according to random state.
    p = random_state.permutation(len(sample_indices))
    sample_indices = sample_indices[p]
    y = y[p]

    # Generate fold indices.
    fold_indices = np.arange(n_folds)

    for class_y in classes:
        is_class_y = y == class_y
        folds_class_y = list(np.array_split(sample_indices[is_class_y], n_folds))
        fold_lengths = [len(fold) for fold in folds]
        sort_idx = np.argsort(fold_lengths)
        for f_y, f in enumerate(fold_indices[sort_idx]):
            folds[f].extend(folds_class_y[f_y])

    # Placeholders for train and test indices.
    train, test = [], []
    for f_1 in fold_indices:
        test.append(folds[f_1])
        train_f_1 = []
        for f_2 in fold_indices:
            if f_1 != f_2:
                train_f_1.extend(folds[f_2])
        train.append(train_f_1)
    return train, test

    # END SOLUTION

