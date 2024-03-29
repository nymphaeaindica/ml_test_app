"""Calculate module for test app"""
import numpy as np
import pandas as pd


target_id_col = 2  # column index of targetTrackID
cmSpeed_z_col = 12  # column index of cmSpeed_z


def cut_anms(features, labels, target_id, cut_ranges):
    """Selecting necessary anomalies """
    n_input = []
    if cmSpeed_z_col in cut_ranges.index:
        tmp = np.logical_and(features[:, cmSpeed_z_col] >= cut_ranges.loc[cmSpeed_z_col][1],
                             features[:, cmSpeed_z_col] <= cut_ranges.loc[cmSpeed_z_col][2])
        tmp_ind = np.nonzero(tmp)[0]
        new_labels = labels[tmp_ind, :]
        new_targets = target_id[tmp_ind, :]
        # n_all = len(np.unique(new_targets))
        # n_all = np.unique(new_targets)   # input targets
        n_input = np.unique(new_targets)
    else:
        n_all = len(np.unique(target_id))
    del new_targets, new_labels

    for ind in cut_ranges.index:
        tmp = np.logical_and(features[:, ind] >= cut_ranges.loc[ind][1], features[:, ind] <= cut_ranges.loc[ind][2])
        tmp_ind = np.nonzero(tmp)[0]
        features = features[tmp_ind, :]
        target_id = target_id[tmp_ind, :]
        labels = labels[tmp_ind, :]
    n = len(np.unique(target_id))  # targets after simple
    n_simple = np.unique(target_id)
    return features, labels, target_id,  n_input, n_simple


def y_true_target(y_true, target_id):
    """Convert anomaly labels to target labels."""
    y = []
    for target in np.unique(target_id):
        ind = np.nonzero(target_id == target)[0]
        tmp = y_true[ind[0]]
        y.append(tmp)
    return y


def read_data(path_list, label_list, features_index, cut_ranges):
    """
    Reader for .aux file.
    Takes necessary features, labels and targets from .dat files
    into numpy arrays.
    """
    features = np.empty([0, len(features_index)])
    labels = np.empty([0, 1])
    target_id = np.empty([0, 1])

    max_val = 0
    n_input = 0
    n_simple = 0
    for ind, path in enumerate(path_list):
            data = pd.read_csv(path, header=None, sep=' ').to_numpy()
            data = np.unique(data[:, :-2], axis=0)
            tmp_target_id = data[:, [target_id_col]]
            tmp_features = data
            if label_list[ind] == 1:
                tmp_labels = np.ones([tmp_features.shape[0], 1])
            else:
                tmp_labels = np.zeros([tmp_features.shape[0], 1])
            tmp_features, tmp_labels, tmp_target_id,  n_input, n_simple = cut_anms(tmp_features, tmp_labels, tmp_target_id, cut_ranges)     # не обрезает фичи
            labels = np.concatenate((labels, tmp_labels), axis=0)
            target_id = np.concatenate((target_id, tmp_target_id + max_val), axis=0)
            max_val = max(target_id) + 1
            features = np.concatenate((features, tmp_features[:, features_index]), axis=0)

    return features, labels, target_id, n_input, n_simple


def dat_keeper(dat, ind, features_index, cut_ranges):
    """
    Reader for single .dat file
    Takes necessary features, labels and targets from file
    into numpy arrays.
    """
    features = np.empty([0, len(features_index)])
    labels = np.empty([0, 1])
    target_id = np.empty([0, 1])
    data = pd.read_csv(dat, header=None, sep=' ').to_numpy()
    data = np.unique(data[:, :-2], axis=0)
    tmp_target_id = data[:, [target_id_col]]
    tmp_features = data
    if ind == 1:
        tmp_labels = np.ones([tmp_features.shape[0], 1])
    else:
        tmp_labels = np.zeros([tmp_features.shape[0], 1])
    tmp_features, tmp_labels, tmp_target_id, n_input, n_simple = cut_anms(tmp_features, tmp_labels, tmp_target_id,
                                                                          cut_ranges)
    labels = np.concatenate((labels, tmp_labels), axis=0)
    target_id = np.concatenate((target_id, tmp_target_id), axis=0)
    # max_val = max(target_id) + 1
    features = np.concatenate((features, tmp_features[:, features_index]), axis=0)
    return features, labels, target_id, n_input, n_simple



#
# def graph(scores, particular_score, thr):
#     all = len(scores) + 1
#     y_axis = np.empty([0, 1])
#     for i in thr:
#         k = 0
#         for n in particular_score:
#             if n >= i:
#                 k = k + 1
#         y = k / all
#         y_axis = np.append(y_axis, y)
#     return y_axis


def output_target(scores, particular_score, fixed_thr):
    """Count number of targets after prediction for fixed threshold"""
    all = len(scores) + 1
    y_axis = np.empty([0, 1])
    k = 0
    for n in particular_score:
        if n >= fixed_thr:
            k = k + 1
    return k

# def read_all(aux, features_index, cut_ranges):
#     """
#     Reader for .aux file.
#     Takes necessary features, labels and targets from .dat files
#     into numpy arrays.
#     """
#     path_list = aux[1]
#     label_list = aux[0].to_numpy()
#     features = np.empty([0, len(features_index)])
#     labels = np.empty([0, 1])
#     target_id = np.empty([0, 1])
#
#     max_val = 0
#     n_input = 0
#     n_simple = 0
#     for ind, path in enumerate(path_list):
#             data = pd.read_csv(path, header=None, sep=' ').to_numpy()
#             data = np.unique(data[:, :-2], axis=0)
#             tmp_target_id = data[:, [target_id_col]]
#             tmp_features = data
#             if label_list[ind] == 1:
#                 tmp_labels = np.ones([tmp_features.shape[0], 1])
#             else:
#                 tmp_labels = np.zeros([tmp_features.shape[0], 1])
#             tmp_features, tmp_labels, tmp_target_id,  n_input, n_simple = cut_anms(tmp_features, tmp_labels, tmp_target_id, cut_ranges)
#             labels = np.concatenate((labels, tmp_labels), axis=0)
#             target_id = np.concatenate((target_id, tmp_target_id + max_val), axis=0)
#             max_val = max(target_id) + 1
#             features = np.concatenate((features, tmp_features[:, features_index]), axis=0)
#
#     return features, labels, target_id, n_input, n_simple
