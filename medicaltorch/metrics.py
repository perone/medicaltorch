from collections import defaultdict

from scipy import spatial
import numpy as np


class MetricManager(object):
    def __init__(self, metric_fns):
        self.metric_fns = metric_fns
        self.result_dict = defaultdict(float)
        self.num_samples = 0 
    
    def __call__(self, prediction, ground_truth):
        self.num_samples += len(prediction)
        for metric_fn in self.metric_fns:
            for p, gt in zip(prediction, ground_truth):
                res = metric_fn(p, gt)
                dict_key = metric_fn.__name__
                self.result_dict[dict_key] += res
            
    def get_results(self):
        res_dict = {}
        for key, val in self.result_dict.items():
            res_dict[key] = val / self.num_samples
        return res_dict
    
    def reset(self):
        self.num_samples = 0
        self.result_dict = defaultdict(float)
        

def numeric_score(prediction, groundtruth):
    """Computation of statistical numerical scores:

    * FP = False Positives
    * FN = False Negatives
    * TP = True Positives
    * TN = True Negatives

    return: tuple (FP, FN, TP, TN)
    """
    FP = np.float(np.sum((prediction == 1) & (groundtruth == 0)))
    FN = np.float(np.sum((prediction == 0) & (groundtruth == 1)))
    TP = np.float(np.sum((prediction == 1) & (groundtruth == 1)))
    TN = np.float(np.sum((prediction == 0) & (groundtruth == 0)))
    return FP, FN, TP, TN


def dice_score(prediction, groundtruth):
    pflat = prediction.flatten()
    gflat = groundtruth.flatten()
    d = (1 - spatial.distance.dice(pflat, gflat)) * 100.0
    if np.isnan(d):
        return 0.0
    return d


def jaccard_score(prediction, groundtruth):
    pflat = prediction.flatten()
    gflat = groundtruth.flatten()
    return (1 - spatial.distance.jaccard(pflat, gflat)) * 100.0


def hausdorff_score(prediction, groundtruth):
    return spatial.distance.directed_hausdorff(prediction, groundtruth)[0]


def precision_score(prediction, groundtruth):
    # PPV
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    if (TP + FP) <= 0.0:
        return 0.0

    precision = np.divide(TP, TP + FP)
    return precision * 100.0


def recall_score(prediction, groundtruth):
    # TPR, sensitivity
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    if (TP + FN) <= 0.0:
        return 0.0
    TPR = np.divide(TP, TP + FN)
    return TPR * 100.0


def specificity_score(prediction, groundtruth):
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    if (TN + FP) <= 0.0:
        return 0.0
    TNR = np.divide(TN, TN + FP)
    return TNR * 100.0


def intersection_over_union(prediction, groundtruth):
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    if (TP + FP + FN) <= 0.0:
        return 0.0
    return TP / (TP + FP + FN) * 100.0


def accuracy_score(prediction, groundtruth):
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    N = FP + FN + TP + TN
    accuracy = np.divide(TP + TN, N)
    return accuracy * 100.0
