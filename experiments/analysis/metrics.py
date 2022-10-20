import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score


def compute_max_accuracy(predictions, golds):
    pairs = list(zip(predictions, golds))
    pairs.sort(key=lambda x: x[0], reverse=True)

    best_acc, best_threshold = None, None
    positives_so_far = 0
    negatives_to_go = len([g for g in golds if g == 0])
    for i, (prediction, gold) in enumerate(pairs):
        assert gold in [0, 1]

        if gold:
            positives_so_far += 1
        else:
            positives_so_far -= 1

        acc = (positives_so_far + negatives_to_go) / len(golds)
        if not best_acc or acc > best_acc:
            best_acc = acc
            best_threshold = prediction

    return best_acc, best_threshold


def compute_f1(predictions, golds, minority_label=0):
    # take the (1-x) since our minority class is 0
    if minority_label == 0:
        golds = 1-np.array(golds)
        predictions = 1-np.array(predictions)
    precision, recall, thresholds = precision_recall_curve(np.array(golds), np.array(predictions), pos_label=1)
    f1_scores = 2*recall*precision/(recall+precision+1e-13)

    return np.max(f1_scores), float(thresholds[np.argmax(f1_scores)])


def compute_auc(predictions, golds):
    return roc_auc_score(np.array(golds), np.array(predictions))


if __name__ == "__main__":
    accuracy, threshold = compute_max_accuracy([0, 0.3, 0.2, 0.4], [0, 1, 0, 0])
    assert accuracy == 0.75 and threshold == 0.3
