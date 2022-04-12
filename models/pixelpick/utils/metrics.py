import numpy as np
import torch
import warnings
from sklearn.metrics import confusion_matrix

import torch.nn.functional as F
from torch.linalg import norm


def compute_spatial_coverage_per_image(indices):
    x_loc, y_loc = indices
    x_loc, y_loc = np.expand_dims(x_loc, axis=1), np.expand_dims(y_loc, axis=1)
    x_loc_t, y_loc_t = x_loc.transpose(), y_loc.transpose()
    dist = np.sqrt((x_loc - x_loc_t) ** 2 + (y_loc - y_loc_t) ** 2)
    try:
        dist = dist[~np.eye(dist.shape[0], dtype=np.bool)].reshape(dist.shape[0], -1).sum() / 2
    except ValueError:
        return np.NaN
    return dist


def compute_spatial_coverage(masks):
    list_dists = list()
    for m in masks:
        list_dists.append(compute_spatial_coverage_per_image(np.where(m)))
    return np.nanmean(list_dists)


def compute_distance(emb, prototypes, l2_norm=False):
    if l2_norm:
        emb = emb / norm(emb, ord=2, dim=1, keepdim=True)  # 1 x 32 x h x w
        # prototypes = prototypes / norm(prototypes, ord=2, dim=-1, keepdim=True)

    n_classes = prototypes.shape[0]
    h, w = emb.shape[2:]
    grid = torch.zeros((n_classes, h, w), dtype=torch.float32)

    # prototypes: n_classes x 1 x 32
    for i, p in enumerate(prototypes):
        # p: 1 x 32
        p = p.unsqueeze(dim=2).unsqueeze(dim=3)  # 1 x 32 x 1 x 1
        p = p.repeat(1, 1, h, w)  # 1 x 32 x h x w
        sim = F.cosine_similarity(p, emb, dim=1)  # 1 x h x w
        # dist = ((emb - p) ** 2).sqrt()  # .sum(dim=1).sqrt()
        # dist = torch.exp(-dist).mean(dim=1)  # 1 x h x w
        # print(dist.shape)
        # grid[i] = dist.squeeze()   # h x w
        grid[i] = sim.squeeze()  # h x w
    return grid  # n_classes x h x w


def prediction(emb, prototypes, non_isotropic=False, return_distance=False):
    # emb: b x n_emb_dims x h x w
    # prototypes: n_classes x n_emb_dims
    b, n_emb_dims, h, w = emb.shape
    n_classes = prototypes.shape[0]

    if non_isotropic:
        emb = emb.unsqueeze(dim=1).repeat((1, n_classes, 1, 1, 1))  # b x n_classes x n_emb_dims x h x w
        prototypes = prototypes.view((1, n_classes, n_emb_dims, 1, 1)).repeat((b, 1, 1, h, w))  # b x n_classes x n_emb_dims x h x w

        dist = (emb - prototypes) ** 2  # .abs()

        return dist.sum(dim=2).argmin(dim=1)

    else:
        n_prototypes = prototypes.shape[1]
        emb_sq = emb.pow(exponent=2).sum(dim=1, keepdim=True)  # b x 1 x h x w
        emb_sq = emb_sq.transpose(1, 0).contiguous().view(1, -1).transpose(1, 0)  # (b x h x w) x 1

        prototypes_sq = prototypes.pow(exponent=2).sum(dim=2, keepdim=True)  # n_classes x n_prototypes x 1
        prototypes_sq = prototypes_sq.view(n_classes * n_prototypes, 1)  # (n_classes * n_prototypes) x 1

        emb = emb.transpose(1, 0).contiguous().view(n_emb_dims, -1).transpose(1, 0)  # (b * h * w) x n_emb_dims
        prototypes = prototypes.view(n_classes * n_prototypes, n_emb_dims)  # (n_classes * n_prototypes) x n_emb_dims

        # emb: (b * h * w) x n_emb_dims, prototypes.t(): n_emb_dims x (n_classes x n_prototypes)
        dist = emb_sq - 2 * torch.matmul(emb, prototypes.t()) + prototypes_sq.t()  # (b x h x w) x (n_classes * n_prototypes)
        dist = dist.view(b * h * w, n_classes, n_prototypes).sum(dim=-1)  # (b x h x w) x n_classes
        dist = dist.transpose(1, 0).view(-1, b, h, w).transpose(1, 0)  # b x n_classes h x w

        if return_distance:
            return dist.argmin(dim=1), dist
        else:
            return dist.argmin(dim=1)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.initialized = False
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = np.multiply(val, weight)
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum = np.add(self.sum, np.multiply(val, weight))
        self.count = self.count + weight
        self.avg = self.sum / self.count

    @property
    def value(self):
        return self.val

    @property
    def average(self):
        return np.round(self.avg, 5)


def batch_pix_accuracy(predict, target):
    predict = predict.int() + 1
    target = target.int() + 1

    pixel_labeled = (target > 0).sum()
    pixel_correct = ((predict == target) * (target > 0)).sum()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct.cpu().numpy(), pixel_labeled.cpu().numpy()


def batch_intersection_union(predict, target, num_class):
    predict = predict + 1
    target = target + 1

    predict = predict * (target > 0).long()
    intersection = predict * (predict == target).long()

    area_inter = torch.histc(intersection.float(), bins=num_class, max=num_class, min=1)
    area_pred = torch.histc(predict.float(), bins=num_class, max=num_class, min=1)
    area_lab = torch.histc(target.float(), bins=num_class, max=num_class, min=1)
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"
    return area_inter.cpu().numpy(), area_union.cpu().numpy()


def eval_metrics(prediction, target, num_classes, ignore_index):
    target = target.clone()
    target[target == ignore_index] = -1
    correct, labeled = batch_pix_accuracy(prediction, target)
    inter, union = batch_intersection_union(prediction, target, num_classes)
    return [np.round(correct, 5), np.round(labeled, 5), np.round(inter, 5), np.round(union, 5)]


class RunningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    @staticmethod
    def _fast_hist(label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "Pixel Acc": acc,
                "Mean Acc": acc_cls,
                "FreqW Acc": fwavacc,
                "Mean IoU": mean_iu,
                "All IoU": iu,
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class EvaluationMetrics(object):
    def __init__(self, n_classes: int):
        self.n_classes = n_classes
        self.metrics_dict = {}

    def _f1(self, true: np.ndarray, pred: np.ndarray) -> dict:
        if true.shape != pred.shape:
            warnings.warn(f"Truth {true.shape} and prediction {pred.shape} shapes should be equal")
            if len(true.shape) == 3:
                true = true[:len(pred), :len(pred[0]), :len(pred[0][0])]
            else:
                true = true[:len(pred), :len(pred[0])]
        labels = np.array(range(self.n_classes))
        pre = {}
        rec = {}

        for label in labels:
            true_positives = np.sum(np.logical_and(np.equal(true, label), np.equal(pred, label)))
            false_positives = np.sum(np.logical_and(np.logical_not(np.equal(true, label)), np.equal(pred, label)))
            false_negatives = np.sum(np.logical_and(np.equal(true, label), np.logical_not(np.equal(pred, label))))
            rec[f"Rec_{label}"] = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else np.core.numeric.NaN
            pre[f"Pre_{label}"] = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else np.core.numeric.NaN

        mean_pre = np.nanmean(list(pre.values()))
        mean_rec = np.nanmean(list(rec.values()))
        f1_dict = {"f1": 2 * mean_pre * mean_rec / (mean_pre + mean_rec), "Pre": mean_pre, "Rec": mean_rec}
        f1_dict.update(pre)
        f1_dict.update(rec)
        return f1_dict

    def _accuracy(self, true: np.ndarray, pred: np.ndarray) -> dict:
        if true.shape != pred.shape:
            warnings.warn(f"Truth {true.shape} and prediction {pred.shape} shapes should be equal")
            if len(true.shape) == 3:
                true = true[:len(pred), :len(pred[0]), :len(pred[0][0])]
            else:
                true = true[:len(pred), :len(pred[0])]
        return {"acc": np.mean(np.equal(true, pred))}

    def _IOU(self, true: np.ndarray, pred: np.ndarray) -> dict:
        if true.shape != pred.shape:
            warnings.warn(f"Truth {true.shape} and prediction {pred.shape} shapes should be equal")
            if len(true.shape) == 3:
                true = true[:len(pred), :len(pred[0]), :len(pred[0][0])]
            else:
                true = true[:len(pred), :len(pred[0])]
        labels = np.array(range(self.n_classes))
        ious = []
        for label in labels:
            intersection = np.sum(np.logical_and(np.equal(true, label), np.equal(pred, label)))
            union = np.sum(np.logical_or(
                np.equal(true, label), np.equal(pred, label)))
            label_iou = intersection * 1.0 / union if union > 0 else np.core.numeric.NaN
            ious.append(label_iou)
        iou_dict = {"IOU_{}".format(label): iou for label, iou in zip(labels, ious)}
        iou_dict["mean_IOU"] = np.nanmean(ious)
        return iou_dict

    def _conf_matrix(self, true: np.ndarray, pred: np.ndarray) -> dict:
        true = true.flatten()
        pred = pred.flatten()
        conf = confusion_matrix(y_true=true, y_pred=pred, labels=np.arange(self.n_classes))
        return {"conf_matrix": conf.tolist()}

    def update(self, true: np.ndarray, pred: np.ndarray) -> dict:
        # y_true = np.argmax(true, -1)
        # y_pred = np.argmax(pred, -1)
        self.metrics_dict.update(self._accuracy(true, pred,))
        self.metrics_dict.update(self._IOU(true, pred,))
        self.metrics_dict.update(self._f1(true, pred,))
        self.metrics_dict.update(self._conf_matrix(true, pred,))
