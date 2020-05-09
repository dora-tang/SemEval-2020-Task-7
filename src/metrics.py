from typing import Dict, Iterable, List, Sequence, Tuple, Union, Optional
from overrides import overrides

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import math

import torch
from allennlp.training.metrics import Metric, F1Measure


# copied from the [github](https://github.com/nyu-mll/jiant/blob/master/jiant/allennlp_mods/correlation.py) of [jiant library](https://github.com/nyu-mll/jiant)
@Metric.register("correlation")
class Correlation(Metric):
    """Aggregate predictions, then calculate specified correlation"""

    def __init__(self, corr_type):
        self._predictions = []
        self._labels = []
        if corr_type == "pearson":
            corr_fn = pearsonr
        elif corr_type == "spearman":
            corr_fn = spearmanr
        else:
            raise ValueError("Correlation type not supported")
        self._corr_fn = corr_fn
        self.corr_type = corr_type

    def _correlation(self, labels, predictions):
        corr = self._corr_fn(labels, predictions)
        if self.corr_type in ["pearson", "spearman"]:
            corr = corr[0]
        return corr

    def __call__(self, predictions, labels):
        """ Accumulate statistics for a set of predictions and labels.

        Values depend on correlation type; Could be binary or multivalued. This is handled by sklearn.

        Args:
            predictions: Tensor or np.array
            labels: Tensor or np.array of same shape as predictions
        """
        # Convert from Tensor if necessary
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        # Verify shape match
        assert predictions.shape == labels.shape, (
            "Predictions and labels must"
            " have matching shape. Got:"
            " preds=%s, labels=%s" % (str(predictions.shape), str(labels.shape))
        )

        predictions = list(predictions.flatten())
        labels = list(labels.flatten())

        self._predictions += predictions
        self._labels += labels

    def get_metric(self, reset=False):
        correlation = self._correlation(self._labels, self._predictions)
        if reset:
            self.reset()
        return correlation

    @overrides
    def reset(self):
        self._predictions = []
        self._labels = []


class RMSE(object):
    """
    RMSE
    """

    def __init__(self):
        self.n_instance = 0
        self.sum_of_square = 0

    def __call__(self, sse: Union[float, torch.Tensor], num: int):
        if isinstance(sse, torch.Tensor):
            sse = sse.item()
        self.sum_of_square += sse
        self.n_instance += num

    def get_metric(self, reset=False) -> float:
        rmse = np.sqrt(self.sum_of_square / self.n_instance)
        if reset:
            self.reset()
        return rmse

    def reset(self):
        self.n_instance = 0
        self.sum_of_square = 0


class LossMetric(object):
    """
    calculate average loss per instance
    """

    def __init__(self):
        self.loss = 0.0
        self.n_instance = 0

    def __call__(self, loss: Union[float, torch.Tensor], num: int):
        # loss is the sum of a batch, num is bsz
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        assert isinstance(loss, float)
        self.loss += loss
        self.n_instance += num

    def get_metric(self, reset=False) -> float:
        average_loss = self.loss / self.n_instance
        if reset:
            self.reset()
        return average_loss

    def reset(self):
        self.loss = 0.0
        self.n_instance = 0


class RMSEPlus(object):
    """
    for full: RMSE
    for top n% + bottom n%: RMSE@10, RMSE@20, RMSE@30, RMSE@40
    """

    def __init__(self):
        self.pred_list = []
        self.real_list = []

    def __call__(
        self, predictions: Union[torch.Tensor, List], labels: Union[torch.Tensor, List]
    ):
        if isinstance(predictions, torch.Tensor):
            # predictions = predictions.detach().cpu().numpy()
            predictions = predictions.data.tolist()
        if isinstance(labels, torch.Tensor):
            # labels = labels.detach().cpu().numpy()
            labels = labels.data.tolist()

        self.real_list += labels
        self.pred_list += predictions

    def get_metric(self, reset=False) -> Dict:
        metrics = {}
        df = pd.DataFrame({"real": self.real_list, "pred": self.pred_list})
        metrics["rmse"] = np.sqrt(np.mean((df["real"] - df["pred"]) ** 2))

        df = df.sort_values(by=["real"], ascending=False)
        for percent in [10, 20, 30, 40]:
            size = math.ceil(len(df) * percent * 0.01)
            # top n % + bottom n %
            df2 = df[:size].append(df[-size:])
            rmse = np.sqrt(np.mean((df2["real"] - df2["pred"]) ** 2))
            metrics[f"rmse{percent}"] = rmse
        if reset:
            self.reset()

        return metrics

    def reset(self):
        self.pred_list = []
        self.real_list = []


class AccReward(object):
    """
    accuracy and reward
    true labels: 0, 1, 2 (should ignore instances with 0 for both measures)
    """

    def __init__(self):
        self.n_instance = 0
        self.correct = 0.0
        self.reward = 0.0

    def __call__(self, predictions, labels, diff):
        # diff: difference between gold scores
        labels = labels.long().detach()
        predictions = predictions.long().detach()
        diff = diff.abs().detach()

        # number of true labels with 1 or 2, ignore 0
        label_mask = (labels != 0).float()
        num = label_mask.sum()
        self.n_instance += num

        # acc
        correct = (predictions == labels).float() * label_mask
        self.correct += correct.sum()

        # reward
        correct_mask = (predictions == labels).float()
        wrong_mask = (predictions != labels).float()
        reward = diff * (correct_mask - wrong_mask) * label_mask
        self.reward += reward.sum()

    def get_metric(self, reset=False) -> Tuple[float, float]:
        acc = float(self.correct) / self.n_instance
        reward = float(self.reward) / self.n_instance
        if reset:
            self.reset()
        # return {"accuracy": acc, "reward": reward}
        return acc, reward

    def reset(self):
        self.n_instance = 0
        self.correct = 0
        self.reward = 0


class RegressionMetrics(object):
    """
    regression task metrics
    """

    def __init__(
        self, loss=False, rmse=False, rmse_plus=False, spearman=False, pearson=False
    ):
        self.scorers = {}
        if loss:
            self.scorers["loss"] = LossMetric()
        if rmse:
            self.scorers["rmse"] = RMSE()
        if rmse_plus:
            self.scorers["rmse_plus"] = RMSEPlus()
        if spearman:
            self.scorers["spearman"] = Correlation("spearman")
            # self.scorers["spearman"] = SpearmanCorrelation()
        if pearson:
            self.scorers["pearson"] = Correlation("pearson")
            # self.scorers["pearson"] = PearsonCorrelation()

    def update_metrics(
        self, sse=None, num=None, predictions=None, labels=None,
    ):
        for k in ["loss", "rmse"]:
            if k in self.scorers:
                assert sse is not None and num is not None
                self.scorers[k](sse, num)
        for k in ["rmse_plus", "spearman", "pearson"]:
            if k in self.scorers:
                assert predictions is not None and labels is not None
                self.scorers[k](predictions, labels)

    def get_metrics(self, reset=False) -> Dict:
        metrics = {}
        if "loss" in self.scorers:
            loss = self.scorers["loss"].get_metric(reset)
            metrics.update({"loss": loss})
        if "rmse" in self.scorers:
            rmse = self.scorers["rmse"].get_metric(reset)
            metrics.update({"rmse": rmse})
        if "rmse_plus" in self.scorers:
            rmse_dict = self.scorers["rmse_plus"].get_metric(reset)
            metrics.update(rmse_dict)
        if "spearman" in self.scorers:
            spearman = self.scorers["spearman"].get_metric(reset)
            metrics.update({"spearman": spearman})
        if "pearson" in self.scorers:
            pearson = self.scorers["pearson"].get_metric(reset)
            metrics.update({"pearson": pearson})
        return metrics

    def reset(self):
        for k in self.scorers:
            self.scorers[k].reset()


class ClassificationMetrics(object):
    """
    classification task metrics
    """

    def __init__(self, loss=False, acc_reward=False, f1=False):
        self.scorers = {}
        if loss:
            self.scorers["loss"] = LossMetric()
        if acc_reward:
            self.scorers["acc_reward"] = AccReward()
        if f1:
            # only need this if classes are unbalanced
            self.scorers["f1_1"] = F1Measure(1)
            self.scorers["f1_2"] = F1Measure(2)

    def update_metrics(
        self, predictions=None, labels=None, diff=None, loss=None, bsz=None
    ):
        if "loss" in self.scorers:
            assert loss is not None and bsz is not None
            self.scorers["loss"](loss, bsz)
        if "acc_reward" in self.scorers:
            assert predictions is not None and labels is not None and diff is not None
            self.scorers["acc_reward"](predictions, labels, diff)

        if all(k in self.scorers for k in ["f1_1", "f1_2"]):
            assert predictions is not None and labels is not None
            # since we already have the pre label, convert back to logits shape for compatability
            logits = torch.nn.functional.one_hot(predictions, 3)
            self.scorers["f1_1"](logits, labels)
            self.scorers["f1_2"](logits, labels)

    def get_metrics(self, reset=False) -> Dict:
        metrics = {}
        if "loss" in self.scorers:
            loss = self.scorers["loss"].get_metric(reset)
            metrics.update({"loss": loss})
        if "acc_reward" in self.scorers:
            acc, reward = self.scorers["acc_reward"].get_metric(reset)
            metrics.update({"accuracy": acc, "reward": reward})
        if all(k in self.scorers for k in ["f1_1", "f1_2"]):
            pcs1, rcl1, f11 = self.scorers["f1_1"].get_metric(reset)
            pcs2, rcl2, f12 = self.scorers["f1_2"].get_metric(reset)
            # macro average
            pcs = (pcs1 + pcs2) / 2
            rcl = (rcl1 + rcl2) / 2
            f1 = (f11 + f12) / 2
            metrics.update({"f1": f1, "precision": pcs, "recall": rcl})
        return metrics

    def reset(self):
        for k in self.scorers:
            self.scorers[k].reset()
