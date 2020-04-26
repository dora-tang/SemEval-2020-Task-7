from util import *


def evaluate_output_task1(true_path, pred_path):
    truth = pd.read_csv(true_path, usecols=["id", "meanGrade"])
    pred = pd.read_csv(pred_path, usecols=["id", "pred"])
    assert sorted(truth.id) == sorted(
        pred.id
    ), "ID mismatch between ground truth and prediction!"
    criterion = nn.MSELoss(reduction="sum")
    size = len(truth.id)
    pred = torch.tensor(pred.pred)
    labels = torch.tensor(truth.meanGrade)

    scorer = RegressionMetrics(
        loss=True, rmse=True, rmse_plus=True, spearman=True, pearson=True
    )
    sse = criterion(pred, labels).item()
    # sse = np.sum((data['meanGrade']-data['pred'])**2)
    scorer.update_metrics(sse=sse, num=size, predictions=pred, labels=labels)
    metrics = scorer.get_metrics(reset=False)
    log_str = [f"{k}: {v:.6f}" for k, v in metrics.items()]
    print(" | ".join(log_str))
    return metrics


def evaluate_output_task2(true_path, pred_path):
    truth = pd.read_csv(true_path, usecols=["id", "label", "meanGrade1", "meanGrade2"])
    pred = pd.read_csv(pred_path, usecols=["id", "pred"])
    assert sorted(truth.id) == sorted(
        pred.id
    ), "ID mismatch between ground truth and prediction!"

    size = len(truth.id)
    pred = torch.tensor(pred.pred)
    labels = torch.tensor(truth.label)
    diff = torch.tensor(truth.meanGrade1 - truth.meanGrade2).abs()

    scorer = ClassificationMetrics(loss=False, acc_reward=True, f1=False)
    scorer.update_metrics(predictions=pred, labels=labels, diff=diff)
    metrics = scorer.get_metrics(reset=False)
    log_str = [f"{k}: {v:.6f}" for k, v in metrics.items()]
    print(" | ".join(log_str))
    return metrics


def main(task, true_path, pred_path):
    if task == "task1":
        return evaluate_output_task1(true_path, pred_path)
    elif task == "task2":
        return evaluate_output_task2(true_path, pred_path)


if __name__ == "__main__":
    """"
    sys.argv[1]: task1, task2
    sys.argv[2]: true_path
    sys.argv[3]: pred_path
    """

    main(*sys.argv[1:])
