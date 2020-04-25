from util import *


class Scheduler(object):
    def __init__(self, args, optimizer):
        if args.schedule == "linear_schedule_with_warmup":
            # t_total = math.ceil(len(task['train_data']) / args.bsz) * args.epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=args.t_total,
            )

        elif args.schedule == "reduce_on_plateau":
            if args.task == "task1":
                mode = "min"
            elif args.task == "task2":
                mode = "max"
            scheduler = ReduceLROnPlateau(
                optimizer, mode=mode, factor=0.5, patience=3, min_lr=0, threshold=0.0001
            )

        elif args.schedule == "none":
            scheduler = None

        # self.schedule = args.schedule
        self.scheduler = scheduler

    def step(self, val_metric=None):
        # if self.schedule == "linear_schedule_with_warmup" and val_loss is None:
        if isinstance(self.scheduler, LambdaLR) and val_metric is None:
            self.scheduler.step()

        # elif self.schedule == "reduce_on_plateau" and val_loss is not None:
        if isinstance(self.scheduler, ReduceLROnPlateau) and val_metric is not None:
            self.scheduler.step(val_metric)
            log.info(
                f"\t# validation passes without improvement: {self.scheduler.num_bad_epochs}"
            )


class Trainer(object):
    def __init__(self, args, model):

        self.model = model.to(device)

        # optimizer
        if args.optimizer == "adam":
            optimizer = optim.Adam(
                model.parameters(), lr=args.lr, eps=args.adam_epsilon
            )
        # default: weight_decay=0.01
        elif args.optimizer == "pytorch_adamw":
            optimizer = AdamW(model.parameters(), lr=args.lr, eps=args.adam_epsilon)

        # default: weight_decay=0
        elif args.optimizer == "huggingface_adamw":
            optimizer = huggingface_AdamW(
                model.parameters(), lr=args.lr, eps=args.adam_epsilon
            )
        self.optimizer = optimizer

        # tensorboard
        if args.tensorboard:
            self.TB_train_log = SummaryWriter(
                os.path.join(args.exp_dir, "tensorboard_train")
            )
            self.TB_validation_log = SummaryWriter(
                os.path.join(args.exp_dir, "tensorboard_val")
            )

        self._init_scorers()

    def train_loop(self, args, task):

        train_iterator = data.BucketIterator(
            task.train_data,
            batch_size=args.bsz,
            sort_key=lambda x: len(x.original),
            # important to shuffle train data between epochs
            shuffle=True,
            device=device,
        )

        scorers = self.tr_scorers
        global_step = 0
        best_epoch = -1

        # scheduler
        args.t_total = len(train_iterator) * args.epochs
        self.scheduler = Scheduler(args, self.optimizer)

        try:
            for epoch in range(1, args.epochs + 1):
                log.info(f"\nEpoch: {epoch:02}")
                [scorers[k].reset() for k in scorers]

                # scorer.reset()

                if args.track:
                    # track training time
                    generator = tqdm(
                        enumerate(train_iterator, 1), total=len(train_iterator)
                    )
                else:
                    generator = enumerate(train_iterator, 1)

                for idx, batch in generator:
                    # check if batch is shuffled between train epochs
                    # if idx == 1:
                    #     log.info(batch.original[0])
                    self._train(args, self.model, batch, self.optimizer, scorers)
                    # print(optimizer.state)
                    self.scheduler.step()
                    global_step += 1
                    # print(scheduler.get_lr()[0])

                    # log every 1/3 epoch
                    log_interval = len(train_iterator) // 3
                    if idx % log_interval == 0:
                        log.info(f"{idx} / {len(train_iterator)}")

                        # calculate train metric
                        tr_metrics_dict = {}
                        for k in scorers:
                            tr_metrics_dict.update(scorers[k].get_metrics(reset=False))

                        # log train metric
                        log_template = [
                            f"{k}: {v:.4f}" for k, v in tr_metrics_dict.items()
                        ]
                        log_string = " | ".join(log_template)
                        log.info(f"{'Train':<10} | " + log_string)

                        # log train metric to tensorboard
                        if args.tensorboard:
                            for key, value in tr_metrics_dict.items():
                                self.TB_train_log.add_scalar(key, value, global_step)

                        # calculate validation metric
                        val_metrics_dict = self.evaluate(
                            args, self.model, task.val_data
                        )
                        if args.tensorboard:
                            for key, value in val_metrics_dict.items():
                                self.TB_validation_log.add_scalar(
                                    key, value, global_step
                                )

                        val_metric = val_metrics_dict[self.val_metric_name]
                        self.scheduler.step(val_metric)

                        comp_fn = (
                            lambda x, y: x < y
                            if self.val_metric_mode == "min"
                            else x > y
                            # if self.val_metric_mode == "max"
                        )

                        if comp_fn(val_metric, self.best_val_metric):
                            self.best_val_metric = val_metric
                            best_model = copy.deepcopy(self.model)
                            log.info(
                                f"\t[Best validation {self.val_metric_name} found]"
                            )
                            best_epoch = epoch

        except KeyboardInterrupt:
            pass

        if args.tensorboard:
            self.TB_train_log.close()
            self.TB_validation_log.close()

        log.info("\nFinish training.")
        log.info(f"Best Epoch: {best_epoch}")
        return best_model

    @staticmethod
    def _train():
        raise NotImplementedError

    @staticmethod
    def evaluate():
        raise NotImplementedError

    @staticmethod
    def write_prediction():
        raise NotImplementedError

    def _init_scorers(self):
        raise NotImplementedError


class Trainer1(Trainer):
    def __init__(self, *kw):
        super().__init__(*kw)

    def _init_scorers(self):
        tr_reg_scorer = RegressionMetrics(
            loss=True, rmse=True, rmse_plus=False, spearman=False, pearson=False
        )
        self.tr_scorers = {"reg_scorer": tr_reg_scorer}
        self.val_metric_name = "loss"
        self.val_metric_mode = "min"
        self.best_val_metric = float("inf")

        # eval_reg_scorer = RegressionMetrics(
        #     loss=True, rmse=True, rmse_plus=True, spearman=True, pearson=True)
        # self.eval_scorers ={"reg_scorer": eval_reg_scorer}

    @staticmethod
    def _train(args, model, batch, optimizer, scorer):
        bsz = len(batch.id)
        model.to(device)
        model.train()
        optimizer.zero_grad()
        out = model(batch)
        loss = out["loss"]
        loss.div(bsz).backward()
        clip_grad_norm_(model.parameters(), args.grad_norm)
        optimizer.step()
        scorer["reg_scorer"].update_metrics(sse=loss.item(), num=bsz)
        return loss

    @staticmethod
    def evaluate(args, model, val_data) -> Dict:
        iterator = data.BucketIterator(
            val_data,
            batch_size=args.bsz,
            sort_key=lambda x: len(x.original),
            sort_within_batch=False,
            shuffle=False,
            device=device,
        )

        scorer = RegressionMetrics(
            loss=True, rmse=True, rmse_plus=True, spearman=True, pearson=True
        )

        model.to(device)
        model.eval()

        with torch.no_grad():
            for batch in iterator:
                out = model(batch)
                loss = out["loss"]
                predictions = out["pred_score"]
                bsz = len(batch.meanGrade)
                scorer.update_metrics(
                    sse=loss.item(),
                    num=bsz,
                    predictions=predictions,
                    labels=batch.meanGrade,
                )

            metrics = scorer.get_metrics(reset=False)

            log_template = [f"{k}: {v:.4f}" for k, v in metrics.items()]
            log_string = " | ".join(log_template)
            log.info(f"{'Validation':<10} | " + log_string)

        return metrics

    @staticmethod
    def write_prediction(args, model, test_data, out_path, mode="minimal", in_path=""):
        """
        for test data without label
        output model prediction
        """
        iterator = data.BucketIterator(
            test_data,
            batch_size=args.bsz,
            sort_key=lambda x: len(x.original),
            sort_within_batch=False,
            shuffle=False,
            device=device,
        )

        pred_list = []
        id_list = []

        model.to(device)
        model.eval()

        with torch.no_grad():
            for batch in iterator:
                out = model(batch)
                predictions = out["pred_score"]
                pred_list += predictions.data.tolist()
                id_list += batch.id

        df_out = pd.DataFrame({"id": id_list, "pred": pred_list})

        if mode == "minimal":
            # test without label, minimal prediction for semeval submission
            df = df_out[["id", "pred"]]
        elif mode == "analysis":
            df_in = pd.read_csv(in_path).drop("grades", axis=1)
            assert sorted(df_in.id) == sorted(
                df_out.id
            ), "ID mismatch between ground truth and prediction!"
            df = pd.merge(left=df_in, right=df_out, how="inner", on="id")
            df = df.sort_values(by=["meanGrade"], ascending=False)

        df["pred"] = round(df["pred"], 6)

        df.to_csv(out_path, index=False, mode="w")
        log.info(f"Save prediction to {out_path}")
        return df


class Trainer2(Trainer):
    def __init__(self, *kw):
        super().__init__(*kw)

    def _init_scorers(self):

        cls_scorer = ClassificationMetrics(loss=True, acc_reward=True, f1=False)
        reg_scorer = RegressionMetrics(
            loss=False, rmse=True, rmse_plus=False, spearman=False, pearson=False
        )

        self.tr_scorers = {
            "cls_scorer": cls_scorer,
            "reg_scorer": reg_scorer,
        }
        self.val_metric_name = "accuracy"
        self.val_metric_mode = "max"
        self.best_val_metric = -float("inf")

    @staticmethod
    def _train(args, model, batch, optimizer, scorer):
        bsz = len(batch.id)

        model.to(device)
        model.train()

        optimizer.zero_grad()
        out = model(batch)
        loss = out["loss"]
        loss.div(bsz).backward()
        clip_grad_norm_(model.parameters(), args.grad_norm)
        optimizer.step()

        # update metrics
        scorer["cls_scorer"].update_metrics(
            predictions=out["pred_label"],
            labels=batch.label,
            diff=(batch.meanGrade1 - batch.meanGrade2).abs(),
            loss=loss.item(),
            bsz=bsz,
        )
        scorer["reg_scorer"].update_metrics(sse=loss.item(), num=bsz * 2)
        return loss

    @staticmethod
    def evaluate(args, model, val_data) -> Dict:

        iterator = data.BucketIterator(
            val_data,
            batch_size=args.bsz,
            sort_key=lambda x: len(x.original),
            sort_within_batch=False,
            shuffle=False,
            device=device,
        )

        cls_scorer = ClassificationMetrics(loss=True, acc_reward=True, f1=False)
        reg_scorer = RegressionMetrics(
            loss=False, rmse=True, rmse_plus=False, spearman=True, pearson=True
        )

        model.to(device)
        model.eval()

        with torch.no_grad():
            for batch in iterator:
                bsz = len(batch.id)
                out = model(batch)

                loss = out["loss"]
                prediction1 = out["pred_score1"]
                prediction2 = out["pred_score2"]
                pred_label = out["pred_label"]

                # update metrics
                reg_scorer.update_metrics(
                    sse=loss.item(),
                    num=bsz * 2,
                    predictions=torch.stack(
                        [prediction1, prediction2], dim=-1
                    ).flatten(),
                    labels=torch.stack(
                        [batch.meanGrade1, batch.meanGrade2], dim=-1
                    ).flatten(),
                )
                cls_scorer.update_metrics(
                    predictions=pred_label,
                    labels=batch.label,
                    diff=(batch.meanGrade1 - batch.meanGrade2).abs(),
                    loss=loss.item(),
                    bsz=bsz,
                )

            # calculate metrics
            metrics = {}
            reg_metrics = reg_scorer.get_metrics(reset=False)
            cls_metrics = cls_scorer.get_metrics(reset=False)
            metrics.update(cls_metrics)
            metrics.update(reg_metrics)

            # log metrics
            log_template = [f"{k}: {v:.4f}" for k, v in metrics.items()]
            log_string = " | ".join(log_template)
            log.info(f"{'Validation':<10} | " + log_string)
        return metrics

    @staticmethod
    def write_prediction(args, model, test_data, out_path, mode="minimal", in_path=""):
        iterator = data.BucketIterator(
            test_data,
            batch_size=args.bsz,
            sort_key=lambda x: len(x.original),
            sort_within_batch=False,
            shuffle=False,
            device=device,
        )

        pred_list = []
        id_list = []
        score1_list = []
        score2_list = []

        model.to(device)
        model.eval()

        with torch.no_grad():
            for batch in iterator:
                out = model(batch)
                prediction1 = out["pred_score1"]
                prediction2 = out["pred_score2"]
                pred_label = out["pred_label"]
                pred_list += pred_label.data.tolist()
                id_list += batch.id
                score1_list += prediction1.data.tolist()
                score2_list += prediction2.data.tolist()

        df_out = pd.DataFrame(
            {
                "id": id_list,
                "pred": pred_list,
                "score1": score1_list,
                "score2": score2_list,
            }
        )
        if mode == "minimal":
            df = df_out[["id", "pred"]]
        elif mode == "analysis":
            df_in = pd.read_csv(in_path).drop("grades1", axis=1).drop("grades2", axis=1)
            assert sorted(df_in.id) == sorted(
                df_out.id
            ), "ID mismatch between ground truth and prediction!"
            df = pd.merge(left=df_in, right=df_out, how="inner", on="id")
            df = df.sort_values(by=["meanGrade1"], ascending=False)

            df["score1"] = round(df["score1"], 6)
            df["score2"] = round(df["score2"], 6)
            # reorder columns
            df = df[
                [
                    "id",
                    "original1",
                    "edit1",
                    "meanGrade1",
                    "score1",
                    "original2",
                    "edit2",
                    "meanGrade2",
                    "score2",
                    "label",
                    "pred",
                ]
            ]

        df.to_csv(out_path, index=False, mode="w")
        log.info(f"Save prediction to {out_path}")

        return df
