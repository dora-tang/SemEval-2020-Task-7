# task 2 specific
from util import *


def get_dataset(csv_data, text_field, preprocess='cbow', mask_token=None):

    # define fields
    label_field = data.Field(sequential=False, use_vocab=False, dtype=torch.float)
    raw_field = data.RawField()

    def preprocess_cbow(train):
        log.info('preprocess for cbow')
        for j in ['1', '2']:
            original = train[f'original{j}']
            old_word_list = []
            context_list = []
            for i in original:
                old_word = re.findall('<(.*)/>', i)[0]
                l = re.findall('(.*)<.*/>(.*)', i)[0]
                l = [j.strip() for j in l]
                context = ' '.join(l).strip()
                old_word_list.append(old_word)
                context_list.append(context)

            train[f'old_word{j}'] = old_word_list
            train[f'new_word{j}'] = train[f'edit{j}']
            train[f'context{j}'] = context_list
        return train

    def preprocess_transformer(train, mask_token):
        log.info('preprocess for transformer')
        for j in ['1', '2']:
            new_list = []
            original = train[f'original{j}']
            edit = train[f'edit{j}']
            mask_list = []
            for original_sent, edit_word in zip(original, edit):
                new_sent = re.sub('<.*/>', f'<{edit_word}/>', original_sent)
                new_list.append(new_sent)
                mask_sent = re.sub('<.*/>', f'<{mask_token}/>', original_sent)
                mask_list.append(mask_sent)
            train[f'new{j}'] = new_list
            train[f'mask{j}'] = mask_list
        return train

    examples = []

    if preprocess == 'cbow':
        csv_data = preprocess_cbow(csv_data)
        field_base = [('id', raw_field),
                      ('original1', text_field), ('old_word1', text_field),
                      ('new_word1', text_field), ('context1', text_field),
                      ('original2', text_field), ('old_word2', text_field),
                      ('new_word2', text_field), ('context2', text_field),
                      ]
        cols_base = [csv_data['id'],
                     csv_data['original1'], csv_data['old_word1'],
                     csv_data['new_word1'],  csv_data['context1'],
                     csv_data['original2'], csv_data['old_word2'],
                     csv_data['new_word2'],  csv_data['context2'],
                     ]
    elif preprocess == 'transformer':
        assert mask_token is not None
        cvs_data = preprocess_transformer(csv_data, mask_token)
        field_base = [('id', raw_field),
                      ('original1', text_field), ('new1', text_field), ('mask1', text_field),
                      ('original2', text_field), ('new2', text_field), ('mask2', text_field),
                      ]
        cols_base = [csv_data['id'],
                     csv_data['original1'], csv_data['new1'],  csv_data['mask1'],
                     csv_data['original2'], csv_data['new2'],  csv_data['mask2'],
                     ]

    total = len(csv_data['id'])

    if 'meanGrade1' not in csv_data.columns:
        # test data has no labels
        field = field_base
        cols = cols_base
    else:
        # train / dev has labels
        field_label = [('meanGrade1', label_field),
                       ('meanGrade2', label_field),
                       ('label', label_field), ]
        cols_label = [csv_data['meanGrade1'],
                      csv_data['meanGrade2'],
                      csv_data['label']]
        field = field_base + field_label
        cols = cols_base + cols_label

    for i in tqdm(zip(*cols), total=total):
        example = data.Example.fromlist(i, field)
        examples.append(example)

    return data.Dataset(examples, field)


def write_prediction(args, model, test_data, out_path, mode='minimal', in_path=''):
    iterator = data.BucketIterator(
        test_data, batch_size=args.bsz,
        sort_key=lambda x: len(x.original),
        sort_within_batch=False,
        shuffle=False,
        device=device)

    pred_list = []
    id_list = []
    score1_list = []
    score2_list = []

    model.eval()
    with torch.no_grad():
        for batch in iterator:
            prediction1, prediction2, pred_label = model(batch)
            pred_list += pred_label.data.tolist()
            id_list += batch.id
            score1_list += prediction1.data.tolist()
            score2_list += prediction2.data.tolist()

    df_out = pd.DataFrame({'id': id_list, 'pred': pred_list,
                           'score1': score1_list, "score2": score2_list})
    if mode == 'minimal':
        df = df_out[['id', 'pred']]
    elif mode == 'analysis':
        df_in = pd.read_csv(in_path).drop('grades1', axis=1).drop('grades2', axis=1)
        assert(sorted(df_in.id) == sorted(df_out.id)
               ), "ID mismatch between ground truth and prediction!"
        df = pd.merge(left=df_in, right=df_out, how='inner', on='id')
        df = df.sort_values(by=['meanGrade1'], ascending=False)

        df['score1'] = round(df['score1'], 6)
        df['score2'] = round(df['score2'], 6)
        df = df[['id',
                 'original1', 'edit1', 'meanGrade1', 'score1',
                 'original2', 'edit2', 'meanGrade2', 'score2',
                 'label', 'pred']]

    df.to_csv(out_path, index=False, mode='w')
    log.info(f'Save prediction to {out_path}')

    return df


def _train(args, model, batch, optimizer, cls_scorer, reg_scorer):
    criterion = nn.MSELoss(reduction='sum')
    bsz = len(batch.id)

    model.train()
    optimizer.zero_grad()
    prediction1, prediction2, pred_label = model(batch)
    loss1 = criterion(prediction1, batch.meanGrade1)
    loss2 = criterion(prediction2, batch.meanGrade2)
    loss_mse = loss1 + loss2
    #loss = loss1 / 2 + loss2 / 2
    # if rankingloss:
    #     criterion2 = nn.MarginRankingLoss(
    #         margin=0.0, size_average=None, reduce=None, reduction='none')
    #
    #     label = batch.label
    #     label_mask = (label != 0).float()
    #     label[label != 1] = -1
    #     loss3 = criterion2(prediction1, prediction2, label)
    #     loss3 = (loss3 * label_mask).sum()
    #     loss += loss3
    #
    #     label = batch.label
    #     label_mask = (label != 0).float()
    #     label[label != 2] = -1
    #     label[label == 2] = 1
    #     loss4 = criterion2(prediction2, prediction1, label)
    #     loss4 = (loss4 * label_mask).sum()
    #     loss += loss4

    # r_mask[r_mask == 0] = -1
    # r = ((prediction1 - prediction2).abs() *
    #     r_mask.float() * (batch.label != 0).float()).sum()
    # r = 0
    # pred_label = torch.argmax(torch.stack(
    #     [prediction1, prediction2], dim=-1), dim=-1) + 1
    # correct_mask = (pred_label.int() == batch.label.int()).float()
    # wrong_mask = (pred_label.int() != batch.label.int()).float()
    # r -= ((prediction1 - prediction2).abs() *
    #       correct_mask.float() * (batch.label != 0).float()).sum()
    # r += ((prediction1 - prediction2).abs() *
    #       wrong_mask.float() * (batch.label != 0).float()).sum()
    #
    # #loss += torch.clamp(r, 0) * 5
    # loss += r * 100  # *  10
    # loss = torch.clamp(loss, 1e-9)
    # if cls:
    #     label = batch.label
    #     label_mask = (label != 0).float()
    #
    #     criterion3 = nn.CrossEntropyLoss(reduction='none')
    #     label = (label - 1)
    #     label[label == -1] = 0
    #     loss3 = criterion3(prediction3, label.long())
    #     loss3 = (loss3 * label_mask).sum()
    #
    #     # criterion3 = nn.CrossEntropyLoss(reduction='sum')
    #     # loss3 = criterion3(prediction3,  label.long())
    #     # loss += loss3
    # if args.diff:
    #     loss_diff = criterion(prediction1-prediction2, batch.meanGrade1 - batch.meanGrade2)
    #     loss = loss_mse + args.diff * loss_diff
    # elif args.ranking:
    #     criterion2 = nn.MarginRankingLoss(
    #         margin=args.margin, size_average=None, reduce=None, reduction='none')
    #     label = batch.label
    #     label_mask = (label != 0).float()
    #     label[label != 1] = -1
    #     loss3 = criterion2(prediction1, prediction2, label)
    #     loss3 = (loss3 * label_mask).sum()
    #     loss = loss_mse + args.ranking * loss3
    # else:
    #     loss = loss_mse
    loss = loss_mse
    loss.div(bsz).backward()
    clip_grad_norm_(model.parameters(), args.grad_norm)
    optimizer.step()

    # update metrics
    cls_scorer.update_metrics(predictions=pred_label, labels=batch.label,
                              diff=(batch.meanGrade1-batch.meanGrade2).abs(), loss=loss.item(), bsz=bsz)
    reg_scorer.update_metrics(sse=loss_mse.item(), num=bsz*2)

    return loss, loss_mse


def evaluate(args, model, val_data):

    iterator = data.BucketIterator(
        val_data, batch_size=args.bsz,
        sort_key=lambda x: len(x.original),
        sort_within_batch=False,
        shuffle=False,
        device=device)

    criterion = nn.MSELoss(reduction='sum')
    cls_scorer = ClassificationMetrics(loss=True, acc_reward=True, f1=False)
    reg_scorer = RegressionMetrics(loss=False, rmse=True, rmse_plus=False,
                                   spearman=True, pearson=True)

    #epoch_loss = 0
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            bsz = len(batch.id)
            prediction1, prediction2, pred_label = model(batch)

            # pred_label3 = torch.argmax(prediction3, dim=-1) + 1
            # acc3 += (pred_label3.long() == batch.label.long()).sum()

            # rmse
            loss1 = criterion(prediction1, batch.meanGrade1)
            loss2 = criterion(prediction2, batch.meanGrade2)
            loss_mse = loss1 + loss2
            # if args.diff:
            #     loss_diff = criterion(prediction1-prediction2, batch.meanGrade1 - batch.meanGrade2)
            #     loss = loss_mse + loss_diff
            # elif ranking:
            #     criterion2 = nn.MarginRankingLoss(
            #         margin=0.0, size_average=None, reduce=None, reduction='none')
            #     label = batch.label
            #     label_mask = (label != 0).float()
            #     label[label != 1] = -1
            #     loss3 = criterion2(prediction1, prediction2, label)
            #     loss3 = (loss3 * label_mask).sum()
            #     loss = loss_mse + loss3
            # else:
            #     loss = loss_mse
            loss = loss_mse
            #epoch_loss += loss.item()

            # update metrics
            reg_scorer.update_metrics(sse=loss_mse.item(), num=bsz*2,
                                      predictions=torch.stack(
                                          [prediction1, prediction2], dim=-1).flatten(),
                                      labels=torch.stack([batch.meanGrade1, batch.meanGrade2], dim=-1).flatten())
            cls_scorer.update_metrics(predictions=pred_label, labels=batch.label,
                                      diff=(batch.meanGrade1-batch.meanGrade2).abs(), loss=loss.item(), bsz=bsz)

        #val_loss = epoch_loss / len(iterator)
        #log_template = [f"loss: {val_loss:<7.4f}"]
        #  print(len(iterator))
        log_template = []

        # calculate metrics
        reg_metrics = reg_scorer.get_metrics(reset=False)
        cls_metrics = cls_scorer.get_metrics(reset=False)
        metrics = cls_metrics
        metrics.update(reg_metrics)
        for k, v in metrics.items():
            log_template += [f"{k}: {v:.4f}"]
        log_string = " | ".join(log_template)

        # print metrics
        phase = "Validation"
        log.info(f"{phase:<10} | " + log_string)
        #acc = metrics['accuracy']
    return metrics


def train_loop(args, model, optimizer, train_data, val_data):

    train_iterator = data.BucketIterator(
        train_data, batch_size=args.bsz,
        sort_key=lambda x: len(x.original),
        # important to shuffle train data between epochs
        shuffle=True,
        device=device)

    model = model.to(device)
    best_val_acc = -float('inf')
    best_epoch = -1
    global_step = 0

    if args.tensorboard:
        TB_train_log = SummaryWriter(os.path.join(args.exp_dir, 'tensorboard_train'))
        TB_validation_log = SummaryWriter(os.path.join(args.exp_dir, 'tensorboard_val'))

    args.t_total = len(train_iterator) * args.epochs
    scheduler = Scheduler(args, optimizer)

    try:
        for epoch in range(1, args.epochs+1):
            log.info(f'Epoch: {epoch:02}')
            # train_loss = 0

            cls_scorer = ClassificationMetrics(loss=True, acc_reward=True, f1=False)
            reg_scorer = RegressionMetrics(loss=False, rmse=True, rmse_plus=False,
                                           spearman=False, pearson=False)

            if args.track:
                # track training time
                generator = tqdm(enumerate(train_iterator, 1), total=len(train_iterator))
            else:
                generator = enumerate(train_iterator, 1)

            for idx, batch in generator:
                global_step += 1
                loss, loss_mse = _train(args, model, batch, optimizer, cls_scorer, reg_scorer)
                scheduler.step()

                # train_loss += loss.item()

                # log every 1/3 epoch
                log_interval = len(train_iterator) // 3
                if idx % log_interval == 0:
                    log.info(f'{idx} / {len(train_iterator)}')
                    log_template = []
                    # log_template = [f"loss: {train_loss/idx:<7.4f}"]

                    # calculate train metrics
                    reg_metrics = reg_scorer.get_metrics(reset=False)
                    cls_metrics = cls_scorer.get_metrics(reset=False)
                    metrics = cls_metrics
                    metrics.update(reg_metrics)
                    for k, v in metrics.items():
                        log_template += [f"{k}: {v:.4f}"]
                    log_string = " | ".join(log_template)

                    if args.tensorboard:
                        for key, value in metrics.items():
                            TB_train_log.add_scalar(key, value, global_step)

                    phase = "Train"
                    log.info(f"{phase:<10} | " + log_string)

                    val_metrics = evaluate(args, model, val_data)
                    val_acc = val_metrics['accuracy']
                    scheduler.step(val_acc)

                    if args.tensorboard:
                        for key, value in val_metrics.items():
                            TB_validation_log.add_scalar(key, value, global_step)

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_model = copy.deepcopy(model)
                        log.info(f"Best validation accuracy found: {best_val_acc:.4f}\n")
                        best_epoch = epoch

    except KeyboardInterrupt:
        pass

    if args.tensorboard:
        TB_train_log.close()
        TB_validation_log.close()

    log.info('\nFinish training.')
    log.info(f'Best Epoch: {best_epoch}')

    return best_model
