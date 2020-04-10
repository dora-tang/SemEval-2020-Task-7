# task 1 specific
from util import *


def get_dataset(csv_data, text_field, preprocess='cbow', mask_token=None):

    # define fields
    label_field = data.Field(sequential=False, use_vocab=False, dtype=torch.float)
    raw_field = data.RawField()

    def preprocess_cbow(train):
        # log.info('preprocess for cbow')
        original = train['original']
        old_word_list = []
        context_list = []
        for i in original:
            old_word = re.findall('<(.*)/>', i)[0]
            l = re.findall('(.*)<.*/>(.*)', i)[0]
            l = [j.strip() for j in l]
            context = ' '.join(l).strip()
            old_word_list.append(old_word)
            context_list.append(context)
        train['old_word'] = old_word_list
        train['new_word'] = train['edit']
        train['context'] = context_list

        csv_data = train
        field_base = [('id', raw_field),  ('original', text_field), ('new_word', text_field),
                      ('old_word', text_field),  ('context', text_field)]
        cols_base = [csv_data['id'], csv_data['original'], csv_data['new_word'],
                     csv_data['old_word'],  csv_data['context']]

        return csv_data,  field_base, cols_base

    def preprocess_transformer(train, mask_token):
        # log.info('preprocess for transformer')
        new_list = []
        original = train['original']
        edit = train['edit']
        mask_list = []
        for original_sent, edit_word in zip(original, edit):
            new_sent = re.sub('<.*/>', f'<{edit_word}/>', original_sent)
            new_list.append(new_sent)
            mask_sent = re.sub('<.*/>', f'<{mask_token}/>', original_sent)
            mask_list.append(mask_sent)
        train['new'] = new_list
        train['mask'] = mask_list

        csv_data = train
        field_base = [('id', raw_field), ('original', text_field),
                      ('new', text_field), ('mask', text_field), ]
        cols_base = [csv_data['id'],  csv_data['original'], csv_data['new'],  csv_data['mask']]

        return csv_data, field_base, cols_base

    if preprocess == 'cbow':
        csv_data, field_base, cols_base = preprocess_cbow(csv_data)
    elif preprocess == 'transformer':
        csv_data, field_base, cols_base = preprocess_transformer(csv_data, mask_token)

    if 'meanGrade' not in csv_data.columns:
        # if data has no labels
        field = field_base
        cols = cols_base
    else:
        # if data has labels
        field_label = [('meanGrade', label_field)]
        cols_label = [csv_data['meanGrade']]
        field = field_base + field_label
        cols = cols_base + cols_label

    total = len(csv_data['id'])
    examples = []
    for i in tqdm(zip(*cols), total=total):
        example = data.Example.fromlist(i, field)
        examples.append(example)

    return data.Dataset(examples, field)


def _train(args, model, batch, optimizer, scorer):
    bsz = len(batch.id)

    criterion = nn.MSELoss(reduction='sum')
    model.train()
    optimizer.zero_grad()
    predictions = model(batch)
    # print(predictions)
    # print(batch.meanGrade)
    loss = criterion(predictions, batch.meanGrade)
    # criterion2 = nn.MarginRankingLoss(margin=0.0, size_average=None, reduce=None, reduction='sum')
    # loss2 = criterion2(predictions[1], predictions[0], torch.ones_like(batch.meanGrade))
    # loss += loss2

    loss.div(bsz).backward()
    clip_grad_norm_(model.parameters(), args.grad_norm)
    optimizer.step()

    scorer.update_metrics(sse=loss.item(), num=bsz)

    return loss


def evaluate(args, model, val_data):
    iterator = data.BucketIterator(
        val_data, batch_size=args.bsz,
        sort_key=lambda x: len(x.original),
        sort_within_batch=False,
        shuffle=False,
        device=device)

    criterion = nn.MSELoss(reduction='sum')
    scorer = RegressionMetrics(loss=True, rmse=True, rmse_plus=True, spearman=True, pearson=True)

    # epoch_loss = 0
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch)
            loss = criterion(predictions, batch.meanGrade)
            bsz = len(batch.meanGrade)
            # epoch_loss += loss.item()
            scorer.update_metrics(sse=loss.item(), num=bsz,
                                  predictions=predictions, labels=batch.meanGrade)

        metrics = scorer.get_metrics(reset=False)
        # loss = epoch_loss / len(iterator)
        # log_template = [f"loss: {loss:<7.4f}"]
        log_template = []
        for k, v in metrics.items():
            log_template += [f"{k}: {v:.4f}"]
        log_string = " | ".join(log_template)
        phase = "Validation"
        log.info(f"{phase:<10} | " + log_string)

    return metrics


def train_loop(args, model, optimizer, train_data, val_data,):
    train_iterator = data.BucketIterator(
        train_data, batch_size=args.bsz,
        sort_key=lambda x: len(x.original),
        # important to shuffle train data between epochs
        shuffle=True,
        device=device)

    model = model.to(device)
    best_val_loss = float('inf')
    best_epoch = -1
    global_step = 0

    if args.tensorboard:
        TB_train_log = SummaryWriter(os.path.join(args.exp_dir, 'tensorboard_train'))
        TB_validation_log = SummaryWriter(os.path.join(args.exp_dir, 'tensorboard_val'))

    args.t_total = len(train_iterator) * args.epochs
    scheduler = Scheduler(args, optimizer)
    scorer = RegressionMetrics(loss=True, rmse=True, rmse_plus=False, spearman=False, pearson=False)

    try:
        for epoch in range(1, args.epochs+1):
            log.info(f'\nEpoch: {epoch:02}')
            # train_loss = 0
            scorer.reset()

            if args.track:
                # track training time
                generator = tqdm(enumerate(train_iterator, 1), total=len(train_iterator))
            else:
                generator = enumerate(train_iterator, 1)

            for idx, batch in generator:
                # check if batch is shuffled between train epochs
                # if idx == 1:
                #     log.info(batch.original[0])
                loss = _train(args, model, batch, optimizer, scorer)
                # print(optimizer.state)
                scheduler.step()
                global_step += 1
                # print(scheduler.get_lr()[0])
                # train_loss += loss.item()

                # log every 1/3 epoch
                log_interval = len(train_iterator) // 3
                if idx % log_interval == 0:
                    log.info(f'{idx} / {len(train_iterator)}')
                    # log_template = [f"loss: {train_loss/idx:<7.4f}"]
                    log_template = []
                    # calculate train metric
                    metrics = scorer.get_metrics(reset=False)
                    for k, v in metrics.items():
                        log_template += [f"{k}: {v:.4f}"]
                    log_string = " | ".join(log_template)
                    phase = "Train"
                    log.info(f"{phase:<10} | " + log_string)

                    if args.tensorboard:
                        for key, value in metrics.items():
                            TB_train_log.add_scalar(key, value, global_step)

                    # calculate validation metric
                    val_metrics = evaluate(args, model, val_data)
                    val_loss = val_metrics['loss']
                    scheduler.step(val_loss)

                    if args.tensorboard:
                        for key, value in val_metrics.items():
                            TB_validation_log.add_scalar(key, value, global_step)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model = copy.deepcopy(model)
                        log.info(f"\t[Best validation loss found]")
                        best_epoch = epoch

    except KeyboardInterrupt:
        pass

    if args.tensorboard:
        TB_train_log.close()
        TB_validation_log.close()

    log.info('\nFinish training.')
    log.info(f'Best Epoch: {best_epoch}')
    return best_model


def write_prediction(args, model, test_data, out_path, mode='minimal', in_path=''):
    """
    for test data without label
    output model prediction
    """
    iterator = data.BucketIterator(
        test_data, batch_size=args.bsz,
        sort_key=lambda x: len(x.original),
        sort_within_batch=False,
        shuffle=False,
        device=device)

    pred_list = []
    id_list = []

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(iterator):
            predictions = model(batch)

            pred_list += predictions.data.tolist()
            id_list += batch.id

            # if idx == 0:
            #     df.to_csv(out_path, index=False, mode='w', header=True)
            # else:
            #     df.to_csv(out_path, index=False, mode='a', header=False)

    df_out = pd.DataFrame({'id': id_list, 'pred': pred_list})
    if mode == 'minimal':
        # test without label, minimal prediction for semeval submission
        df = df_out[['id', 'pred']]
    elif mode == 'analysis':
        df_in = pd.read_csv(in_path).drop('grades', axis=1)
        assert(sorted(df_in.id) == sorted(df_out.id)
               ), "ID mismatch between ground truth and prediction!"
        df = pd.merge(left=df_in, right=df_out, how='inner', on='id')
        df = df.sort_values(by=['meanGrade'], ascending=False)

    df['pred'] = round(df['pred'], 6)
    df.to_csv(out_path, index=False, mode='w')
    log.info(f'Save prediction to {out_path}')
    #df = pd.read_csv(out_path)
    return df
