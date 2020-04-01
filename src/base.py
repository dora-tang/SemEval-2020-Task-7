from util import *


def get_dataset(csv_data, text_field, preprocess='cbow', mask_token=None):

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
        return train

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
        return train

    # define fields
    label_field = data.Field(sequential=False, use_vocab=False, dtype=torch.float)
    raw_field = data.RawField()

    examples = []

    if preprocess == 'cbow':
        csv_data = preprocess_cbow(csv_data)
        field_base = [('id', raw_field),  ('original', text_field), ('new_word', text_field),
                      ('old_word', text_field),  ('context', text_field)]
        cols_base = [csv_data['id'], csv_data['original'], csv_data['new_word'],
                     csv_data['old_word'],  csv_data['context']]
    elif preprocess == 'transformer':
        assert mask_token is not None
        cvs_data = preprocess_transformer(csv_data, mask_token)
        field_base = [('id', raw_field), ('original', text_field),
                      ('new', text_field), ('mask', text_field), ]
        cols_base = [csv_data['id'],  csv_data['original'], csv_data['new'],  csv_data['mask']]

    total = len(csv_data['id'])

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

    for i in tqdm(zip(*cols), total=total):
        example = data.Example.fromlist(i, field)
        examples.append(example)

    return data.Dataset(examples, field)


def _train(args, model, batch, optimizer):
    criterion = nn.MSELoss(reduction='sum')
    model.train()
    optimizer.zero_grad()
    predictions = model(batch)
    loss = criterion(predictions, batch.meanGrade)
    # criterion2 = nn.MarginRankingLoss(margin=0.0, size_average=None, reduce=None, reduction='sum')
    # loss2 = criterion2(predictions[1], predictions[0], torch.ones_like(batch.meanGrade))
    # loss += loss2
    bsz = len(batch.meanGrade)
    loss.div(bsz).backward()
    clip_grad_norm_(model.parameters(), args.grad_norm)
    optimizer.step()
    return loss


def _evaluate(args, model, val_data):
    iterator = data.BucketIterator(
        val_data, batch_size=args.bsz,
        sort_key=lambda x: len(x.original),
        sort_within_batch=False,
        shuffle=False,
        device=device)
    criterion = nn.MSELoss(reduction='sum')
    epoch_loss = 0
    scorer = RMSE()
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch)
            loss = criterion(predictions, batch.meanGrade)
            bsz = len(batch.meanGrade)
            epoch_loss += loss.item()
            # scorer.accumulate(predictions, batch.meanGrade)
            scorer.accumulate(loss.item(), bsz)
        rmse = scorer.calculate(clear=True)
        val_loss = epoch_loss / len(iterator)
        log.info(f'\tValid Loss: {val_loss:.4f} | Valid RMSE: {rmse:.4f}')
    return val_loss


def train_loop(args, model, optimizer, train_data, val_data):
    train_iterator = data.BucketIterator(
        train_data, batch_size=args.bsz,
        sort_key=lambda x: len(x.original),
        # important to shuffle train data between epochs
        shuffle=True,
        device=device)

    model = model.to(device)
    best_val_loss = float('inf')
    best_epoch = -1

    try:
        for epoch in range(1, args.epochs+1):
            log.info(f'\nEpoch: {epoch:02}')
            train_loss = 0
            scorer = RMSE()

            if args.track:
                # track training time
                generator = tqdm(enumerate(train_iterator, 1), total=len(train_iterator))
            else:
                generator = enumerate(train_iterator, 1)

            for idx, batch in generator:
                # check if batch is shuffled between train epochs
                # if idx == 1:
                #     log.info(batch.original[0])
                loss = _train(args, model, batch, optimizer)
                train_loss += loss.item()
                bsz = len(batch.id)
                scorer.accumulate(loss.item(), bsz)

                # log every 1/3 epoch
                log_interval = len(train_iterator) // 3
                if idx % log_interval == 0:
                    rmse = scorer.calculate(clear=False)
                    log.info(f'{idx} / {len(train_iterator)}')
                    log.info(f'\tTrain Loss: {train_loss/idx:.4f} | Train RMSE: {rmse:.4f}')
                    val_loss = _evaluate(args, model, val_data)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model = copy.deepcopy(model)
                        log.info(f"\t[Best validation loss found]")
                        best_epoch = epoch
                        # best_model_state = copy.deepcopy(model.state_dict())
                        # torch.save(model.state_dict(), 'task-1-diff-bilstm-model.pth')
    except KeyboardInterrupt:
        pass

    log.info('\nFinish training.')
    log.info(f'Best Epoch: {best_epoch}')
    return best_model


def calc_rmse(args, model, test_data, all=True):
    """
    for test data with label
    calculate RMSE
    if all=True: also calculate RMSE@10, 20, 30, 40
    """
    iterator = data.BucketIterator(
        test_data, batch_size=args.bsz,
        sort_key=lambda x: len(x.original),
        sort_within_batch=False,
        shuffle=False,
        device=device)

    pred_list = []
    real_list = []

    model.eval()
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch)
            pred = predictions.data.tolist()
            real = batch.meanGrade.data.tolist()

            real_list += real
            pred_list += pred

    df = pd.DataFrame({'real': real_list, 'pred': pred_list})

    rmse = np.sqrt(np.mean((df['real'] - df['pred']) ** 2))
    log.info(f'RMSE: {rmse:.6f}')

    if all:
        df = df.sort_values(by=['real'], ascending=False)
        for percent in [10, 20, 30, 40]:
            size = math.ceil(len(df) * percent * 0.01)
            # top n % + bottom n %
            df2 = df[:size].append(df[-size:])
            rmse = np.sqrt(np.mean((df2['real'] - df2['pred'])**2))
            log.info(f'\tRMSE@{percent}: {rmse:.6f}')

    return df


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

            pred = predictions.data.tolist()
            pred_list += pred
            id_list += batch.id

            # if mode == 'minimal':
            # test without label, minimal prediction for semeval submission
            # rows = {'id': batch.id, 'pred': pred}
            # df = pd.DataFrame(rows)
            # if idx == 0:
            #     df.to_csv(out_path, index=False, mode='w', header=True)
            # else:
            #     df.to_csv(out_path, index=False, mode='a', header=False)

    df_out = pd.DataFrame({'id': id_list, 'pred': pred_list})
    if mode == 'minimal':
        df = df_out[['id', 'pred']]
    elif mode == 'analysis':
        df_in = pd.read_csv(in_path).drop('grades', axis=1)
        assert(sorted(df_in.id) == sorted(df_out.id)), "ID mismatch between ground truth and prediction!"
        df = pd.merge(left=df_in, right=df_out, how='inner', on='id')
        df = df.sort_values(by=['meanGrade'], ascending=False)

    df['pred'] = round(df['pred'], 6)
    df.to_csv(out_path, index=False, mode='w')
    log.info(f'Save prediction to {out_path}')
    #df = pd.read_csv(out_path)
    return df


class CBOW(nn.Module):
    def __init__(self, embedding, pad_idx, feature='edit-context'):
        super().__init__()
        # self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.embedding = embedding
        self.pad_idx = pad_idx

        embedding_dim = embedding.weight.shape[-1]
        hdim = embedding_dim * 4

        self.classifier = Classifier(d_inp=hdim, n_classes=1,
                                     cls_type="mlp", dropout=0.4, d_hid=256)
        self.pooler_max = Pooler(project=False, pool_type='max')
        self.pooler_mean = Pooler(project=False, pool_type='mean')
        self.feature = feature

    def forward_edit_context(self, batch):
        # mean pooling
        new_word = batch.new_word.squeeze(-1)
        new_word_emb = self.embedding(new_word)
        new = self.pooler_mean(new_word_emb)

        # max pooling
        context = batch.context
        context_emb = self.embedding(context)
        context = self.pooler_max(context_emb, context != self.pad_idx)

        m = [new, context, (context-new).abs(), context * new]
        pair_emb = torch.cat(m, dim=-1)
        pred = self.classifier(pair_emb).squeeze()
        return pred

    def forward_edit_original(self, batch):
        # mean pooling
        new_word = batch.new_word.squeeze(-1)
        new_word_emb = self.embedding(new_word)
        new = self.pooler_mean(new_word_emb)

        # mean pooling
        old_word = batch.old_word.squeeze(-1)
        old_word_emb = self.embedding(old_word)
        old = self.pooler_mean(old_word_emb)

        m = [new, old, (old-new).abs(), old * new]
        pair_emb = torch.cat(m, dim=-1)
        pred = self.classifier(pair_emb).squeeze()
        return pred

    def forward(self, batch):
        if self.feature == 'edit-context':
            return self.forward_edit_context(batch)
        elif self.feature == 'edit-original':
            return self.forward_edit_original(batch)

    # def forward(self, batch):
    #
    #     old_word = batch.old_word.squeeze(-1)
    #     old_word_emb = self.embedding(old_word)
    #     old = self.pooler_mean(old_word_emb)
    #
    #     new_word = batch.new_word.squeeze(-1)
    #     new_word_emb = self.embedding(new_word)
    #     new = self.pooler_mean(new_word_emb)
    #
    #     context = batch.context
    #     context_emb = self.embedding(context)
    #     context = self.pooler_max(context_emb, context != self.pad_idx)
    #
    #     old_diff = torch.cat([(old-new).abs(), old * new], dim=-1)
    #     old_context = torch.cat([(old-context).abs(), old * context], dim=-1)
    #     context_diff = torch.cat([(context-new).abs(), context * new], dim=-1)
    #     # context_diff = torch.cat([new, context, (context-new).abs(), context*new], dim=-1)
    #     # context_diff = torch.cat([context*new], dim=-1)
    #     # context_diff = torch.cat([(context-new).abs()], dim=-1)
    #     # context_diff = torch.cat([(context-new)**2], dim=-1)
    #
    #     # diff_emb = torch.cat([context_diff, old_context, old_diff], dim=-1)
    #     # diff_emb = torch.cat([context_diff,  old_diff], dim=-1)
    #     diff_emb = torch.cat([new, context, context_diff, ], dim=-1)
    #     pred = self.classifier(diff_emb).squeeze()
    #     # diff_emb2 = torch.cat([old, context, old_context, ], dim=-1)
    #     # pred2 = self.classifier(diff_emb2)
    #
    #     return pred  # , pred2.squeeze()


class PretrainedTransformer(nn.Module):
    def __init__(self, transformer, finetune=False, feature='edit-context', pad_token_id=0, sep_token_id=None,):
        super().__init__()
        self.transformer = transformer
        self.pad_token_id = pad_token_id
        self.sep_token_id = sep_token_id

        d_inp = transformer.config.hidden_size
        d_cls = d_inp * 4
        self.pooler = Pooler(project=False, d_inp=d_inp, pool_type="mean")
        self.feature = feature

        # d_proj = 256
        # d_cls = d_proj * 4
        # self.pooler = Pooler(project=True d_inp=d_inp, d_proj=d_proj, pool_type="mean")

        if not finetune:
            self.scalar_mix = ScalarMix(transformer.config.num_hidden_layers+1, do_layer_norm=False)
            self.classifier = Classifier(d_inp=d_cls, n_classes=1,
                                         cls_type="mlp", dropout=0.4, d_hid=256)
            self.freeze_transformer()
            # self.classifier = nn.Linear( d_cls, 1)
        else:
            self.scalar_mix = None
            self.classifier = nn.Linear(d_cls, 1)
            log.info('Finetune pretrained transformer!')
            log.info(f'The model has {count_parameters(self):,} trainable parameters')

    def freeze_transformer(self):
        log.info('Freezing pretrained transformer!')
        log.info(f'Before, the model has {count_parameters(self):,} trainable parameters')
        for name, param in self.named_parameters():
            if name.startswith('transformer'):
                param.requires_grad = False
        log.info(f'Now, the model has {count_parameters(self):,} trainable parameters')

    def forward_sentence(self, inp, mix=True,):

        inp_mask = (inp != self.pad_token_id) & (inp != self.sep_token_id)
        sep_mask = (inp == self.sep_token_id)

        outputs = self.transformer(inp, attention_mask=inp_mask)
        last_hidden_state, pooler_output, hidden_states = outputs

        if mix:
            hidden = self.scalar_mix(hidden_states)
        else:
            hidden = last_hidden_state

        span = torch.nonzero(input=sep_mask, as_tuple=True)[1].view(-1, 2)
        pool_mask = torch.zeros_like(sep_mask)
        for row, (start, end) in enumerate(span):
            pool_mask[row][start+1: end].fill_(1)
        pool_mask = pool_mask.bool()
        out = self.pooler(hidden, pool_mask)
        # out2 = self.pooler(hidden, pool_mask == 0)
        return out

    def forward_edit_context(self, batch):
        mix = self.scalar_mix is not None
        q1 = self.forward_sentence(batch.new, mix)
        q2 = self.forward_sentence(batch.mask, mix)
        m = [q1, q2, (q1-q2).abs(), q1*q2]
        pair_emb = torch.cat(m, dim=-1)
        pred = self.classifier(pair_emb).squeeze()
        return pred

    def forward_edit_original(self, batch):
        mix = self.scalar_mix is not None
        q1 = self.forward_sentence(batch.new, mix)
        q2 = self.forward_sentence(batch.original, mix)
        m = [q1, q2, (q1-q2).abs(), q1*q2]
        pair_emb = torch.cat(m, dim=-1)
        pred = self.classifier(pair_emb).squeeze()
        return pred

    def forward(self, batch):
        if self.feature == 'edit-context':
            return self.forward_edit_context(batch)
        elif self.feature == 'edit-original':
            return self.forward_edit_original(batch)

    # def forward(self, batch):
    #     mix = self.scalar_mix is not None
    #
    #     # q1 = self.forward_sentence(batch.original, mix)
    #     q2 = self.forward_sentence(batch.new, mix)
    #     q3 = self.forward_sentence(batch.mask, mix)
    #
    #     # m =[ q2, q2*q1, q2*q3, (q2-q1).abs(), (q2-q3).abs()]
    #     # m =[ q2, (q1-q3).abs(), (q2-q1).abs(), (q2-q3).abs()]
    #     # pair_emb = torch.cat([q1,q2, q1-q2,q1*q2], dim=-1)
    #     # pair_emb = torch.cat([q1,q2, q3,], dim=-1)
    #     # m = [q2, (q2-q3)]
    #     # m = [q2, q2-q3, q2-q1, q1-q3, ]
    #     # m = [q2, q1, q3, (q2-q1), (q2-q3), (q1-q3), q2*q1, q2*q3, q1* q3]
    #     # m = [q2, q1, q3, q2*q3, q1*q3]
    #     # m = [q2, q1, q3, q2*q3*q1]
    #     # m = [q2, q3, q2-q3, q2*q3]
    #     # m = [q2, q3, q1, q2-q3 ]
    #     # m = [q1, q2, q3]
    #     # m = [(q2-q3).abs(), q2*q3]
    #     m = [q2, q3, (q2-q3).abs(), q2*q3]
    #     pair_emb = torch.cat(m, dim=-1)
    #     pred = self.classifier(pair_emb).squeeze()
    #
    #     # diff_emb2 = torch.cat([q1, q3, (q1-q3).abs(), q1*q3], dim=-1)
    #     # pred2 = self.classifier(diff_emb2).squeeze()
    #
    #     return pred
    # def forward2(self, batch):
    #     #transformer = self.transformer
    #
    #     #text = batch.new
    #     # segments_tensors= None
    #
    #     o = batch.original2
    #     n = batch.new[:,1:]
    #     text = torch.cat([o, n ], dim=-1)
    #
    #     segments_ids_A = torch.ones_like(o)
    #     segments_ids_B = torch.zeros_like(n)
    #     segments_tensors = torch.cat([segments_ids_A ,segments_ids_B], dim=-1).to(device)
    #
    #     mask = text != 0
    #
    #     #bert(text,attention_mask = mask, token_type_ids=segments_tensors)
    #
    #
    #     text = text.to(device)
    #     nopad_mask = (text != 0)
    #     outputs = transformer(text, attention_mask= nopad_mask,token_type_ids=segments_tensors )
    #     last_hidden_state , pooler_output, hidden_states =outputs
    #     #out = pooler_output
    #
    #     mask = (mask & ((segments_tensors==0).bool()))
    #     out = self.pooler(last_hidden_state, mask)
    #     #log.info(out.shape)
    #
    #     pred = self.classifier(out)
    #     return pred.squeeze()
