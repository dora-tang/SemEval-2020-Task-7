from util import *

# prevent gradient norm from being too large
GRAD_NORM = 5


def get_dataset2(csv_data, text_field, preprocess='cbow', mask_token=None):

    def preprocess_cbow(train):
        print('preprocess for cbow')
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
        print('preprocess for transformer')
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

    # define fields
    LABEL = data.Field(sequential=False, use_vocab=False, dtype=torch.float)
    ID = data.RawField()
    label_field = LABEL
    id_field = ID

    print("read data from csv")
    examples = []

    if preprocess == 'cbow':
        csv_data = preprocess_cbow(csv_data)
        field_base = [('id', id_field),
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
        field_base = [('id', id_field),
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


def write_prediction2(model, iterator, out_path):
    pred_list = []
    id_list = []

    model.eval()
    with torch.no_grad():
        for batch in iterator:

            prediction1, prediction2, pred_label = model(batch)

            pred = pred_label.data.tolist()
            pred_list += pred
            id = batch.id
            id_list += id

    df = pd.DataFrame({'id': np.array(id_list), 'pred': np.array(pred_list)})

    df.to_csv(out_path, index=False)
    print(f'Save prediction to {out_path}')

    return df


class Scorer2():
    def __init__(self):
        self.n_instance = 0
        self.correct = 0.
        self.reward = 0.

    def accumulate(self, pred, label, diff):
        label = label.long().detach()
        pred = pred.long().detach()
        diff = diff.abs().detach()

        # number of true labels with 1 or 2, ignore 0
        label_mask = (label != 0).float()
        num = label_mask.sum()
        self.n_instance += num

        # acc
        correct = (pred == label).float() * label_mask
        self.correct += correct.sum()

        # reward
        correct_mask = (pred == label).float()
        wrong_mask = (pred != label).float()
        reward = diff * (correct_mask - wrong_mask) * label_mask
        self.reward += reward.sum()

    def calculate(self, clear=True):
        acc = float(self.correct) / self.n_instance
        reward = float(self.reward) / self.n_instance
        if clear:
            self._clear()
        return acc, reward

    def _clear(self):
        self.n_instance = 0
        self.correct = 0
        self.reward = 0


def _train2(args, model, batch, optimizer, scorer):
    criterion = nn.MSELoss(reduction='sum')

    model.train()
    optimizer.zero_grad()
    prediction1, prediction2, pred_label = model(batch)
    loss1 = criterion(prediction1, batch.meanGrade1)
    loss2 = criterion(prediction2, batch.meanGrade2)
    loss_mse = loss1 + loss2
    #loss = loss1 / 2 + loss2 / 2

    scorer.accumulate(pred=pred_label, label=batch.label,
                      diff=(batch.meanGrade1-batch.meanGrade2).abs())

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
    if args.diff:
        loss_diff = criterion(prediction1-prediction2, batch.meanGrade1 - batch.meanGrade2)
        loss = loss_mse + args.diff * loss_diff
    elif args.ranking:
        criterion2 = nn.MarginRankingLoss(
            margin=args.margin, size_average=None, reduce=None, reduction='none')
        label = batch.label
        label_mask = (label != 0).float()
        label[label != 1] = -1
        loss3 = criterion2(prediction1, prediction2, label)
        loss3 = (loss3 * label_mask).sum()
        loss = loss_mse + args.ranking * loss3
    else:
        loss = loss_mse
    bsz = len(batch.id)
    loss.div(bsz).backward()
    clip_grad_norm_(model.parameters(), GRAD_NORM)
    optimizer.step()

    return loss, loss_mse


def evaluate2(args, model, iterator):
    criterion = nn.MSELoss(reduction='sum')
    epoch_loss = 0
    model.eval()
    scorer = RMSE()

    scorer2 = Scorer2()
    with torch.no_grad():
        for batch in iterator:
            prediction1, prediction2, pred_label = model(batch)

            # pred_label3 = torch.argmax(prediction3, dim=-1) + 1
            # acc3 += (pred_label3.long() == batch.label.long()).sum()

            # rmse
            loss1 = criterion(prediction1, batch.meanGrade1)
            loss2 = criterion(prediction2, batch.meanGrade2)
            loss_mse = loss1 + loss2
            if args.diff:
                loss_diff = criterion(prediction1-prediction2, batch.meanGrade1 - batch.meanGrade2)
                loss = loss_mse + loss_diff
            # elif ranking:
            #     criterion2 = nn.MarginRankingLoss(
            #         margin=0.0, size_average=None, reduce=None, reduction='none')
            #     label = batch.label
            #     label_mask = (label != 0).float()
            #     label[label != 1] = -1
            #     loss3 = criterion2(prediction1, prediction2, label)
            #     loss3 = (loss3 * label_mask).sum()
            #     loss = loss_mse + loss3
            else:
                loss = loss_mse

            bsz = len(batch.id)
            epoch_loss += loss.item()

            scorer.accumulate(loss_mse.item() / 2, bsz)
            scorer2.accumulate(pred=pred_label,
                               label=batch.label,
                               diff=(batch.meanGrade1-batch.meanGrade2).abs())

        rmse = scorer.calculate(clear=True)
        acc, reward = scorer2.calculate(clear=True)

        valid_loss = epoch_loss / len(iterator)
        print(
            f'\tValid Loss: {valid_loss:.4f} | Valid RMSE: {rmse:.4f} | Valid Acc: {acc :.4f} | Valid Reward: {reward :.4f}')
    return acc


def train_loop(args, model, optimizer, train_iterator, valid_iterator):

    model = model.to(device)
    best_valid_acc = float(-0.1)

    try:
        for epoch in range(args.epochs):
            print(f'Epoch: {epoch+1:02}')
            train_loss = 0
            scorer = RMSE()
            scorer2 = Scorer2()

            if args.track:
                # track training time
                generator = tqdm(enumerate(train_iterator, 1), total=len(train_iterator))
            else:
                generator = enumerate(train_iterator, 1)
            for idx, batch in generator:
                loss, loss_mse = _train2(args, model, batch, optimizer, scorer2)
                train_loss += loss.item()
                bsz = len(batch.id)
                scorer.accumulate(loss_mse.item() / 2, bsz)

                # log every 1/3 epoch
                log_interval = len(train_iterator) // 3
                if idx % log_interval == 0:
                    rmse = scorer.calculate(clear=False)
                    acc, reward = scorer2.calculate(clear=False)
                    print(f'{idx} / {len(train_iterator)}')
                    print(
                        f'\tTrain Loss: {train_loss/idx:.4f} | Train RMSE: {rmse:.4f} | Train Acc: {acc:.4f} | Train Reward: {reward:.4f}')
                    valid_acc = evaluate2(args, model, valid_iterator, )

                    if valid_acc > best_valid_acc:
                        best_valid_acc = valid_acc
                        best_model = copy.deepcopy(model)
                        print(f"Best validation accuracy found: {best_valid_acc:.4f}\n")
                        # best_model_state = copy.deepcopy(model.state_dict())
                        # torch.save(model.state_dict(), 'task-1-diff-bilstm-model.pth')
    except KeyboardInterrupt:
        pass
    return best_model


class CBOW2(nn.Module):
    def __init__(self, embedding, pad_idx):
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

    def forward(self, batch):

        new_word = batch.new_word1.squeeze(-1)
        new_word_emb = self.embedding(new_word)
        new = self.pooler_mean(new_word_emb)

        context = batch.context1
        context_emb = self.embedding(context)
        context = self.pooler_max(context_emb, context != self.pad_idx)

        diff_emb = torch.cat(
            [new, context, (context-new).abs(), context * new], dim=-1)
        pred1 = self.classifier(diff_emb).squeeze()

        new_word = batch.new_word2.squeeze(-1)
        new_word_emb = self.embedding(new_word)
        new = self.pooler_mean(new_word_emb)

        context = batch.context2
        context_emb = self.embedding(context)
        context = self.pooler_max(context_emb, context != self.pad_idx)

        diff_emb = torch.cat(
            [new, context, (context-new).abs(), context * new], dim=-1)
        pred2 = self.classifier(diff_emb).squeeze()

        # pred = torch.cat([pred1, pred2], dim=-1)
        pred_label = torch.argmax(torch.stack(
            [pred1, pred2], dim=-1), dim=-1) + 1  # 1, 2

        return pred1, pred2, pred_label


class PretrainedTransformer2(nn.Module):
    def __init__(self, transformer, finetune=False, pad_token_id=0, sep_token_id=None):
        super().__init__()
        self.transformer = transformer
        self.pad_token_id = pad_token_id
        self.sep_token_id = sep_token_id

        d_inp = transformer.config.hidden_size

        d_cls = d_inp * 4
        self.pooler = Pooler(project=False, d_inp=d_inp, pool_type="mean")

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
            print('Finetune pretrained transformer!')
            print(f'The model has {count_parameters(self):,} trainable parameters')

        # self.classifier3 = Classifier(d_inp=(d_cls // 2), n_classes=2,
        #                               cls_type="log_reg", dropout=0.4, d_hid=256)

    def forward_sentence(self, inp, mix=True,):

        inp_mask = (inp != self.pad_token_id) & (inp != self.sep_token_id)
        sep_mask = (inp == self.sep_token_id)

        outputs = self.transformer(inp, attention_mask=inp_mask)
        last_hidden_state, pooler_output, hidden_states = outputs

        if mix:
            hidden = self.scalar_mix(hidden_states)
        else:
            hidden = last_hidden_state

        span = torch.nonzero(sep_mask, as_tuple=True)[1].view(-1, 2)
        pool_mask = torch.zeros_like(sep_mask)
        for row, (start, end) in enumerate(span):
            pool_mask[row][start+1:end].fill_(1)
        pool_mask = pool_mask.bool()
        out = self.pooler(hidden, pool_mask)
        # out2 = self.pooler(hidden, pool_mask == 0)

        return out

    def forward(self, batch):
        mix = self.scalar_mix is not None

        q21 = self.forward_sentence(batch.new1, mix)
        q31 = self.forward_sentence(batch.mask1, mix)
        m = [q21, q31, (q21-q31).abs(), q21*q31]
        pair_emb = torch.cat(m, dim=-1)
        pred1 = self.classifier(pair_emb).squeeze()

        q22 = self.forward_sentence(batch.new2, mix)
        q33 = self.forward_sentence(batch.mask2, mix)
        m = [q22, q33, (q22-q33).abs(), q22*q33]
        pair_emb = torch.cat(m, dim=-1)
        pred2 = self.classifier(pair_emb).squeeze()

        # pred3 = self.classifier3(torch.cat([q2, q22], dim=-1))  # .squeeze()
        # diff_emb2 = torch.cat([q1, q3, (q1-q3).abs(), q1*q3], dim=-1)
        # pred2 = self.classifier(diff_emb2).squeeze()

        pred_label = torch.argmax(torch.stack(
            [pred1, pred2], dim=-1), dim=-1) + 1  # 1, 2

        return pred1, pred2, pred_label  # , pred3

    def freeze_transformer(self):
        print('Freezing pretrained transformer!')
        print(f'Before, the model has {count_parameters(self)} trainable parameters')
        for name, param in self.named_parameters():
            if name.startswith('transformer'):
                param.requires_grad = False
        print(f'Now, the model has {count_parameters(self)} trainable parameters')
