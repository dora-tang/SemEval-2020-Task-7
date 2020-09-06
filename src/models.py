from util import *

""" simple modules """


class Classifier(nn.Module):
    def __init__(self, d_inp, n_classes, cls_type="log_reg", dropout=0.2, d_hid=512):
        """
        cls_type: log_reg, cls
        """
        super().__init__()

        # logistic regression
        if cls_type == "log_reg":
            classifier = nn.Linear(d_inp, n_classes)
        # mlp
        elif cls_type == "mlp":
            classifier = nn.Sequential(
                nn.Linear(d_inp, d_hid),
                nn.Tanh(),
                nn.LayerNorm(d_hid),
                nn.Dropout(dropout),
                nn.Linear(d_hid, n_classes),
            )
        self.classifier = classifier

    def forward(self, seq_emb):
        logits = self.classifier(seq_emb)
        return logits


class Pooler(nn.Module):
    """ Do pooling, possibly with a projection beforehand """

    def __init__(self, project=True, d_inp=512, d_proj=512, pool_type="max"):
        """
        pool_type: max, mean, attn
        """
        super(Pooler, self).__init__()
        self.project = nn.Linear(d_inp, d_proj) if project else lambda x: x
        self.pool_type = pool_type

        if self.pool_type == "attn":
            d_in = d_proj if project else d_inp
            self.attn = nn.Linear(d_in, 1, bias=False)

    def forward(self, sequence, mask=None):
        """
        sequence: (bsz, T, d_inp)
        mask: nopad_mask (bsz, T) or (bsz, T, 1)
        """

        # sequence is (bsz, d_inp), no need to pool
        if len(sequence.shape) == 2:
            return sequence

        # no pad in sequence
        if mask is None:
            mask = torch.ones(sequence.shape[:2], device=device)

        if len(mask.size()) < 3:
            mask = mask.unsqueeze(dim=-1)  # (bsz, T, 1)
        pad_mask = mask == 0
        proj_seq = self.project(sequence)  # (bsz, T, d_proj) or (bsz, T, d_inp)

        if self.pool_type == "max":
            proj_seq = proj_seq.masked_fill(pad_mask, -float("inf"))
            seq_emb = proj_seq.max(dim=1)[0]

        elif self.pool_type == "mean":
            proj_seq = proj_seq.masked_fill(pad_mask, 0)
            seq_emb = proj_seq.sum(dim=1) / mask.sum(dim=1).float()

        # elif self.pool_type == "attn":
        #     # (bsz, T)
        #     # dot product attention
        #     scores = self.attn(proj_seq).squeeze()
        #     pad_mask = (mask == 0).squeeze()
        #     scores = scores.masked_fill(pad_mask, -1e9)
        #     attn_weights = torch.softmax(scores, dim=1)
        #     # (bsz, D) = (bsz, 1, T) * (bsz, T, D)
        #     seq_emb = torch.matmul(attn_weights.unsqueeze(1), proj_seq).squeeze()

        return seq_emb


def get_span_mask(
    span: torch.Tensor,
    sent_len: int,
    mode: str = "range",
    include_start: bool = True,
    include_end: bool = True,
) -> torch.Tensor:
    """
    Args:
        span: tensor (bsz * 2), last dimension is [start_index, end_index], index range in [0, sent_len-1]
        sent_len: int
        mode: 'end_point' or 'range'

    Returns:
        a boolean mask tensor, where index in span is True

    Examples:
        example 1:
            span = [[0,3],[1,4]], sent_len=5, include_start=False, include_end=False, mode='range'
            return span_mask = [[False,True,True,False,False],[False,False,True,True,False]]
        example 2:
            span = [[0,3],[1,4]], sent_len=5, include_start=True, include_end=True, mode='range'
            return span_mask = [[True,True,True,True,False],[False,True,True,True,True]]
        example 3:
            span = [[0,3],[1,4]], sent_len=5, include_start=False, include_end=False, mode='end_point'
            return span_mask = [[False,True,True,False,False],[False,False,True,True,False]]
        example 4:
            span = [[0,3],[1,4]], sent_len=5, include_start=True, include_end=True, mode='end_point'
            return span_mask = [[True,False,False,True,False],[False,True,False,False,True]]
    """

    bsz = span.shape[0]
    index_tensor = (
        torch.tensor(list(range(sent_len)), device=device)
        .unsqueeze(0)
        .expand(bsz, sent_len)
    )
    start_index, end_index = span.split(1, dim=-1)

    if not include_start:
        start_index = start_index + 1
    if not include_end:
        end_index = end_index - 1

    if mode == "range":
        start_mask = (index_tensor - start_index) >= 0
        end_mask = (index_tensor - end_index) <= 0
        span_mask = start_mask & end_mask
    elif mode == "end_point":
        start_mask = (index_tensor - start_index) == 0
        end_mask = (index_tensor - end_index) == 0
        span_mask = start_mask | end_mask

    return span_mask


""" core modules """


class CBOW(nn.Module):
    def __init__(self, embedding, pad_idx, feature="edit-context", mode="task1"):
        """
        feature: edit-context, edit-original
        mode: task1, task2
        """
        super().__init__()
        # self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.embedding = embedding
        self.pad_idx = pad_idx

        embedding_dim = embedding.weight.shape[-1]
        hdim = embedding_dim * 4

        self.classifier = Classifier(
            d_inp=hdim, n_classes=1, cls_type="mlp", dropout=0.4, d_hid=256
        )
        self.pooler_max = Pooler(project=False, pool_type="max")
        self.pooler_mean = Pooler(project=False, pool_type="mean")

        self.feature = feature
        self.mode = mode

    def words2vec(self, words, pooler):
        if len(words.shape) == 2:
            words = words.squeeze(-1)
        emb = self.embedding(words)
        pooled_emb = pooler(emb)
        return pooled_emb

    def predict_edit_score(self, new_word, old_word=None, context=None):
        # mean pooling
        q1 = self.words2vec(new_word, self.pooler_mean)
        # max pooling
        if self.feature == "edit-context":
            q2 = self.words2vec(context, self.pooler_max)
        # mean pooling
        elif self.feature == "edit-original":
            q2 = self.words2vec(old_word, self.pooler_mean)
        m = [q1, q2, (q1 - q2).abs(), q1 * q2]
        pair_emb = torch.cat(m, dim=-1)
        score = self.classifier(pair_emb).squeeze()
        return score

    def forward_task1(self, batch):
        score = self.predict_edit_score(batch.new_word, batch.old_word, batch.context)

        out = {}
        out["pred_score"] = score
        if hasattr(batch, "meanGrade"):
            criterion = nn.MSELoss(reduction="sum")
            out["loss"] = criterion(score, batch.meanGrade)
        return out

    def compare_edit_score(
        self, new_word1, old_word1, context1, new_word2, old_word2, context2
    ):
        score1 = self.predict_edit_score(new_word1, old_word1, context1)
        score2 = self.predict_edit_score(new_word2, old_word2, context2)
        # label is 1 or 2
        label = torch.argmax(torch.stack([score1, score2], dim=-1), dim=-1) + 1
        return score1, score2, label

    def forward_task2(self, batch):
        score1, score2, label = self.compare_edit_score(
            batch.new_word1,
            batch.old_word1,
            batch.context1,
            batch.new_word2,
            batch.old_word2,
            batch.context2,
        )

        out = {"pred_score1": score1, "pred_score2": score2, "pred_label": label}

        # if "label" in batch and "meanGrade1" in batch and "meanGrade2" in batch:
        if hasattr(batch, "label"):
            criterion = nn.MSELoss(reduction="sum")
            loss1 = criterion(score1, batch.meanGrade1)
            loss2 = criterion(score2, batch.meanGrade2)
            loss = loss1 + loss2
            out["loss"] = loss
        return out

    def forward(self, batch):
        if self.mode == "task1":
            return self.forward_task1(batch)
        elif self.mode == "task2":
            return self.forward_task2(batch)


class PretrainedTransformer(nn.Module):
    def __init__(
        self,
        transformer,
        finetune=False,
        feature="edit-context",
        mode="task1",
        pad_token_id=0,
        sep_token_id=None,
    ):
        """
        feature: edit-context, edit-original, edit
        mode: task1, task2
        """
        super().__init__()
        self.transformer = transformer
        self.pad_token_id = pad_token_id
        self.sep_token_id = sep_token_id

        self.feature = feature
        self.mode = mode

        d_inp = transformer.config.hidden_size
        if feature in ["edit-context", "edit-original"]:
            d_cls = d_inp * 4
        elif feature == "edit":
            d_cls = d_inp
        self.pooler = Pooler(project=False, d_inp=d_inp, pool_type="mean")

        if not finetune:
            self.scalar_mix = ScalarMix(
                transformer.config.num_hidden_layers + 1, do_layer_norm=False
            )
            self.classifier = Classifier(
                d_inp=d_cls, n_classes=1, cls_type="mlp", dropout=0.4, d_hid=256
            )
            # self.classifier = nn.Linear(d_cls, 1)
            self.freeze_transformer()
        else:
            self.scalar_mix = None
            self.classifier = nn.Linear(d_cls, 1)
            log.info("Finetune pretrained transformer!")
            log.info(f"The model has {count_parameters(self):,} trainable parameters")

    def freeze_transformer(self):
        log.info("Freezing pretrained transformer!")
        log.info(
            f"Before, the model has {count_parameters(self):,} trainable parameters"
        )
        for name, param in self.named_parameters():
            if name.startswith("transformer"):
                param.requires_grad = False
        log.info(f"Now, the model has {count_parameters(self):,} trainable parameters")

    def sent2vec(self, inp):
        inp_mask = (inp != self.pad_token_id) & (inp != self.sep_token_id)
        sep_mask = inp == self.sep_token_id

        outputs = self.transformer(inp, attention_mask=inp_mask)
        last_hidden_state, pooler_output, hidden_states = outputs

        if self.scalar_mix is not None:
            hidden = self.scalar_mix(hidden_states)
        else:
            hidden = last_hidden_state

        span = torch.nonzero(input=sep_mask, as_tuple=True)[1].view(-1, 2)
        span_mask = get_span_mask(
            span=span, sent_len=inp.shape[-1], include_start=False, include_end=False
        )
        out = self.pooler(hidden, span_mask)

        return out

    def predict_edit_score(self, edit_sentence, auxilary_sentence):
        q1 = self.sent2vec(edit_sentence)
        if auxilary_sentence is None:
            emb = q1
        else:
            q2 = self.sent2vec(auxilary_sentence)
            m = [q1, q2, (q1 - q2).abs(), q1 * q2]
            emb = torch.cat(m, dim=-1)
        score = self.classifier(emb).squeeze()
        # print(score)
        return score

    def forward_task1(self, batch):
        if self.feature == "edit-original":
            auxilary_sentence = batch.original
        elif self.feature == "edit-context":
            auxilary_sentence = batch.mask
        elif self.feature == "edit":
            auxilary_sentence = None
        edit_sentence = batch.new
        score = self.predict_edit_score(edit_sentence, auxilary_sentence)

        out = {}
        out["pred_score"] = score

        # if meanGrade1" in batch
        if hasattr(batch, "meanGrade"):
            criterion = nn.MSELoss(reduction="sum")
            out["loss"] = criterion(score, batch.meanGrade)
        return out

    def compare_edit_pair(
        self, edit_sentence1, edit_sentence2, auxilary_sentence1, auxilary_sentence2
    ):
        score1 = self.predict_edit_score(edit_sentence1, auxilary_sentence1)
        score2 = self.predict_edit_score(edit_sentence2, auxilary_sentence2)
        # label is 1 or 2
        label = torch.argmax(torch.stack([score1, score2], dim=-1), dim=-1) + 1
        return score1, score2, label

    def forward_task2(self, batch):
        if self.feature == "edit-original":
            auxilary_sentence1, auxilary_sentence2 = batch.original1, batch.original2
        elif self.feature == "edit-context":
            auxilary_sentence1, auxilary_sentence2 = batch.mask1, batch.mask2
        elif self.feature == "edit":
            auxilary_sentence1, auxilary_sentence2 = None, None
        edit_sentence1, edit_sentence2 = batch.new1, batch.new2
        score1, score2, label = self.compare_edit_pair(
            edit_sentence1, edit_sentence2, auxilary_sentence1, auxilary_sentence2
        )

        out = {"pred_score1": score1, "pred_score2": score2, "pred_label": label}

        # if "label" in batch and "meanGrade1" in batch and "meanGrade2" in batch
        if hasattr(batch, "label"):
            criterion = nn.MSELoss(reduction="sum")
            loss1 = criterion(score1, batch.meanGrade1)
            loss2 = criterion(score2, batch.meanGrade2)
            loss = loss1 + loss2
            out["loss"] = loss
        return out

    def forward(self, batch):
        if self.mode == "task1":
            return self.forward_task1(batch)
        elif self.mode == "task2":
            return self.forward_task2(batch)
