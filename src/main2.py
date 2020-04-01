from base2 import *
from code.score_task_2 import score_task_2


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-bsz', type=int, default=32, help='batch size')
    parser.add_argument('-lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('-seed', type=int, default=1, help='random seed')
    parser.add_argument('-epochs', type=int, default=10, help='max epochs')
    parser.add_argument('-exp',  type=str, default='exp', help='experiment directory')
    parser.add_argument('-model', type=str, default='cbow',
                        choices=['cbow', 'transformer', ], help='')
    parser.add_argument('-transformer', type=str, default='bert-base-uncased', help='')
    # 'bert-base-uncased', 'bert-large-uncased','roberta-base', 'roberta-large', 'roberta-large-mnli',
    parser.add_argument('-finetune', action='store_true', help='')
    parser.add_argument('-no_split', action='store_true', help='not split validation data')
    parser.add_argument('-track', action='store_true', help='track training time')
    parser.add_argument('-save_model', action='store_true', help='save model state')
    parser.add_argument('-train_more', action='store_true', help='')
    # parser.add_argument('-project', action='store_true', help='')
    parser.add_argument('-cache', type=str, default='cache/transformers',
                        help='cache directory for pretrained transformers')
    parser.add_argument('-ranking', type=float, default=0, help='')
    parser.add_argument('-diff', type=float, default=0, help='')
    parser.add_argument('-margin', type=float, default=0, help='')
    args = parser.parse_args()
    return args


args = parse_args()
seed_all(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cache_dir = args.cache

dir = os.path.join('../exp2', args.exp)
os.makedirs(dir, exist_ok=True)
model_path = os.path.join(dir, 'task-2-model.pth')
out_path = os.path.join(dir, 'task-2-output.csv')

log_path = os.path.join(dir, 'log.log')
init_log()
log.getLogger().addHandler(log.FileHandler(log_path, mode='w'))

args_json = json.dumps(vars(args), indent=2)
print(f'Arguments: {args_json}')
print(f'Device: {device}')

train_path = "../data/task-2/train.csv"
valid_path = "../data/task-2/dev.csv"
final_valid_path = "../data/task-2/test_without_label.csv"

train_path2 = "../data/semeval-2020-task-7-extra-training-data/task-2/train_funlines.csv"
if args.train_more:
    train_path = [train_path, train_path2]
else:
    train_path = [train_path]


''' Preprocessing (Model Specific) '''

if args.model == 'cbow':
    TEXT = data.Field(tokenize='spacy', sequential=True, batch_first=True, use_vocab=True)
    preprocess = 'cbow'
    mask_token = None

elif args.model == 'transformer':
    model_name = args.transformer
    MODELS_DICT = init_transformers()
    model_class, tokenizer_class = MODELS_DICT[model_name]
    tokenizer = tokenizer_class.from_pretrained(model_name, cache_dir=cache_dir)
    TEXT = data.Field(tokenize=lambda x: tokenize_and_cut(x, tokenizer=tokenizer),
                      batch_first=True,
                      use_vocab=False,
                      preprocessing=tokenizer.convert_tokens_to_ids,
                      # init_token = init_token_idx,
                      # eos_token = eos_token_idx,
                      pad_token=tokenizer.pad_token_id,
                      unk_token=tokenizer.unk_token_id,)
    preprocess = 'transformer'
    mask_token = tokenizer.mask_token


""" Load Data """


train, valid, test, final_valid = read_csv_split(
    train_path, valid_path, final_valid_path, no_split=args.no_split)

train_data, valid_data, test_data, final_valid_data = \
    [get_dataset2(split, TEXT, preprocess=preprocess, mask_token=mask_token)
     for split in [train, valid, test, final_valid]]

print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')
print(f'Number of final_valid examples: {len(final_valid_data)}')

valid_iterator, test_iterator, final_valid_iterator = data.BucketIterator.splits(
    (valid_data, test_data, final_valid_data),
    batch_size=args.bsz,
    # the BucketIterator needs to be told what function it should use to group the data.
    sort_key=lambda x: len(x.original1),
    sort_within_batch=False,
    device=device)
train_iterator = data.BucketIterator(
    train_data,
    batch_size=args.bsz,
    sort_key=lambda x: len(x.original1),
    sort_within_batch=False,
    shuffle=True,
    device=device)
# train_iterator = data.Iterator(
#     train_data,
#     batch_size=args.bsz,
#     # sort_key=lambda x: len(x.original1),
#     # sort_within_batch=False,
#     shuffle=True,
#     device=device)


''' Define Model '''

if args.model == 'cbow':
    TEXT, embedding = load_word_embedding(TEXT, train_data)
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    embedding.weight.requires_grad = args.finetune

    model = CBOW2(embedding, PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

elif args.model == 'transformer':
    transformer = model_class.from_pretrained(
        model_name, cache_dir=cache_dir, output_hidden_states=True)
    model = PretrainedTransformer2(transformer, finetune=args.finetune, pad_token_id=tokenizer.pad_token_id,
                                   sep_token_id=tokenizer.sep_token_id,)
    optimizer = AdamW(model.parameters())


''' Training '''

print(f'The model has {count_parameters(model)} trainable parameters')
# print(model)
best_model = train_loop(args, model, optimizer, train_iterator, valid_iterator,)
#epochs=args.epochs, track=args.track, diff=args.diff, ranking=args.ranking)



''' Testing '''

print('\nFinish training.')

print('\nTest accuracy of last epoch')
print('valid')
evaluate2(args, model, valid_iterator)
print('test')
evaluate2(args, model, test_iterator)


print('\nTest accuracy of best epoch')
print('valid')
evaluate2(args, best_model, valid_iterator)
print('test')
evaluate2(args, best_model, test_iterator)


print('\nPrediction for full validation data')
full_valid_data = data.Dataset(valid_data.examples + test_data.examples, valid_data.fields)
full_valid_iterator = data.BucketIterator(
    full_valid_data,
    batch_size=args.bsz,
    sort_key=lambda x: len(x.original),
    sort_within_batch=False,
    shuffle=False,
    device=device)
evaluate2(args, best_model, full_valid_iterator)
write_prediction2(best_model, full_valid_iterator, os.path.join(dir, 'task-2-dev-output.csv'))
score_task_2(truth_loc=valid_path, prediction_loc=os.path.join(dir, 'task-2-dev-output.csv'))


print('\nPrediction for test data')
write_prediction2(best_model, final_valid_iterator, out_path)


# save model
if args.save_model:
    torch.save(best_model.state_dict(), model_path)
    print(f"Save best model state to {model_path}")
    # load
    # model.load_state_dict(torch.load(model_path))
else:
    print('\nNOT save model state.')
print('')
