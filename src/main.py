from util import *
from models import *
from trainer import *
from tasks import *
from code.score_task_1 import score_task_1
from code.score_task_2 import score_task_2


def parse_args(cl_args):
    parser = argparse.ArgumentParser(description="")
    """ train config """
    parser.add_argument("-bsz", type=int, default=32, help="batch size")
    parser.add_argument("-seed", type=int, default=1, help="random seed")
    parser.add_argument("-epochs", type=int, default=10, help="number of epochs")
    # prevent gradient norm from being too large
    parser.add_argument(
        "-grad_norm", type=float, default=5, help="gradient norm clipping"
    )

    """ model config """
    parser.add_argument(
        "-model", type=str, default="cbow", choices=["cbow", "transformer",], help=""
    )
    # chocies: 'bert-base-uncased', 'bert-large-uncased','roberta-base', 'roberta-large', 'roberta-large-mnli'
    parser.add_argument("-transformer", type=str, default="bert-base-uncased", help="")
    parser.add_argument("-finetune", type=int, default=0, choices=[0, 1], help="")
    parser.add_argument(
        "-feature",
        type=str,
        default="edit-context",
        choices=["edit-context", "edit-original", "edit"],
        help="",
    )
    """ dataset """
    parser.add_argument(
        "-task", type=str, default="task1", choices=["task1", "task2",], help=""
    )
    parser.add_argument(
        "-train_extra",
        type=int,
        default=0,
        choices=[0, 1],
        help="use funlines dataset as additional dataset if 1",
    )

    """ pipeline """
    parser.add_argument("-do_train", action="store_true", help="")
    parser.add_argument("-do_eval", action="store_true", help="")
    parser.add_argument("-do_predict", action="store_true", help="")
    """ additional options """
    parser.add_argument("-track", action="store_true", help="track training time")
    parser.add_argument("-save_model", action="store_true", help="save model state")
    parser.add_argument(
        "-tensorboard", action="store_true", help="use tensorboard for logging metrics"
    )
    """ paths """
    parser.add_argument(
        "-exp", type=str, default="exp", help="name of experiment output directory"
    )
    parser.add_argument("-train_path", type=str, default="", help="")
    parser.add_argument("-train_extra_path", type=str, default="", help="")
    parser.add_argument("-val_path", type=str, default="", help="")
    parser.add_argument("-test_without_label_path", type=str, default="", help="")
    parser.add_argument("-test_with_label_path", type=str, default="", help="")
    """ optimizer """
    parser.add_argument(
        "-optimizer",
        type=str,
        default="adam",
        choices=["adam", "pytorch_adamw", "huggingface_adamw",],
        help="",
    )
    parser.add_argument("-lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "-adam_epsilon", default=1e-8, type=float, help="epsilon for Adam optimizer."
    )
    """ scheduler """
    parser.add_argument(
        "-schedule",
        type=str,
        default="none",
        choices=["none", "reduce_on_plateau", "linear_schedule_with_warmup"],
        help="",
    )
    parser.add_argument(
        "-warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )
    args = parser.parse_args(cl_args)
    return args


args = parse_args(sys.argv[1:])

if args.task == "task1":
    default_data_dir = "../data/task-1"
    exp_dir = os.path.join("../exp1", args.exp)
    out_path = os.path.join(exp_dir, "task-1-output.csv")
    task_class = Task1
    trainer_class = Trainer1
    score_task = score_task_1

elif args.task == "task2":
    default_data_dir = "../data/task-2"
    exp_dir = os.path.join("../exp2", args.exp)
    out_path = os.path.join(exp_dir, "task-2-output.csv")
    task_class = Task2
    trainer_class = Trainer2
    score_task = score_task_2

args.exp_dir = exp_dir
args = args_default_path(args, default_data_dir)
config_path = os.path.join(exp_dir, "params.json")
model_path = os.path.join(exp_dir, "model_state.th")
log_path = os.path.join(exp_dir, "log.log")
seed_all(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
init_log()


if args.do_train:
    # clear experiment directory
    if os.path.isdir(exp_dir):
        shutil.rmtree(exp_dir)
    os.makedirs(exp_dir, exist_ok=True)

    log.getLogger().addHandler(log.FileHandler(log_path, mode="w"))
    log.info("---- Phase: train -----")
    with open(config_path, "w") as conf:
        args_dict = copy.deepcopy(vars(args))
        args_exclude = [
            "exp",
            "track",
            "save_model",
            "tensorboard",
            "do_eval",
            "do_train",
            "do_predict",
            "exp_dir",
            "train_path",
            "train_extra_path",
            "val_path",
            "test_with_label_path",
            "test_without_label_path",
        ]
        for k in args_exclude:
            del args_dict[k]

        json.dump(args_dict, conf, indent=2)

elif args.do_eval or args.do_predict:
    # assert experiment directory exists
    assert os.path.isdir(exp_dir)
    assert os.path.exists(config_path)
    assert os.path.exists(model_path)

    log.getLogger().addHandler(log.FileHandler(log_path, mode="a"))
    log.info("\n\n======================================== ")
    with open(config_path, "r") as conf:
        args_dict = json.load(conf)
        for k, v in args_dict.items():
            vars(args)[k] = v

args_json = json.dumps(vars(args), indent=2)
log.info(f"Arguments: {args_json}")
log.info(f"Device: {device}")


""" Preprocessing (Model Specific) """

if args.model == "transformer":
    # model_name = args.transformer
    tokenizer, transformer = init_transformers(args.transformer)
    TEXT = data.Field(
        tokenize=lambda x: tokenize_and_cut(x, tokenizer=tokenizer),
        batch_first=True,
        use_vocab=False,
        preprocessing=tokenizer.convert_tokens_to_ids,
        pad_token=tokenizer.pad_token_id,
        unk_token=tokenizer.unk_token_id,
        # init_token = init_token_idx,
        # eos_token = eos_token_idx,
    )
    preprocess_config = {
        "preprocess": "transformer",
        "mask_token": tokenizer.mask_token,
        "text_field": TEXT,
    }

elif args.model == "cbow":
    TEXT = data.Field(
        tokenize="spacy", sequential=True, batch_first=True, use_vocab=True
    )
    preprocess_config = {"preprocess": "cbow", "mask_token": None, "text_field": TEXT}

task = task_class()
task.read(args, preprocess_config)

""" Define Model """

if args.model == "cbow":
    TEXT, embedding = load_word_embedding(TEXT, task.train_data)
    embedding.weight.requires_grad = bool(args.finetune)
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    model = CBOW(embedding, pad_idx=PAD_IDX, feature=args.feature, mode=args.task,)


elif args.model == "transformer":
    model = PretrainedTransformer(
        transformer,
        finetune=bool(args.finetune),
        feature=args.feature,
        mode=args.task,
        pad_token_id=tokenizer.pad_token_id,
        sep_token_id=tokenizer.sep_token_id,
    )

""" Training """

if args.do_train:
    log.info(f"The model has {count_parameters(model)} trainable parameters")
    trainer = trainer_class(args, model)
    best_model = trainer.train_loop(args, task)

    val_metrics = trainer_class.evaluate(args, best_model, task.val_data)
    val_metrics = {k: np.around(float(v), 6) for k, v in val_metrics.items()}
    val_json = json.dumps(val_metrics, indent=2)
    log.info(f"\nvalidation result: {val_json}")

    # save model
    if args.save_model:
        torch.save(best_model.state_dict(), model_path)
        log.info(f"\nSave best model state to {model_path}")
    else:
        log.info("\nNOT save model state.")

""" Evaluation """

if args.do_eval:
    log.info("\n---- Phase: eval -----")
    if not args.do_train:
        model.load_state_dict(torch.load(model_path))
    else:
        model = best_model
    log.info(f"Evaluate: {args.test_with_label_path}")

    # eval
    test_metrics = trainer_class.evaluate(args, model, task.test_with_label_data)
    test_metrics = {k: np.around(float(v), 6) for k, v in test_metrics.items()}
    test_json = json.dumps(test_metrics, indent=2)
    log.info(f"\ntest result: {test_json}")

    # write prediciton for analysis
    log.info("")
    name = re.findall(".*/(.*).csv", os.path.abspath(args.test_with_label_path))[0]
    eval_out_path = os.path.join(exp_dir, f"output-{name}.csv")
    trainer_class.write_prediction(
        args,
        model,
        task.test_with_label_data,
        eval_out_path,
        mode="analysis",
        in_path=args.test_with_label_path,
    )
    # debug
    score_task(truth_loc=args.test_with_label_path, prediction_loc=eval_out_path)

""" Prediction """

if args.do_predict:
    log.info("\n---- Phase: predict -----")
    if not args.do_train:
        model.load_state_dict(torch.load(model_path))
    else:
        model = best_model
    log.info(f"Write prediction for: {args.test_without_label_path}")
    trainer_class.write_prediction(
        args, model, task.test_without_label_data, out_path, mode="minimal"
    )
