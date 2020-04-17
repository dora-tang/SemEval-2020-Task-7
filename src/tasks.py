from util import *


class Task(object):
    def __init__(self):
        self.train_data = None
        self.val_data = None
        self.test_with_label = None
        self.test_without_label = None

    def read(self, args, preprocess_config):

        if args.train_extra:
            train_path_list = [args.train_path, args.train_extra_path]
        else:
            train_path_list = [args.train_path]
        val_path = test_with_label_path = test_without_label_path = None
        if args.do_train:
            val_path = args.val_path
        if args.do_eval:
            test_with_label_path = args.test_with_label_path
        if args.do_predict:
            test_without_label_path = args.test_without_label_path

        task = self._read(
            args,
            train_path=train_path_list,
            val_path=val_path,
            test_without_label_path=test_without_label_path,
            test_with_label_path=test_with_label_path,
            preprocess_config=preprocess_config,
        )
        return task

    def _read(
        self,
        args,
        train_path=None,
        val_path=None,
        test_with_label_path=None,
        test_without_label_path=None,
        preprocess_config=None,
    ):
        task = {}
        log.info("")
        for split_name, path in [
            ("train_data", "train_path"),
            ("val_data", "val_path"),
            ("test_with_label_data", "test_with_label_path"),
            ("test_without_label_data", "test_without_label_path"),
        ]:
            path = eval(path)
            if path is not None:
                # log.info(f"read {split_name} from {path}")
                if split_name == "train_data":
                    split = pd.concat([pd.read_csv(i) for i in path])
                else:
                    split = pd.read_csv(path)

                task[f"{split_name}"] = self.get_dataset(
                    csv_data=split, preprocess_config=preprocess_config
                )

        for split_name in task.keys():
            length = len(task[f"{split_name}"])
            log.info(f"Number of examples in {split_name}: {length}")

        self.__dict__.update(task)
        return task

    def get_dataset(self, csv_data, preprocess_config):
        # id
        raw_field = data.RawField()
        field_id = [("id", raw_field)]
        cols_id = [csv_data["id"]]

        # label
        cols_label, field_label = self.get_label(csv_data)

        # text
        cols_text, field_text = self.get_text(csv_data, preprocess_config)

        # formulate dataset
        field = field_id + field_text + field_label
        cols = cols_id + cols_text + cols_label
        total = len(csv_data["id"])
        examples = []
        for i in tqdm(zip(*cols), total=total):
            example = data.Example.fromlist(i, field)
            examples.append(example)

        return data.Dataset(examples, field)

    def get_label(csv_data):
        raise NotImplementedError

    def get_text(self, csv_data, preprocess_config):
        raise NotImplementedError


class Task1(Task):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_label(csv_data):
        # label
        label_field = data.Field(sequential=False, use_vocab=False, dtype=torch.float)
        if "meanGrade" in csv_data.columns:
            # train / dev has labels
            field_label = [("meanGrade", label_field)]
            cols_label = [csv_data["meanGrade"]]
        else:
            # test data has no labels
            field_label = []
            cols_label = []

        return cols_label, field_label

    def get_text(self, csv_data, preprocess_config):
        # preprocess="cbow", mask_token=None, text_field,
        text_field = preprocess_config["text_field"]
        mask_token = preprocess_config["mask_token"]
        preprocess = preprocess_config["preprocess"]

        if preprocess == "cbow":
            cols_text, field_text = self.preprocess_cbow(csv_data, text_field)
        elif preprocess == "transformer":
            assert mask_token is not None
            cols_text, field_text = self.preprocess_transformer(
                csv_data, mask_token, text_field
            )

        return cols_text, field_text

    @staticmethod
    def preprocess_cbow(csv_data, text_field):
        original = csv_data["original"]
        old_word_list = []
        context_list = []
        for i in original:
            old_word = re.findall("<(.*)/>", i)[0]
            l = re.findall("(.*)<.*/>(.*)", i)[0]
            l = [j.strip() for j in l]
            context = " ".join(l).strip()
            old_word_list.append(old_word)
            context_list.append(context)
        csv_data["old_word"] = old_word_list
        csv_data["new_word"] = csv_data["edit"]
        csv_data["context"] = context_list

        field_text = [
            ("original", text_field),
            ("new_word", text_field),
            ("old_word", text_field),
            ("context", text_field),
        ]
        cols_text = [
            csv_data["original"],
            csv_data["new_word"],
            csv_data["old_word"],
            csv_data["context"],
        ]

        return cols_text, field_text

    @staticmethod
    def preprocess_transformer(csv_data, mask_token, text_field):
        new_list = []
        original = csv_data["original"]
        edit = csv_data["edit"]
        mask_list = []
        for original_sent, edit_word in zip(original, edit):
            new_sent = re.sub("<.*/>", f"<{edit_word}/>", original_sent)
            new_list.append(new_sent)
            mask_sent = re.sub("<.*/>", f"<{mask_token}/>", original_sent)
            mask_list.append(mask_sent)
        csv_data["new"] = new_list
        csv_data["mask"] = mask_list

        field_text = [
            ("original", text_field),
            ("new", text_field),
            ("mask", text_field),
        ]
        cols_text = [
            csv_data["original"],
            csv_data["new"],
            csv_data["mask"],
        ]

        return cols_text, field_text


class Task2(Task):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_label(csv_data):
        label_field = data.Field(sequential=False, use_vocab=False, dtype=torch.float)

        if "label" in csv_data.columns:
            field_label = [
                ("meanGrade1", label_field),
                ("meanGrade2", label_field),
                ("label", label_field),
            ]
            cols_label = [
                csv_data["meanGrade1"],
                csv_data["meanGrade2"],
                csv_data["label"],
            ]
        else:
            # test data has no labels
            field_label = []
            cols_label = []
        return cols_label, field_label

    def get_text(self, csv_data, preprocess_config):
        # preprocess="cbow", mask_token=None, text_field,
        text_field = preprocess_config["text_field"]
        mask_token = preprocess_config["mask_token"]
        preprocess = preprocess_config["preprocess"]

        if preprocess == "cbow":
            cols_text, field_text = self.preprocess_cbow(csv_data, text_field)
        elif preprocess == "transformer":
            assert mask_token is not None
            cols_text, field_text = self.preprocess_transformer(
                csv_data, mask_token, text_field
            )

        return cols_text, field_text

    @staticmethod
    def preprocess_cbow(csv_data, text_field):
        # log.info("preprocess for cbow")
        for j in ["1", "2"]:
            original = csv_data[f"original{j}"]
            old_word_list = []
            context_list = []
            for i in original:
                old_word = re.findall("<(.*)/>", i)[0]
                l = re.findall("(.*)<.*/>(.*)", i)[0]
                l = [j.strip() for j in l]
                context = " ".join(l).strip()
                old_word_list.append(old_word)
                context_list.append(context)

            csv_data[f"old_word{j}"] = old_word_list
            csv_data[f"new_word{j}"] = csv_data[f"edit{j}"]
            csv_data[f"context{j}"] = context_list

        field_text = [
            ("original1", text_field),
            ("old_word1", text_field),
            ("new_word1", text_field),
            ("context1", text_field),
            ("original2", text_field),
            ("old_word2", text_field),
            ("new_word2", text_field),
            ("context2", text_field),
        ]
        cols_text = [
            csv_data["original1"],
            csv_data["old_word1"],
            csv_data["new_word1"],
            csv_data["context1"],
            csv_data["original2"],
            csv_data["old_word2"],
            csv_data["new_word2"],
            csv_data["context2"],
        ]

        return cols_text, field_text

    @staticmethod
    def preprocess_transformer(csv_data, mask_token, text_field):
        # log.info("preprocess for transformer")
        for j in ["1", "2"]:
            new_list = []
            original = csv_data[f"original{j}"]
            edit = csv_data[f"edit{j}"]
            mask_list = []
            for original_sent, edit_word in zip(original, edit):
                new_sent = re.sub("<.*/>", f"<{edit_word}/>", original_sent)
                new_list.append(new_sent)
                mask_sent = re.sub("<.*/>", f"<{mask_token}/>", original_sent)
                mask_list.append(mask_sent)
            csv_data[f"new{j}"] = new_list
            csv_data[f"mask{j}"] = mask_list

        field_text = [
            ("original1", text_field),
            ("new1", text_field),
            ("mask1", text_field),
            ("original2", text_field),
            ("new2", text_field),
            ("mask2", text_field),
        ]
        cols_text = [
            csv_data["original1"],
            csv_data["new1"],
            csv_data["mask1"],
            csv_data["original2"],
            csv_data["new2"],
            csv_data["mask2"],
        ]

        return cols_text, field_text
