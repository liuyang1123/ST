"""
Author:"""
from dataset import Dataset
import numpy as np

babi_map = {
    "1": "qa1_single-supporting-fact",
    "2": "qa2_two-supporting-facts",
    "3": "qa3_three-supporting-facts",
    "4": "qa4_two-arg-relations",
    "5": "qa5_three-arg-relations",
    "6": "qa6_yes-no-questions",
    "7": "qa7_counting",
    "8": "qa8_lists-sets",
    "9": "qa9_simple-negation",
    "10": "qa10_indefinite-knowledge",
    "11": "qa11_basic-coreference",
    "12": "qa12_conjunction",
    "13": "qa13_compound-coreference",
    "14": "qa14_time-reasoning",
    "15": "qa15_basic-deduction",
    "16": "qa16_basic-induction",
    "17": "qa17_positional-reasoning",
    "18": "qa18_size-reasoning",
    "19": "qa19_path-finding",
    "20": "qa20_agents-motivations",
    "MCTest": "MCTest",
    "19changed": "19changed",
    "joint": "all_shuffled",
    "sh1": "../shuffled/qa1_single-supporting-fact",
    "sh2": "../shuffled/qa2_two-supporting-facts",
    "sh3": "../shuffled/qa3_three-supporting-facts",
    "sh4": "../shuffled/qa4_two-arg-relations",
    "sh5": "../shuffled/qa5_three-arg-relations",
    "sh6": "../shuffled/qa6_yes-no-questions",
    "sh7": "../shuffled/qa7_counting",
    "sh8": "../shuffled/qa8_lists-sets",
    "sh9": "../shuffled/qa9_simple-negation",
    "sh10": "../shuffled/qa10_indefinite-knowledge",
    "sh11": "../shuffled/qa11_basic-coreference",
    "sh12": "../shuffled/qa12_conjunction",
    "sh13": "../shuffled/qa13_compound-coreference",
    "sh14": "../shuffled/qa14_time-reasoning",
    "sh15": "../shuffled/qa15_basic-deduction",
    "sh16": "../shuffled/qa16_basic-induction",
    "sh17": "../shuffled/qa17_positional-reasoning",
    "sh18": "../shuffled/qa18_size-reasoning",
    "sh19": "../shuffled/qa19_path-finding",
    "sh20": "../shuffled/qa20_agents-motivations",
}

class BABIDataset(Dataset):
    """
    -
    """

    def _extract(self):
        return

    def _load_glove(self, dim=50):
        word2vec = {}

        print "==> loading glove"
        with open('./datasets/dmn/data/glove/glove.6B.' + str(dim) + 'd.txt') as f:
            for line in f:
                l = line.split()
                word2vec[l[0]] = map(float, l[1:])

        print "==> glove is loaded"

        return word2vec


    def _process(self):
        """
        After the files are downloaded, create a representation of the dataset.
        """
        self.word2vec = self._load_glove()
        tr_task = self.config.get("training_task", "1")
        ts_task = self.config.get("testing_task", tr_task)
        use_10k = self.config.get("use_10k", False)

        babi_name = babi_map[tr_task]
        babi_test_name = babi_map[ts_task]

        folder = 'en-10k' if use_10k else 'en'
        train_raw = self._init_babi(self.workdir + folder + '/%s_train.txt' % babi_name)
        test_raw = self._init_babi(self.workdir + folder + '/%s_test.txt' % babi_test_name)

        self.vocab = {}
        self.ivocab = {}

        tr_c, tr_q, tr_a, tr_g, tr_imask, tr_c_size, tr_q_size, tr_max_s, tr_max_q = self._process_dataset(train_raw)
        ts_c, ts_q, ts_a, ts_g, ts_imask, ts_c_size, ts_q_size, ts_max_s, ts_max_q = self._process_dataset(test_raw)

        self.train = [[np.array(tr_c), np.array(tr_q)], np.array(tr_a)]
        self.train_seq_length = [tr_c_size, tr_q_size]
        self.train_imask = tr_imask
        self.test = [[np.array(ts_c), np.array(ts_q)], np.array(ts_a)]
        self.test_seq_length = [ts_c_size, ts_q_size]
        self.test_imask = ts_imask
        self.max_input_size = max(tr_max_s, ts_max_s)
        self.max_question_size = max(tr_max_q, ts_max_q)
        # TODO Set max lengths for training samples

        self._num_examples_train = self.train[0][0].shape[0]
        self._num_examples_test = self.test[0][0].shape[0]
        self._num_examples_validation = 0

        self.vocab_size = len(self.vocab)

    def _filter_by_range(self, obj, start, end):
        """
        -
        """
        return np.array(obj)[:,start:end]

    def _shuffle(self, obj, perm):
        """
        -
        """
        return np.array(obj)[:,perm]

    def _init_babi(self, fname):
        tasks = []
        task = None
        for i, line in enumerate(open(fname)):
            _id = int(line[0:line.find(' ')])
            if _id == 1:
                task = {"C": "", "Q": "", "A": "", "G": ""}

            line = line.strip()
            line = line.replace('.', ' . ')
            line = line[line.find(' ') + 1:]
            if line.find('?') == -1:
                task["C"] += line
            else:
                idx = line.find('?')
                tmp = line[idx + 1:].split('\t')
                task["Q"] = line[:idx]
                task["A"] = tmp[1].strip()
                task["G"] = tmp[2].strip()
                tasks.append(task.copy())

        return tasks

    def _process_dataset(self, dataset):
        # Each element of dataset is dictionary
        inputs = []
        questions = []
        answers = []
        gate_values = []
        input_masks = []

        input_seq_length = question_seq_length = []

        max_sentence = max_question = 0

        for x in dataset:
            inp = x["C"].lower().split(' ') # Split sentence in word
            inp = [w for w in inp if len(w) > 0]
            q = x["Q"].lower().split(' ')
            q = [w for w in q if len(w) > 0]

            inp_vector = [self._process_word(word = w,
                                        word2vec = self.word2vec,
                                        vocab = self.vocab,
                                        ivocab = self.ivocab,
                                        word_vector_size = self.config.get("word_vector_length"),
                                        to_return = "word2vec") for w in inp]

            q_vector = [self._process_word(word = w,
                                        word2vec = self.word2vec,
                                        vocab = self.vocab,
                                        ivocab = self.ivocab,
                                        word_vector_size = self.config.get("word_vector_length"),
                                        to_return = "word2vec") for w in q]

            inputs.append(np.vstack(inp_vector).astype(np.float32))
            questions.append(np.vstack(q_vector).astype(np.float32))
            a = self._process_word(word = x["A"],
                             word2vec = self.word2vec,
                             vocab = self.vocab,
                             ivocab = self.ivocab,
                             word_vector_size = self.config.get("word_vector_length"),
                             to_return = "index")
            answers.append(a)

            max_sentence = max(max_sentence, len(inp_vector))
            max_question = max(max_question, len(q_vector))

            input_seq_length.append(len(inp_vector))
            question_seq_length.append(len(q_vector))

            #gate_values.append(int(x["G"]))
            # NOTE: here we assume the answer is one word!
            mode = self.config.get("input_mask_mode", 'sentence')
            if mode == 'word':
                input_masks.append(np.array([index for index, w in enumerate(inp)], dtype=np.int32))
            elif mode == 'sentence':
                input_masks.append(np.array([index for index, w in enumerate(inp) if w == '.'], dtype=np.int32))
            else:
                raise Exception("invalid input_mask_mode")

        return inputs, questions, answers, gate_values, input_masks, input_seq_length, question_seq_length, max_sentence, max_question

    def _create_vector(self, word, word2vec, word_vector_size, silent=False):
        # if the word is missing from Glove, create some fake vector and store in glove!
        vector = np.random.uniform(0.0,1.0,(word_vector_size,))
        word2vec[word] = vector
        if (not silent):
            print "utils.py::create_vector => %s is missing" % word
        return vector


    def _process_word(self, word, word2vec, vocab, ivocab, word_vector_size, to_return="word2vec", silent=False):
        if not word in word2vec:
            self._create_vector(word, word2vec, word_vector_size, silent)
        if not word in vocab:
            next_index = len(vocab)
            vocab[word] = next_index
            ivocab[next_index] = word

        if to_return == "word2vec":
            return word2vec[word]
        elif to_return == "index":
            return vocab[word]
        elif to_return == "onehot":
            raise Exception("to_return = 'onehot' is not implemented yet")


    def next_batch(self, batch_size, dataset="train"):
        """
        Return the next `batch_size` examples from this data set.
        """
        if dataset == "train":
            start = self._index_in_epoch_train
            self._index_in_epoch_train += batch_size
            if self._index_in_epoch_train > self._num_examples_train:
                # Finished epoch
                self._epochs_completed_train += 1
                # Shuffle the data
                perm = np.arange(self._num_examples_train)
                np.random.shuffle(perm)
                self.train[0] = self._shuffle(self.train[0], perm)
                self.train[1] = self.train[1][perm]
                # Start next epoch
                start = 0
                self._index_in_epoch_train = batch_size
                assert batch_size <= self._num_examples_train
            end = self._index_in_epoch_train
            return self._filter_by_range(self.train[0], start, end), self.train[1][start:end], self.train_seq_length[0][start:end], self.train_seq_length[1][start:end], self.train_imask[start:end]
        elif dataset == "test":
            start = self._index_in_epoch_test
            self._index_in_epoch_test += batch_size
            if self._index_in_epoch_test > self._num_examples_test:
                # Finished epoch
                self._epochs_completed_test += 1
                # Shuffle the data
                perm = np.arange(self._num_examples_test)
                np.random.shuffle(perm)
                self.test[0] = self._shuffle(self.test[0], perm)
                self.test[1] = self.test[1][perm]
                # Start next epoch
                start = 0
                self._index_in_epoch_test = batch_size
                assert batch_size <= self._num_examples_test
            end = self._index_in_epoch_test
            return self._filter_by_range(self.test[0], start, end), self.test[1][start:end], self.test_seq_length[0][start:end], self.test_seq_length[1][start:end], self.test_imask[start:end]

        raise ValueError(
            'Invalid dataset, options are: train, test, validation')
