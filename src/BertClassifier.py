import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from tensorflow import keras

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except ValueError:
    strategy = tf.distribute.get_strategy() # for CPU and single GPU
    print('Number of replicas:', strategy.num_replicas_in_sync)

#https://huggingface.co/transformers/glossary.html#attention-mask
class BertClassifier:

    MODEL_FILE_PATH = './model/bert_model.pkl'
    EPOCHS = 10
    MAX_LEN = 300

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.model = TFBertModel.from_pretrained('bert-base-multilingual-cased')
        self.build_model()

    def pre_process_hyp_prem_pairs(self, premise, hypothesis):
        premise = "[CLS]" + premise + "[SEP]"
        hypothesis = hypothesis + "[SEP]"
        premise_tokens = self.tokenizer.tokenize(premise)
        hypothesis_tokens = self.tokenizer.tokenize(hypothesis)
        hypothesis_ids = self.tokenizer.convert_tokens_to_ids(hypothesis_tokens)
        premise_ids = self.tokenizer.convert_tokens_to_ids(premise_tokens)
        padding_length = self.MAX_LEN - len(hypothesis_tokens) - len(premise_tokens)
        padding_tokens = []
        for i in range(0, padding_length):
            padding_tokens.append("[PAD]")
        padding_ids = self.tokenizer.convert_tokens_to_ids(padding_tokens)

        type_s1 = tf.zeros_like(hypothesis_ids)
        type_s2 = tf.ones_like(premise_ids)
        type_pads = tf.ones_like(padding_ids)

        main_input_word_ids = tf.concat([hypothesis_ids, premise_ids], 0)
        main_input_mask = tf.ones_like(main_input_word_ids)
        pad_input_mask = tf.zeros_like(padding_ids)

        input_word_ids = tf.concat([hypothesis_ids, premise_ids, padding_ids], 0)
        input_type_ids = tf.concat([type_s1, type_s2, type_pads], 0)
        input_mask = tf.concat([main_input_mask, pad_input_mask], 0)
        print(len(main_input_word_ids))
        # encoded_dict = self.tokenizer(premise, hypothesis, add_special_tokens=True, padding=True, max_length=self.MAX_LEN, pad_to_max_length=True)
        # # encoded_dict = self.tokenizer([premise, hypothesis],  padding=True)x
        # input_ids = encoded_dict["input_ids"]
        # attention_mask = encoded_dict["attention_mask"]
        # token_type_ids = encoded_dict["token_type_ids"]
        # # input_ids = tf.convert_to_tensor(input_ids)
        # # attention_mask = tf.convert_to_tensor(attention_mask)
        # # token_type_ids = tf.convert_to_tensor(token_type_ids)
        inputs = {
                    'input_word_ids': input_word_ids,
                    'input_mask': input_mask,
                    'input_type_ids': input_type_ids
                 }
        return inputs

    def build_model(self):
        input_word_ids = tf.keras.Input(shape=(self.MAX_LEN,), dtype=tf.int32, name="input_word_ids")
        input_mask = tf.keras.Input(shape=(self.MAX_LEN,), dtype=tf.int32, name="input_mask")
        input_type_ids = tf.keras.Input(shape=(self.MAX_LEN,), dtype=tf.int32, name="input_type_ids")

        embedding = self.model([input_word_ids, input_mask, input_type_ids])[0]
        output = keras.Sequential([
                        keras.layers.Dropout(0.7),
                        keras.layers.Dense(3, activation='softmax')
                    ])(embedding[:, 0, :])
        self.nn_model = keras.Model(inputs=[input_word_ids, input_mask, input_type_ids], outputs=output)
        self.nn_model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                              metrics=['accuracy'])

    def train(self, inputs, labels):
        self.nn_model.fit(inputs, labels, epochs=self.EPOCHS, batch_size=64)
        keras.models.save_model(self.nn_model, self.MODEL_FILE_PATH)

    def evaluate(self, inputs, labels):
        prepared_model = keras.models.load_model(self.MODEL_FILE_PATH)
        if prepared_model:
            self.nn_model = prepared_model
        test_loss, test_acc = self.nn_model.evaluate(inputs, labels, verbose=2)
        return test_acc

    def predict(self, inputs):
        prepared_model = keras.models.load_model(self.MODEL_FILE_PATH)
        if prepared_model:
            self.nn_model = prepared_model
        predictions = self.nn_model.predict(inputs)
        return predictions