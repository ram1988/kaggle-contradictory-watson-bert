import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from BertClassifier import BertClassifier

classifier = BertClassifier()

def load_sentences(file_path):
    df = pd.read_csv(file_path)
    print(df)
    return df["premise"], df["hypothesis"], (df["label"] if "label" in df else None), df["id"]

def prepare_train_dataset():
    train_premise, train_hypothesis, train_labels, train_ids = load_sentences("../data/train.csv")
    print("the total recs-->")
    print(len(train_premise))
    encoded_train_dataset = []

    input_word_ids = []
    input_mask = []
    input_type_ids =  []

    for i, premise in enumerate(train_premise):
        hypothesis = train_hypothesis[i]
        encoded = classifier.pre_process_hyp_prem_pairs(premise, hypothesis)
        input_word_ids.append(encoded['input_word_ids'])
        print(str(i) + '----' + str(len(encoded['input_word_ids'])))
        input_mask.append(encoded['input_mask'])
        input_type_ids.append(encoded['input_type_ids'])

    # https://www.tensorflow.org/guide/ragged_tensor
    # To handle the inputs of varied lengths
    input_word_ids = tf.convert_to_tensor(input_word_ids)
    input_mask = tf.convert_to_tensor(input_mask)
    input_type_ids = tf.convert_to_tensor(input_type_ids)

    inputs = {
        'input_word_ids': input_word_ids,
        'input_mask': input_mask,
        'input_type_ids': input_type_ids}
    train_labels = np.array(train_labels)

    prepared_train_dataset = [inputs, train_labels]
    pickle.dump(prepared_train_dataset, open("prepared_train_dataset.pkl", "wb"))
    return encoded_train_dataset, train_labels

def train_and_evaluate():
    # prepare_train_dataset()
    prepared_train_dataset = pickle.load(open("prepared_train_dataset.pkl","rb"))
    train_data = prepared_train_dataset[0]
    train_labels = prepared_train_dataset[1]
    total_len = len(train_data)

    classifier.train(train_data, train_labels)

    # eval_records = train_data[total_eval_len:]
    # eval_record_labels = train_labels[total_eval_len:]
    #
    # classifier.train(train_records, train_record_labels)
    # accuracy = classifier.evaluate(eval_records, eval_record_labels)
    # print(accuracy)

def predict_outcome(text):
    encoded = classifier.pre_process_text(text)
    encoded = np.array([encoded])
    result = classifier.predict(encoded)
    result = result[0]
    result_0 = result[0]
    result_1 = result[1]
    predicted = 1 if result_1 > result_0 else 0
    return predicted

def predict_outcomes():
    test_texts, test_labels, test_id = load_sentences("../data/test.csv")
    with open('outcome.csv', 'a') as outcome_file:
        outcome_file.write("id,target")
        for i, test_text in enumerate(test_texts):
            print(test_id[i])
            predicted = predict_outcome(test_text)
            outcome_file.write(str(test_id[i]) + "," + str(predicted) + "\n")


prepare_train_dataset()
train_and_evaluate()
# premise = "These are issues that we wrestle with in practice groups of law firms, she said. "
# hypothesis = "Practice groups are not permitted to work on these issues."
# op, input_mask, input_type_ids = classifier.pre_process_hyp_prem_pairs(premise, hypothesis)
# print(op)
# print(input_mask)
# print(input_type_ids)