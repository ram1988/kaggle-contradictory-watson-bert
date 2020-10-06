import pandas as pd
import numpy as np
import tensorflow as tf
from BertClassifier import BertClassifier

classifier = BertClassifier()

def load_sentences(file_path):
    df = pd.read_csv(file_path)
    print(df)
    return df["premise"], df["hypothesis"], df["id"], (df["label"] if "label" in df else None)

def prepare_dataset(is_training):
    if is_training:
        premises, hypothesis, ids, labels = load_sentences("../data/train.csv")
    else:
        premises, hypothesis, ids, labels = load_sentences("../data/test.csv")

    print("the total recs-->")
    print(len(premises))

    input_word_ids = []
    input_mask = []
    input_type_ids =  []

    for i, premise in enumerate(premises):
        encoded = classifier.pre_process_hyp_prem_pairs(premise, hypothesis[i])
        input_word_ids.append(encoded['input_word_ids'])
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
    train_labels = np.array(labels) if labels is not None else None
    outcome = {
        'inputs': inputs,
        'labels': train_labels,
        'ids': ids
    }
    return outcome

def train_and_evaluate():
    # prepare_train_dataset()
    outcome = prepare_dataset(True)
    prepared_train_dataset = outcome['inputs']
    train_labels = outcome['labels']
    classifier.train(prepared_train_dataset, train_labels)

    # eval_records = train_data[total_eval_len:]
    # eval_record_labels = train_labels[total_eval_len:]
    #
    # classifier.train(train_records, train_record_labels)
    # accuracy = classifier.evaluate(eval_records, eval_record_labels)
    # print(accuracy)

def predict_outcomes():
    outcomes = prepare_dataset(False)
    test_inputs = outcomes['inputs']
    print(test_inputs)
    ids = outcomes['ids']
    print("test ids")
    print(len(ids))
    results = classifier.predict(test_inputs)
    print(results)
    predictions = [np.argmax(i) for i in results]
    submission = pd.DataFrame(ids, columns=['id'])
    # print(submission)
    # print(predictions)
    submission['prediction'] = predictions
    submission.to_csv("submission.csv", index=False)

prepare_dataset(True)
train_and_evaluate()
predict_outcomes()
# premise = "These are issues that we wrestle with in practice groups of law firms, she said. "
# hypothesis = "Practice groups are not permitted to work on these issues."
# op, input_mask, input_type_ids = classifier.pre_process_hyp_prem_pairs(premise, hypothesis)
# print(op)
# print(input_mask)
# print(input_type_ids)