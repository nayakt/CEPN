#!/usr/bin/env python
# coding=utf-8
""" task2_evaluate.py - Scoring program for Fincausal 2020 Task 2

    usage: task2_evaluate.py [-h] {from-folder,from-file} ...

    positional arguments:
      {from-folder,from-file}
                            Use from-file for basic mode or from-folder for
                            Codalab compatible mode

Usage 1: Folder mode

    usage: task2_evaluate.py from-folder [-h] input output

    Codalab mode with input and output folders

    positional arguments:
      input       input folder with ref (reference) and res (result) sub folders
      output      output folder where score.txt is written

    optional arguments:
      -h, --help  show this help message and exit
    task2_evaluate input output

    input, output folders must follow the Codalab competition convention for scoring bundle
    e.g.
        ├───input
        │   ├───ref
        │   └───res
        └───output

Usage 2: File mode

    usage: task2_evaluate.py from-file [-h] [--ref_file REF_FILE] pred_file [score_file]

    Basic mode with path to input and output files

    positional arguments:
      ref_file    reference file (default: ../../data/fnp2020-fincausal-task2.csv)
      pred_file   prediction file to evaluate
      score_file  path to output score file (or stdout if not provided)

    optional arguments:
      -h, --help  show this help message and exit
"""
import argparse
import logging
import os
import unittest
import sys
import json

from collections import namedtuple

import nltk

from sklearn import metrics


def build_token_index(text):
    """
    build a dictionary of all tokenized items from text with their respective positions in the text.
    E.g. "this is a basic example of a basic method" returns
     {'this': [0], 'is': [1], 'a': [2, 6], 'basic': [3, 7], 'example': [4], 'of': [5], 'method': [8]}
    :param text: reference text to index
    :return: dict() of text token, each token with their respective position(s) in the text
    """
    tokens = nltk.word_tokenize(text)
    token_index = {}
    for position, token in enumerate(tokens):
        if token in token_index:
            token_index[token].append(position)
        else:
            token_index[token] = [position]
    return tokens, token_index


def get_tokens_sequence(text, token_index):
    tokens = nltk.word_tokenize(text)
    # build list of possible position for each token
    # positions = [word_index[word] for word in words]
    positions = []
    for token in tokens:
        if token in token_index:
            positions.append(token_index[token])
            continue
        # Special case when '.' is not tokenized properly
        alt_token = ''.join([token, '.'])
        if alt_token in token_index:
            logging.debug(f'tokenize fix ".": {alt_token}')
            positions.append(token_index[alt_token])
            # TODO: discard the next token if == '.' ?
            continue
        # Special case when/if ',' is not tokenized properly - TBC
        # alt_token = ''.join([token, ','])
        # if alt_token in token_index:
        #     logging.debug(f'tokenize fix ",": {alt_token}')
        #     positions.append(token_index[alt_token])
        #     continue
        else:
            logging.warning(f'get_tokens_sequence "{token}" discarded')
    # No matching ? stop here
    if len(positions) == 0:
        return positions
    # recursively process the list of token positions to return combinations of consecutive tokens
    seqs = _get_sequences(*positions)
    # Note: several sequences can possibly be found in the reference text, when similar text patterns are repeated
    # always return the longest
    return max(seqs, key=len)


def _get_sequences(*args, value=None, path=None):
    """
    Recursive method to select sequences of successive tokens using their position relative to the reference text.
    A sequence is the list of successive indexes in the tokenized reference text.
    Implemented as a product() of successive positions constrained by their
    :param args: list of list of positions
    :param value: position of the previous token (i.e. next token position must be in range [value+1, value+3]
    :param path: debugging - current sequence
    :return:
    """
    logging.debug(path)
    # end of recursion
    if len(args) == 1:
        if value is not None:
            # return items matching constraint (i.e. within range with previous token)
            return [x for x in args[0] if x > value and (x < value+3)]
        else:
            # Special case where text is restricted to a single token
            # return all positions on first call (i.e. value is None)
            return [args[0]]
    else:
        # iterate over current token possible positions and combine with other tokens from recursive call
        # result is a list of explored sequences (i.e. list of list of positions)
        result = []
        for x in args[0]:
            # <Debug> keep track of current explored sequence
            p = [x] if path is None else list(path + [x])
            # </Debug>
            if value is None or (x > value and (x < value+3)):
                seqs = _get_sequences(*args[1:], value=x, path=p)
                # when recursion returns empty list and current position match constraint (either only value
                # or value within range) add current position as a single result
                if len(seqs) == 0 and (value is None or (x > value and (x < value+3))):
                    result.append([x])
                else:
                    # otherwise combine current position with recursion results (whether returned sequences are list
                    # or single number) and add to the list of results for this token position
                    for s in seqs:
                        res = [x] + s if type(s) is list else [x, s]
                        result.append(res)
        return result


def encode_causal_tokens(text, cause, effect, class_name='CE'):
    """
    Encode text, cause and effect into a single list with each token represented by their respective
    class labels ('-','C','E')
    :param text: reference text
    :param cause: causal substring in reference text
    :param effect: effect substring in reference text
    :return: text string converted as a list of tuple(token, label)
    """
    # Get reference text tokens and token index
    logging.debug(f'Reference: {text}')
    words, wi = build_token_index(text)
    logging.debug(f'Token index: {wi}')

    # init labels with default class label
    labels = ['-' for _ in range(len(words))]

    # encode cause using token index
    if class_name in ['C', 'CE']:
        logging.debug(f'Cause: {cause}')
        cause_seq = get_tokens_sequence(cause, wi)
        logging.debug(f'Cause seq.: {cause_seq}')
        for position in cause_seq:
            labels[position] = 'C'

    # encode effect using token index
    if class_name in ['E', 'CE']:
        logging.debug(f'Effect: {effect}')
        effect_seq = get_tokens_sequence(effect, wi)
        logging.debug(f'Effect seq.: {effect_seq}')
        for position in effect_seq:
            labels[position] = 'E'

    logging.debug(labels)

    return zip(words, labels)


def evaluate(truth, predict, classes):
    """
    Fincausal 2020 Task 2 evaluation: returns precision, recall and F1 comparing submitting data to reference data.
    :param truth: list of Task2Data(index, text, cause, effect, labels) - reference data set
    :param predict: list of Task2Data(index, text, cause, effect, labels) - submission data set
    :param classes: list of classes
    :return: tuple(precision, recall, f1, exact match)
    """
    exact_match = 0
    y_truth = []
    y_predict = []
    multi = {}
    # First pass - process text sections with single causal relations and store others in `multi` dict()
    for t, p in zip(truth, predict):
        # Process Exact Match
        exact_match += 1 if all([x == y for x, y in zip(t.labels, p.labels)]) else 0
        # PRF: Text section with multiple causal relationship ?
        if t.index.count('.') == 2:
            # extract root index and add to the list to be processed later
            root_index = '.'.join(t.index.split('.')[:-1])
            if root_index in multi:
                multi[root_index][0].append(t.labels)
                multi[root_index][1].append(p.labels)
            else:
                multi[root_index] = [[t.labels], [p.labels]]
        else:
            # Accumulate data for precision, recall, f1 scores
            y_truth.extend(t.labels)
            y_predict.extend(p.labels)
            # exact_match += 1 if all([x == y for x, y in zip(t.labels, p.labels)]) else 0
    # Second pass - deal with text sections having multiple causal relations
    for index, section in multi.items():
        # section[0] list of possible truth labels
        # section[1] list of predicted labels
        candidates = section[1]
        # for each possible combination of truth labels - try to find the best match in predicted labels
        # then repeat, removing this match from the list of remaining predicted labels
        for t in section[0]:
            # for p in candidates:
            #     exact_match += 1 if all([x == y for x, y in zip(t.labels, p.labels)]) else 0
            #     break
            best = None
            for p in candidates:
                f1 = metrics.f1_score(t, p, labels=classes, average='weighted', zero_division=0)
                if best is None or f1 > best[1]:
                    best = (p, f1)
            # Use best to add to global evaluation
            y_truth.extend(t)
            y_predict.extend(best[0])
            # Remove best from list of candidate for next iteration
            candidates.remove(best[0])
        # Ensure all candidate predictions have been reviewed
        assert len(candidates) == 0

    precision, recall, f1, _ = metrics.precision_recall_fscore_support(y_truth, y_predict,
                                                                       labels=classes,
                                                                       average='weighted',
                                                                       zero_division=0)
    return precision, recall, f1


Task2Data = namedtuple('Task2Data', ['index', 'text', 'cause', 'effect', 'labels'])


def get_data(csv_lines, class_name='CE'):
    """
    Retrieve Task 2 data from CSV content (separator is ';') as a list of (index, text, cause, effect).
    :param csv_lines:
    :return: list of Task2Data(index, text, cause, effect, labels)
    """
    result = []
    for line in csv_lines:
        line = line.rstrip('\n')

        index, text, cause, effect = line.split(';')[:4]

        text = text.lstrip()
        cause = cause.lstrip()
        effect = effect.lstrip()

        _, labels = zip(*encode_causal_tokens(text, cause, effect, class_name))

        result.append(Task2Data(index, text, cause, effect, labels))

    return result


def is_exist(ce, ce_set):
    for item in ce_set:
        if item[0] == ce[0] and item[1] == ce[1]:
            return True
    return False


def exact_match(gt_lines, pred_lines):
    gt_dct = {}
    pred_dct = {}
    for i in range(len(gt_lines)):
        gt_line = gt_lines[i].strip()
        gt_parts = gt_line.split(';')
        sent = gt_parts[1].strip()
        # index = gt_parts[0].strip()
        # if index.count('.') == 2:
        #     index = index[:-2]

        gt_cause = gt_parts[2].strip()
        gt_effect = gt_parts[3].strip()
        if sent not in gt_dct:
            gt_dct[sent] = [(gt_cause, gt_effect)]
        else:
            if not is_exist((gt_cause, gt_effect), gt_dct[sent]):
                gt_dct[sent].append((gt_cause, gt_effect))

        pred_line = pred_lines[i].strip()
        pred_parts = pred_line.split(';')
        pred_cause = pred_parts[2].strip()
        pred_effect = pred_parts[3].strip()
        if sent not in pred_dct:
            pred_dct[sent] = [(pred_cause, pred_effect)]
        else:
            if not is_exist((pred_cause, pred_effect), pred_dct[sent]):
                pred_dct[sent].append((pred_cause, pred_effect))
    exact_match = 0
    total = 0
    for key in gt_dct:
        gt_triples = gt_dct[key]
        pred_triples = pred_dct[key]
        total += len(gt_triples)
        for gt in gt_triples:
            for pt in pred_triples:
                if gt[0] == pt[0] and gt[1] == pt[1]:
                    exact_match += 1
                    break
    return round(exact_match / total, 3)


def evaluate_files(gold_file, submission_file, output_file=None):
    """
    Evaluate Precision, Recall, F1 scores between gold_file and submission_file
    If output_file is provided, scores are saved in this file and printed to std output.
    :param gold_file: path to reference data
    :param submission_file: path to submitted data
    :param output_file: path to output file as expected by Codalab competition framework
    :return:
    """
    if os.path.exists(gold_file) and os.path.exists(submission_file):
        with open(gold_file, 'r', encoding='utf-8') as fp:
            ref_csv = fp.readlines()
        with open(submission_file, 'r', encoding='utf-8') as fp:
            sub_csv = fp.readlines()

        # Get data (skipping headers)
        logging.info('* Loading reference data')
        y_true = get_data(ref_csv[1:])
        logging.info('* Loading prediction data')
        y_pred = get_data(sub_csv[1:])

        em = exact_match(ref_csv[1:], sub_csv[1:])

        logging.info(f'Load Data: check data set length = {len(y_true) == len(y_pred)}')
        logging.info(f'Load Data: check data set ref. text = {all([x.text == y.text for x, y in zip(y_true, y_pred)])}')
        assert len(y_true) == len(y_pred)
        assert all([x.text == y.text for x, y in zip(y_true, y_pred)])

        # Process data using classes: -, C & E
        precision, recall, f1 = evaluate(y_true, y_pred, ['-', 'C', 'E'])

        scores = [
            "F1: %f\n" % f1,
            "Recall: %f\n" % recall,
            "Precision: %f\n" % precision,
            "ExactMatch: %f\n" % em
        ]

        for s in scores:
            print(s, end='')
        if output_file is not None:
            with open(output_file, 'w', encoding='utf-8') as fp:
                for s in scores:
                    fp.write(s)
    else:
        # Submission file most likely being the wrong one - tell which one we are looking for
        logging.error(f'{os.path.basename(gold_file)} not found')

    ## Save for control
    import pandas as pd
    df = pd.DataFrame.from_records(y_true)
    df.columns = ['Index', 'Text', 'Cause', 'Effect', 'TRUTH']
    dfpred = pd.DataFrame.from_records(y_pred)
    dfpred.columns = ['Index', 'Text', 'Cause', 'Effect', 'PRED']
    df['PRED'] = dfpred['PRED']
    df['TRUTH'] = df['TRUTH'].apply(lambda x: ' '.join(x))
    df['PRED'] = df['PRED'].apply(lambda x: ' '.join(x))


    ctrlpath = submission_file.split('/')
    ctrlpath.pop()
    ctrlpath = '/'.join([path_ for path_ in ctrlpath])
    df.to_csv(os.path.join(ctrlpath, 'origin_control.csv'), header=1, index=0)


def evaluate_lines(ref_csv, sub_csv, output_file=None):
    """
    Evaluate Precision, Recall, F1 scores between gold_file and submission_file
    If output_file is provided, scores are saved in this file and printed to std output.
    :param gold_file: path to reference data
    :param submission_file: path to submitted data
    :param output_file: path to output file as expected by Codalab competition framework
    :return:
    """
    # with open(gold_file, 'r', encoding='utf-8') as fp:
    #     ref_csv = fp.readlines()
    # with open(submission_file, 'r', encoding='utf-8') as fp:
    #     sub_csv = fp.readlines()

    # Get data (skipping headers)
    logging.info('* Loading reference data')
    y_true = get_data(ref_csv)
    logging.info('* Loading prediction data')
    y_pred = get_data(sub_csv)

    em = exact_match(ref_csv, sub_csv)

    logging.info(f'Load Data: check data set length = {len(y_true) == len(y_pred)}')
    logging.info(f'Load Data: check data set ref. text = {all([x.text == y.text for x, y in zip(y_true, y_pred)])}')
    assert len(y_true) == len(y_pred)
    assert all([x.text == y.text for x, y in zip(y_true, y_pred)])

    # Process data using classes: -, C & E
    precision, recall, f1 = evaluate(y_true, y_pred, ['-', 'C', 'E'])

    return round(precision, 3), round(recall, 3), round(f1, 3), round(em, 3)


def from_folder(args):
    # Folder mode - Codalab usage
    submit_dir = os.path.join(args.input, 'res')
    truth_dir = os.path.join(args.input, 'ref')
    output_dir = args.output

    if not os.path.isdir(submit_dir):
        logging.error("%s doesn't exist" % submit_dir)

    if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    o_file = os.path.join(output_dir, 'scores.txt')

    gold_list = os.listdir(truth_dir)
    for gold in gold_list:
        g_file = os.path.join(truth_dir, gold)
        s_file = os.path.join(submit_dir, gold)

        evaluate_files(g_file, s_file, o_file)


def from_file(args):
    return evaluate_files(args.ref_file, args.pred_file, args.score_file)
