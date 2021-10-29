import sys
import os
import numpy as np
import random

from collections import OrderedDict
import datetime
import pickle
import json
from tqdm import tqdm
from recordclass import recordclass
import math
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.backends.cudnn.deterministic = True
from transformers import AutoTokenizer, AutoModel, AdamW
import configparser
import nltk
from eval import evaluate_lines
from nltk.tokenize import sent_tokenize


def custom_print(*msg):
    for i in range(0, len(msg)):
        if i == len(msg) - 1:
            print(msg[i])
            logger.write(str(msg[i]) + '\n')
        else:
            print(msg[i], ' ', end='')
            logger.write(str(msg[i]))


def get_sample(uid, data, datatype):
    # Sample = recordclass("Sample", "Id SrcLen SrcWords BertLen BertTokens TokenMap GT GTLen TrgLen TrgPointers")
    org_sent = data['org_sent'].strip()
    bert_sent = data['bert_sent'].strip()
    token_map = data['bert_to_org_token_map']
    gt_causality = data['original_causality']
    trg_ptrs = data['bert_pointers']
    global max_cardinality
    global max_span

    if datatype in [1, 2] and len(gt_causality) > max_cardinality:
        max_cardinality = len(gt_causality)

    org_tokens = org_sent.split(' ')
    bert_tokens = ['[unused0]'] + bert_sent.split(' ')
    pos_tags = ['[unused0]'] + data['pos_tags']
    boundary_tags = ['O' for i in range(len(bert_tokens))]
    if 'bert_boundary_tags' in data:
        boundary_tags = ['[unused0]'] + data['bert_boundary_tags']

    trg_pointers = []
    for ptr_str in trg_ptrs:
        elements = ptr_str.strip().split(' ')
        trg_pointers.append((int(elements[0]) + 1, int(elements[1]) + 1,
                             int(elements[2]) + 1, int(elements[3]) + 1))
        span_len = int(elements[1]) - int(elements[0]) + 1
        effect_len = int(elements[3]) - int(elements[2]) + 1
        if effect_len > span_len:
            span_len = effect_len
        if datatype in [1, 2] and span_len > max_span:
            max_span = span_len

    if datatype == 1:
        # trg_pointers = sorted(trg_pointers, key=lambda element: (element[0], element[2]))
        # random.shuffle(trg_pointers)
        trg_pointers.append((0, -1, -1, -1))

    if datatype == 1 and (len(bert_tokens) > max_src_len or len(trg_pointers) > max_trg_len):
        return False, None

    sample = Sample(Id=uid, SrcLen=len(org_tokens), SrcWords=org_tokens,
                    BertLen=len(bert_tokens), BertTokens=bert_tokens, POSTags=pos_tags,
                    BoundaryTags=boundary_tags,
                    TokenMap=token_map, GT=gt_causality, GTLen=len(gt_causality),
                    TrgLen=len(trg_pointers), TrgPointers=trg_pointers)
    return True, sample


def get_data(src_lines, datatype):
    samples = []
    uid = 1
    for i in range(0, len(src_lines)):
        data = json.loads(src_lines[i].strip())
        status, sample = get_sample(uid, data, datatype)
        if status:
            samples.append(sample)
            uid += 1
    return samples


def build_pos_vocab(lines):
    char_v = OrderedDict()
    char_v['<PAD>'] = 0
    char_v['<UNK>'] = 1
    char_v['['] = 2
    char_v[']'] = 3
    char_v['0'] = 4
    char_v['S'] = 5
    char_v['E'] = 6
    char_v['P'] = 7
    char_idx = 8

    pos_vocab = OrderedDict()
    pos_vocab['<PAD>'] = 0
    pos_vocab['<UNK>'] = 1
    pos_vocab['[unused0]'] = 2
    pos_vocab['[SEP]'] = 3
    k = 4
    for line in lines:
        data = json.loads(line.strip())
        tags = data['pos_tags']
        for t in tags:
            if t not in pos_vocab:
                pos_vocab[t] = k
                k += 1

        tokens = data['bert_sent'].strip().split(' ')
        for token in tokens:
            for c in token:
                if c not in char_v:
                    char_v[c] = char_idx
                    char_idx += 1
    return pos_vocab, char_v


def get_answer_pointers(arg1start_preds, arg1end_preds, arg2start_preds, arg2end_preds, sent_len):
    arg1_prob = -1.0
    arg1start = -1
    arg1end = -1
    max_ent_len = int(sent_len * max_span_percent)   # 0.8
    window = sent_len   # 100
    for i in range(0, sent_len):
        for j in range(i, min(sent_len, i + max_ent_len)):
            if arg1start_preds[i] * arg1end_preds[j] > arg1_prob:
                arg1_prob = arg1start_preds[i] * arg1end_preds[j]
                arg1start = i
                arg1end = j

    arg2_prob = -1.0
    arg2start = -1
    arg2end = -1
    for i in range(max(0, arg1start - window), arg1start):
        for j in range(i, min(arg1start, i + max_ent_len)):
            if arg2start_preds[i] * arg2end_preds[j] > arg2_prob:
                arg2_prob = arg2start_preds[i] * arg2end_preds[j]
                arg2start = i
                arg2end = j
    for i in range(arg1end + 1, min(sent_len, arg1end + window)):
        for j in range(i, min(sent_len, i + max_ent_len)):
            if arg2start_preds[i] * arg2end_preds[j] > arg2_prob:
                arg2_prob = arg2start_preds[i] * arg2end_preds[j]
                arg2start = i
                arg2end = j
    # return arg1start, arg1end, arg2start, arg2end

    arg2_prob1 = -1.0
    arg2start1 = -1
    arg2end1 = -1
    for i in range(0, sent_len):
        for j in range(i, min(sent_len, i + max_ent_len)):
            if arg2start_preds[i] * arg2end_preds[j] > arg2_prob1:
                arg2_prob1 = arg2start_preds[i] * arg2end_preds[j]
                arg2start1 = i
                arg2end1 = j

    arg1_prob1 = -1.0
    arg1start1 = -1
    arg1end1 = -1
    for i in range(max(0, arg2start1 - window), arg2start1):
        for j in range(i, min(arg2start1, i + max_ent_len)):
            if arg1start_preds[i] * arg1end_preds[j] > arg1_prob1:
                arg1_prob1 = arg1start_preds[i] * arg1end_preds[j]
                arg1start1 = i
                arg1end1 = j
    for i in range(arg2end1 + 1, min(sent_len, arg2end1 + window)):
        for j in range(i, min(sent_len, i + max_ent_len)):
            if arg1start_preds[i] * arg1end_preds[j] > arg1_prob1:
                arg1_prob1 = arg1start_preds[i] * arg1end_preds[j]
                arg1start1 = i
                arg1end1 = j
    if arg1_prob * arg2_prob > arg1_prob1 * arg2_prob1:
        return arg1start, arg1end, arg2start, arg2end
    else:
        return arg1start1, arg1end1, arg2start1, arg2end1


def is_full_match(triplet, triplets):
    for t in triplets:
        if t[0] == triplet[0] and t[1] == triplet[1]:
            return True
    return False


def get_gt_triples(src_words, pointers):
    triples = []
    for i in range(len(pointers)):
        arg1 = ' '.join(src_words[pointers[i][0]:pointers[i][1] + 1])
        arg2 = ' '.join(src_words[pointers[i][2]:pointers[i][3] + 1])
        triplet = (arg1.strip(), arg2.strip())
        if not is_full_match(triplet, triples):
            triples.append(triplet)
    return triples


def construct_sent(bert_tokens, start, end, org_tokens, token_map):
    while start > 0:
        if bert_tokens[start].startswith('##'):
            start -= 1
        else:
            break
    while end > 0:
        if bert_tokens[end].startswith('##'):
            end -= 1
        else:
            break
    try:
        org_start = token_map[str(start)]
        org_end = token_map[str(end)]
    except:
        custom_print(bert_tokens)
        custom_print(org_tokens)
        custom_print(start)
        custom_print(end)
        custom_print(token_map)
        org_start = 0
        org_end = 0
    # while org_end < len(org_tokens):
    #     if len(org_tokens[org_end]) > 0 and org_tokens[org_end][-1] not in [',', '.', '?', ':', ';']:
    #         org_end += 1
    #     elif len(org_tokens[org_end]) == 0:
    #         org_end += 1
    #     else:
    #         break
    span = ' '.join(org_tokens[org_start: org_end + 1])
    span = span.strip()
    if len(span) > 0 and span[-1] == ',':
        span = span[:-1]
    return span.strip()


def get_pred_triples(arg1s, arg1e, arg2s, arg2e, cardinality, bert_tokens, org_tokens, token_map, gt_len):
    triples = []
    all_triples = []
    min_length = 0
    if dataset_name in datasets[1:]:
        min_length = gt_len
    if dataset_name == datasets[0] and use_cardinality:
        if cardinality < 0.5:
            return triples, all_triples
        else:
            min_length = 1
    for i in range(0, len(arg1s)):
        if np.argmax(arg1s[i]) == 0 and len(triples) >= min_length:
            break
        # pred_idx = np.argmax(arg1s[i])
        # if job_mode in ['train', 'train5fold'] and np.argmax(arg1s[i]) == 0 and len(triples) >= min_length:
        #     break
        # if job_mode == 'eval' and len(triples) == gt_len:
        #     break
        s1, e1, s2, e2 = get_answer_pointers(arg1s[i][1:], arg1e[i][1:], arg2s[i][1:], arg2e[i][1:],
                                             len(bert_tokens)-1)
        arg1 = construct_sent(bert_tokens[1:], s1, e1, org_tokens, token_map)
        arg2 = construct_sent(bert_tokens[1:], s2, e2, org_tokens, token_map)
        if len(arg1) == 0 or len(arg2) == 0:
            continue
        triplet = (arg1, arg2)
        all_triples.append(triplet)
        if not is_full_match(triplet, triples):
            triples.append(triplet)
    return triples, all_triples


def get_EM_F1(data, preds):
    gt_pos = 0
    pred_pos = 0
    total_pred_pos = 0
    correct_pos = 0
    cnt = 0
    for i in range(0, len(data)):
        # gt_triples = get_gt_triples(data[i].SrcWords, data[i].TrgPointers)
        gt_triples = data[i].GT
        if use_cardinality:
            pred_triples, all_pred_triples = get_pred_triples(preds[0][i], preds[1][i], preds[2][i], preds[3][i],
                                                              preds[4][i], data[i].BertTokens, data[i].SrcWords,
                                                              data[i].TokenMap, data[i].GTLen)
        else:
            pred_triples, all_pred_triples = get_pred_triples(preds[0][i], preds[1][i], preds[2][i], preds[3][i],
                                                              None, data[i].BertTokens, data[i].SrcWords,
                                                              data[i].TokenMap, data[i].GTLen)
        if len(pred_triples) < data[i].GTLen:
            cnt += 1
        total_pred_pos += len(all_pred_triples)
        gt_pos += len(gt_triples)
        pred_pos += len(pred_triples)
        for gt_triple in gt_triples:
            if is_full_match(gt_triple, pred_triples):
                correct_pos += 1

    res = list()
    res.append('less pair extracted: ' + str(cnt))
    res.append('total_pred_pos: ' + str(total_pred_pos))
    res.append('pred_pos: ' + str(pred_pos))
    res.append('gt_pos: ' + str(gt_pos))
    res.append('correct_pos: ' + str(correct_pos))
    custom_print('\t'.join(res))

    p = round(float(correct_pos) / (pred_pos + 1e-8), 3)
    r = round(float(correct_pos) / (gt_pos + 1e-8), 3)
    f1 = round((2 * p * r) / (p + r + 1e-8), 3)
    res = list()
    res.append('Prec.: ' + str(p))
    res.append('Rec.: ' + str(r))
    res.append('F1.: ' + str(f1))
    custom_print('\t'.join(res))
    return f1


def get_Token_F1(data, preds):
    gt_lines = []
    pred_lines = []
    cnt = 0
    for i in range(0, len(data)):
        index = '0000.' + str(data[i].Id)
        sent = ' '.join(data[i].SrcWords)
        gt_triples = data[i].GT
        if data[i].TrgLen == 1:
            cause = gt_triples[0][0]
            effect = gt_triples[0][1]
            out_line = index + '; ' + sent + '; ' + cause + '; ' + effect
            gt_lines.append(out_line)
        else:
            for j in range(data[i].TrgLen):
                cur_index = index + '.' + str(j + 1)
                cause = gt_triples[j][0]
                effect = gt_triples[j][1]
                out_line = cur_index + '; ' + sent + '; ' + cause + '; ' + effect
                gt_lines.append(out_line)

        if use_cardinality:
            pred_triples, all_pred_triples = get_pred_triples(preds[0][i], preds[1][i], preds[2][i], preds[3][i],
                                                              preds[4][i], data[i].BertTokens, data[i].SrcWords,
                                                              data[i].TokenMap, data[i].GTLen)
        else:
            pred_triples, all_pred_triples = get_pred_triples(preds[0][i], preds[1][i], preds[2][i], preds[3][i],
                                                              None, data[i].BertTokens, data[i].SrcWords,
                                                              data[i].TokenMap, data[i].GTLen)
        if len(pred_triples) < data[i].GTLen:
            cnt += 1
        if data[i].TrgLen == 1:
            cause = ''
            effect = ''
            if len(pred_triples) > 0:
                cause = pred_triples[0][0]
                effect = pred_triples[0][1]
            out_line = index + '; ' + sent + '; ' + cause + '; ' + effect
            pred_lines.append(out_line)
        else:
            for j in range(data[i].TrgLen):
                cur_index = index + '.' + str(j + 1)
                cause = ''
                effect = ''
                if j < len(pred_triples):
                    cause = pred_triples[j][0]
                    effect = pred_triples[j][1]
                out_line = cur_index + '; ' + sent + '; ' + cause + '; ' + effect
                pred_lines.append(out_line)
    p, r, f1, em_f1 = evaluate_lines(gt_lines, pred_lines)
    res = list()
    res.append('Prec.: ' + str(p))
    res.append('Rec.: ' + str(r))
    res.append('Token F1.: ' + str(f1))
    res.append('EM F1:' + str(em_f1))
    custom_print('\t'.join(res))
    return f1


def get_F1(data, preds):
    if metric == 'EM_F1':
        return get_EM_F1(data, preds)
    else:
        return get_Token_F1(data, preds)


def write_test_res(data, preds, outfile):
    writer = open(outfile, 'w')
    less_cnt = 0
    more_cnt = 0
    for i in range(0, len(data)):
        json_data = OrderedDict()
        json_data['id'] = data[i].Id
        json_data['sent'] = ' '.join(data[i].SrcWords)
        json_data['original_causality'] = data[i].GT
        json_data['GTLen'] = data[i].GTLen
        if use_cardinality:
            pred_triples, all_pred_triples = get_pred_triples(preds[0][i], preds[1][i], preds[2][i], preds[3][i],
                                                              preds[4][i], data[i].BertTokens, data[i].SrcWords,
                                                              data[i].TokenMap, data[i].GTLen)
        else:
            pred_triples, all_pred_triples = get_pred_triples(preds[0][i], preds[1][i], preds[2][i], preds[3][i],
                                                              None, data[i].BertTokens, data[i].SrcWords,
                                                              data[i].TokenMap, data[i].GTLen)
        json_data['predicted_causality'] = pred_triples
        writer.write(json.dumps(json_data) + '\n')
        if len(pred_triples) < data[i].TrgLen:
            less_cnt += 1
        if len(pred_triples) > data[i].TrgLen:
            more_cnt += 1
    writer.close()
    custom_print('Less pair extracted: ', less_cnt)
    custom_print('More pair extracted: ', more_cnt)


def get_unique_set(in_set):
    out_set = []
    for p in in_set:
        if not is_full_match(p, out_set):
            out_set.append(p)
    return out_set


def get_word_idx(char_pos, mapping):
    for widx in mapping:
        if mapping[widx][0] <= char_pos <= mapping[widx][1]:
            return widx
    raise Exception


def apply_fincausal_rule_1(text, pred):
    new_pred = []
    sents = sent_tokenize(text)
    span_dct = OrderedDict()
    idx = 1
    for p in pred:
        cspan = p[0]
        if cspan not in span_dct:
            span_dct[cspan] = idx
            idx += 1
        espan = p[1]
        if espan not in span_dct:
            span_dct[espan] = idx
            idx += 1
    sent_to_span = OrderedDict()
    for sent in sents:
        sent_to_span[sent] = []
    for span in span_dct:
        for sent in sents:
            if span in sent:
                sent_to_span[sent].append(span)
                break
    span_to_new_span = OrderedDict()
    for span in span_dct:
        span_to_new_span[span] = span
    for sent in sent_to_span:
        if len(sent_to_span[sent]) == 1:
            span = sent_to_span[sent][0]
            span_to_new_span[span] = sent
    for p in pred:
        new_cause = span_to_new_span[p[0]]
        new_effect = span_to_new_span[p[1]]
        new_pred.append((new_cause, new_effect))
    return new_pred


def apply_fincausal_rule_3(text, pred):
    error_cnt = 0
    new_pred = []
    sents = sent_tokenize(text)
    word_idx_to_char_idx_list = []
    for sent in sents:
        word_idx_to_char_idx = OrderedDict()
        char_idx = 0
        word_idx = 0
        tokens = sent.split(' ')
        for t in tokens:
            word_idx_to_char_idx[word_idx] = (char_idx, char_idx + len(t) - 1)
            word_idx += 1
            char_idx += len(t)
        word_idx_to_char_idx_list.append(word_idx_to_char_idx)
    for p in pred:
        matched = False
        for idx in range(len(sents)):
            sent = sents[idx]
            cur_word_idx_to_char_idx = word_idx_to_char_idx_list[idx]
            if p[0] in sent and p[1] in sent:
                try:
                    cause_start_char_idx = sent.find(p[0])
                    cause_start = get_word_idx(cause_start_char_idx, cur_word_idx_to_char_idx)
                    cause_end = cause_start + len(p[0].split(' ')) - 1
                    effect_start_char_idx = sent.find(p[1])
                    effect_start = get_word_idx(effect_start_char_idx, cur_word_idx_to_char_idx)
                    effect_end = effect_start + len(p[1].split(' ')) - 1
                    if cause_start < effect_start:
                        new_cause = sent[:cause_start_char_idx + len(p[0])].strip()
                        new_effect = sent[effect_start_char_idx:].strip()
                    else:
                        new_effect = sent[:effect_start_char_idx + len(p[1])].strip()
                        new_cause = sent[cause_start_char_idx:].strip()
                    if new_cause[-1] == ',':
                        new_cause = new_cause[:-1]
                    if new_effect[-1] == ',':
                        new_effect = new_effect[:-1]
                    new_pred.append((new_cause, new_effect))
                    matched = True
                except:
                    error_cnt += 1
                    # print('Error')
                break
        if not matched:
            new_pred.append(p)
    return new_pred


def fincausal_score(sent_list, gt_list, pred_list):
    gt_lines = []
    pred_lines = []
    for id in range(len(sent_list)):
        index = '0000.' + str(id)
        sent = sent_list[id].strip()
        gt_triples = gt_list[id]
        if len(gt_triples) == 1:
            cause = gt_triples[0][0]
            effect = gt_triples[0][1]
            out_line = index + '; ' + sent + '; ' + cause + '; ' + effect
            gt_lines.append(out_line)
        else:
            for j in range(len(gt_triples)):
                cur_index = index + '.' + str(j + 1)
                cause = gt_triples[j][0]
                effect = gt_triples[j][1]
                out_line = cur_index + '; ' + sent + '; ' + cause + '; ' + effect
                gt_lines.append(out_line)

        pred_triples = pred_list[id]
        if len(gt_triples) == 1:
            cause = ''
            effect = ''
            if len(pred_triples) > 0:
                cause = pred_triples[0][0]
                effect = pred_triples[0][1]
            out_line = index + '; ' + sent + '; ' + cause + '; ' + effect
            pred_lines.append(out_line)
        else:
            for j in range(len(gt_triples)):
                cur_index = index + '.' + str(j + 1)
                cause = ''
                effect = ''
                if j < len(pred_triples):
                    cause = pred_triples[j][0]
                    effect = pred_triples[j][1]
                out_line = cur_index + '; ' + sent + '; ' + cause + '; ' + effect
                pred_lines.append(out_line)
    print(len(gt_lines))
    print(len(pred_lines))
    p, r, f1, em_f1 = evaluate_lines(gt_lines, pred_lines)
    return p, r, f1, em_f1


def get_fincausal_score(in_file):
    lines = open(in_file).readlines()
    sent_list = []
    gt_list = []
    pred_list = []
    for line in lines:
        data = json.loads(line.strip())
        sent_list.append(data['sent'])
        gt = get_unique_set(data['original_causality'])
        gt_list.append(gt)
        pred = get_unique_set(data['predicted_causality'])
        pred = pred[:min(len(gt), len(pred))]
        pred = apply_fincausal_rule_1(data['sent'], pred)
        pred = apply_fincausal_rule_3(data['sent'], pred)
        pred_list.append(pred)
    p, r, f1, em_f1 = fincausal_score(sent_list, gt_list, pred_list)
    return p, r, f1, em_f1


def write_blind_res(data, id_to_sent, preds, outfile):
    writer = open(outfile, 'w')
    writer.write('Index; Text; Cause; Effect' + '\n')
    less_cnt = 0
    more_cnt = 0
    for i in range(0, len(data)):
        index = data[i].Id
        sent = id_to_sent[index]
        if use_cardinality:
            pred_triples, all_pred_triples = get_pred_triples(preds[0][i], preds[1][i], preds[2][i], preds[3][i],
                                                              preds[4][i], data[i].BertTokens, data[i].SrcWords,
                                                              data[i].TokenMap, data[i].GTLen)
        else:
            pred_triples, all_pred_triples = get_pred_triples(preds[0][i], preds[1][i], preds[2][i], preds[3][i],
                                                              None, data[i].BertTokens, data[i].SrcWords,
                                                              data[i].TokenMap, data[i].GTLen)
        if len(pred_triples) < data[i].TrgLen:
            less_cnt += 1
        if len(pred_triples) > data[i].TrgLen:
            more_cnt += 1
        if data[i].TrgLen == 1:
            cause = pred_triples[0][0]
            effect = pred_triples[0][1]
            out_line = index + '; ' + sent + '; ' + cause + '; ' + effect + '\n'
            writer.write(out_line)
        else:
            for j in range(data[i].TrgLen):
                cur_index = index + '.' + str(j + 1)
                cause = ''
                effect = ''
                if j < len(pred_triples):
                    cause = pred_triples[j][0]
                    effect = pred_triples[j][1]
                out_line = cur_index + '; ' + sent + '; ' + cause + '; ' + effect + '\n'
                writer.write(out_line)
    custom_print('Less pair extracted: ', less_cnt)
    custom_print('More pair extracted: ', more_cnt)
    writer.close()


def shuffle_data(data):
    custom_print(len(data))
    # data.sort(key=lambda x: x.SrcLen)
    num_batch = int(len(data) / batch_size)
    rand_idx = random.sample(range(num_batch), num_batch)
    new_data = []
    for idx in rand_idx:
        new_data += data[batch_size * idx: batch_size * (idx + 1)]
    if len(new_data) < len(data):
        new_data += data[num_batch * batch_size:]
    return new_data


def get_max_len(sample_batch):
    src_max_len = len(sample_batch[0].BertTokens)
    for idx in range(1, len(sample_batch)):
        if len(sample_batch[idx].BertTokens) > src_max_len:
            src_max_len = len(sample_batch[idx].BertTokens)

    trg_max_len = len(sample_batch[0].TrgPointers)
    for idx in range(1, len(sample_batch)):
        if len(sample_batch[idx].TrgPointers) > trg_max_len:
            trg_max_len = len(sample_batch[idx].TrgPointers)

    return src_max_len, trg_max_len


def get_words_index_seq(words, tags, boundary_tags, max_len):
    toks = ['[CLS]'] + [wd for wd in words] + ['[SEP]'] + ['[PAD]' for i in range(max_len - len(words))]
    bert_ids = bert_tokenizer.convert_tokens_to_ids(toks)
    bert_mask = [1 for idx in range(len(words) + 2)] + [0 for idx in range(max_len - len(words))]
    src_mask = [0 for i in range(len(words))] + [1 for i in range(max_len + 1 - len(words))]

    pos_seq = list()
    for t in tags:
        if t in pos_vocab:
            pos_seq.append(pos_vocab[t])
        else:
            pos_seq.append(pos_vocab['<UNK>'])
    pos_seq.append(pos_vocab['[SEP]'])
    pad_len = max_len - len(words)
    for i in range(0, pad_len):
        pos_seq.append(pos_vocab['<PAD>'])

    boundary_seq = [sent_boundary_tags[t] for t in boundary_tags] + [sent_boundary_tags['[SEP]']]
    boundary_seq += [sent_boundary_tags['<PAD>'] for i in range(pad_len)]

    char_seq = list()
    for i in range(0, conv_filter_size - 1):
        char_seq.append(char_vocab['<PAD>'])
    for word in words + ['[SEP]']:
        for c in word[0:min(len(word), max_word_len)]:
            if c in char_vocab:
                char_seq.append(char_vocab[c])
            else:
                char_seq.append(char_vocab['<UNK>'])
        pad_len = max_word_len - len(word)
        for i in range(0, pad_len):
            char_seq.append(char_vocab['<PAD>'])
        for i in range(0, conv_filter_size - 1):
            char_seq.append(char_vocab['<PAD>'])

    pad_len = max_len - len(words)
    for i in range(0, pad_len):
        for i in range(0, max_word_len + conv_filter_size - 1):
            char_seq.append(char_vocab['<PAD>'])
    return bert_ids, bert_mask, src_mask, pos_seq, char_seq, boundary_seq


def get_padded_pointers(pointers, pidx, max_len):
    idx_list = []
    for p in pointers:
        idx_list.append(p[pidx])
    pad_len = max_len + 1 - len(pointers)
    for i in range(0, pad_len):
        idx_list.append(-1)
    return idx_list


def get_pointer_location(pointers, pidx, src_max_len, trg_max_len):
    loc_seq = []
    for p in pointers:
        cur_seq = [0 for i in range(src_max_len)]
        cur_seq[p[pidx]] = 1
        loc_seq.append(cur_seq)
    pad_len = trg_max_len + 1 - len(pointers)
    for i in range(pad_len):
        cur_seq = [0 for i in range(src_max_len)]
        loc_seq.append(cur_seq)
    return loc_seq


def get_batch_data(cur_samples, is_training=False):
    """
    Returns the training samples and labels as numpy array
    """
    batch_src_max_len, batch_trg_max_len = get_max_len(cur_samples)
    batch_trg_max_len += 1
    src_words_list = list()
    bert_mask_list = list()
    pos_seq_list = list()
    src_char_seq = list()
    src_words_mask_list = list()
    src_sent_boundary_tags_list = list()
    arg1_start_seq = list()
    arg1_end_seq = list()
    arg2_start_seq = list()
    arg2_end_seq = list()
    cardinality_target = list()

    for sample in cur_samples:
        bert_ids, bert_mask, src_mask, pos_seq, char_seq, bound_seq = get_words_index_seq(sample.BertTokens,
                                                                                          sample.POSTags,
                                                                                          sample.BoundaryTags,
                                                                                          batch_src_max_len)
        src_sent_boundary_tags_list.append(bound_seq)
        src_words_list.append(bert_ids)
        bert_mask_list.append(bert_mask)
        src_words_mask_list.append(src_mask)
        pos_seq_list.append(pos_seq)
        src_char_seq.append(char_seq)

        if is_training:
            arg1_start_seq.append(get_padded_pointers(sample.TrgPointers, 0, batch_trg_max_len))
            arg1_end_seq.append(get_padded_pointers(sample.TrgPointers, 1, batch_trg_max_len))
            arg2_start_seq.append(get_padded_pointers(sample.TrgPointers, 2, batch_trg_max_len))
            arg2_end_seq.append(get_padded_pointers(sample.TrgPointers, 3, batch_trg_max_len))
            if dataset_name == datasets[0]:
                if sample.GTLen == 0:
                    cardinality_target.append(0)
                else:
                    cardinality_target.append(1)
            else:
                if sample.GTLen == 1:
                    cardinality_target.append(0)
                else:
                    cardinality_target.append(1)

    return {'src_words': np.array(src_words_list, dtype=np.float32),
            'bert_mask': np.array(bert_mask_list),
            'src_pos_tags': np.array(pos_seq_list),
            'src_chars': np.array(src_char_seq),
            'src_words_mask': np.array(src_words_mask_list),
            'src_boundary_tags': np.array(src_sent_boundary_tags_list),
            'arg1_start': np.array(arg1_start_seq),
            'arg1_end': np.array(arg1_end_seq),
            'arg2_start': np.array(arg2_start_seq),
            'arg2_end': np.array(arg2_end_seq),
            'cardinality_target': np.array(cardinality_target)}


def get_orthogonal(x, y):
    xy = torch.sum(torch.mul(x, y), dim=-1).squeeze()
    yy = torch.sum(torch.mul(y, y), dim=-1).squeeze()
    xyy = torch.mul(xy.unsqueeze(1), y)
    x_plus = torch.div(xyy, yy.unsqueeze(1))
    x_minus = x - x_plus
    return x_plus, x_minus


class MFAttention(nn.Module):
    def __init__(self, q_dim, kv_dim, mfc=1):
        super(MFAttention, self).__init__()
        self.kv_dim = kv_dim
        self.q_dim = q_dim
        self.mfc = mfc
        self.ctx_list = nn.ModuleList()
        self.query_list = nn.ModuleList()
        self.v_list = nn.ModuleList()
        for i in range(self.mfc):
            self.ctx_list.append(nn.Linear(self.kv_dim, self.kv_dim, bias=False))
            self.query_list.append(nn.Linear(self.q_dim, self.kv_dim, bias=True))
            self.v_list.append(nn.Linear(self.kv_dim, 1))

    def forward(self, q, kv, kv_mask):
        uh = self.ctx_list[0](kv)
        wq = self.query_list[0](q)
        wquh = torch.tanh(wq + uh)
        attn_weights = self.v_list[0](wquh).squeeze()
        attn_weights.data.masked_fill_(kv_mask.data, -float('inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.unsqueeze(1)
        for i in range(1, self.mfc):
            uh = self.ctx_list[i](kv)
            wq = self.query_list[i](q)
            wquh = torch.tanh(wq + uh)
            cur_attn_weights = self.v_list[i](wquh).squeeze()
            cur_attn_weights.data.masked_fill_(kv_mask.data, -float('inf'))
            cur_attn_weights = F.softmax(cur_attn_weights, dim=-1)
            cur_attn_weights = cur_attn_weights.unsqueeze(1)
            attn_weights = torch.cat((attn_weights, cur_attn_weights), 1)
        attn_weights = torch.max(attn_weights, 1, keepdim=True)[0]
        ctx = torch.bmm(attn_weights, kv).squeeze() / torch.sum(attn_weights.squeeze(), -1)
        return ctx, attn_weights.squeeze()


class MHAttention(nn.Module):
    def __init__(self, q_dim, kv_dim, mfc=1):
        super(MHAttention, self).__init__()
        self.q_dim = q_dim
        self.kv_dim = kv_dim
        self.head_dim = int(kv_dim/mfc)
        self.mfc = mfc
        self.w_list = nn.ModuleList()
        self.ctx_list = nn.ModuleList()
        self.query_list = nn.ModuleList()
        self.v_list = nn.ModuleList()
        for i in range(self.mfc):
            self.w_list.append(nn.Linear(self.kv_dim, self.head_dim))
            self.ctx_list.append(nn.Linear(self.head_dim, self.head_dim, bias=False))
            self.query_list.append(nn.Linear(self.q_dim, self.head_dim, bias=True))
            self.v_list.append(nn.Linear(self.head_dim, 1))

    def forward(self, q, kv, kv_mask):
        head_kv = self.w_list[0](kv)
        uh = self.ctx_list[0](head_kv)
        wq = self.query_list[0](q)
        wquh = torch.tanh(wq + uh)
        attn_weights = self.v_list[0](wquh).squeeze()
        attn_weights.data.masked_fill_(kv_mask.data, -float('inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.unsqueeze(1)
        ctx = torch.bmm(attn_weights, head_kv).squeeze()
        for i in range(1, self.mfc):
            head_kv = self.w_list[i](kv)
            uh = self.ctx_list[i](head_kv)
            wq = self.query_list[i](q)
            wquh = torch.tanh(wq + uh)
            cur_attn_weights = self.v_list[i](wquh).squeeze()
            cur_attn_weights.data.masked_fill_(kv_mask.data, -float('inf'))
            cur_attn_weights = F.softmax(cur_attn_weights, dim=-1)
            cur_attn_weights = cur_attn_weights.unsqueeze(1)
            cur_ctx = torch.bmm(cur_attn_weights, head_kv).squeeze()
            ctx = torch.cat((ctx, cur_ctx), -1)
            attn_weights = torch.cat((attn_weights, cur_attn_weights), 1)
        attn_weights = torch.max(attn_weights, 1)[0]
        return ctx, attn_weights


class Attention(nn.Module):
    def __init__(self, q_dim, kv_dim):
        super(Attention, self).__init__()
        self.kv_dim = kv_dim
        self.q_dim = q_dim
        self.linear_ctx = nn.Linear(self.kv_dim, self.kv_dim, bias=False)
        self.linear_query = nn.Linear(self.q_dim, self.kv_dim, bias=True)
        self.v = nn.Linear(self.kv_dim, 1)

    def forward(self, q, kv, kv_mask):
        uh = self.linear_ctx(kv)
        wq = self.linear_query(q)
        wquh = torch.tanh(wq + uh)
        attn_weights = self.v(wquh).squeeze()
        attn_weights.data.masked_fill_(kv_mask.data, -float('inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        ctx = torch.bmm(attn_weights.unsqueeze(1), kv).squeeze()
        return ctx, attn_weights


def get_attention(q_dim, kv_dim, mfc):
    # return MHAttention(q_dim, kv_dim, mfc)
    if mfc == 1:
        return Attention(q_dim, kv_dim)
    else:
        return MHAttention(q_dim, kv_dim, mfc)


class SentTagEmbeddings(nn.Module):
    def __init__(self, tag_len, tag_dim, drop_out_rate):
        super(SentTagEmbeddings, self).__init__()
        self.embeddings = nn.Embedding(tag_len, tag_dim, padding_idx=0)
        self.dropout = nn.Dropout(drop_out_rate)

    def forward(self, pos_seq):
        pos_embeds = self.embeddings(pos_seq)
        pos_embeds = self.dropout(pos_embeds)
        return pos_embeds


class POSEmbeddings(nn.Module):
    def __init__(self, tag_len, tag_dim, drop_out_rate):
        super(POSEmbeddings, self).__init__()
        self.embeddings = nn.Embedding(tag_len, tag_dim, padding_idx=0)
        self.dropout = nn.Dropout(drop_out_rate)

    def forward(self, pos_seq):
        pos_embeds = self.embeddings(pos_seq)
        pos_embeds = self.dropout(pos_embeds)
        return pos_embeds


class CharEmbeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim, drop_out_rate):
        super(CharEmbeddings, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1d = nn.Conv1d(char_embed_dim, char_feature_dim, 3)
        self.max_pool = nn.MaxPool1d(max_word_len + conv_filter_size - 1, max_word_len + conv_filter_size - 1)
        self.dropout = nn.Dropout(drop_out_rate)

    def forward(self, char_seq):
        char_embeds = self.embeddings(char_seq)
        char_embeds = self.dropout(char_embeds)
        char_embeds = char_embeds.permute(0, 2, 1)
        char_feature = torch.tanh(self.max_pool(self.conv1d(char_embeds)))
        char_feature = char_feature.permute(0, 2, 1)
        return char_feature


class BERT(nn.Module):
    def __init__(self, drop_out_rate):
        super(BERT, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        if not update_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.dropout = nn.Dropout(drop_out_rate)

    def forward(self, input_ids, bert_mask, is_training=False):
        seq_out, pooled_out = self.bert(input_ids, attention_mask=bert_mask)
        seq_out = seq_out[:, 1:, :]
        # seq_out = self.dropout(seq_out)
        # pooled_out = self.dropout(pooled_out)
        return seq_out, pooled_out


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, drop_out_rate):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.drop_rate = drop_out_rate
        self.bert_vec = BERT(drop_out_rate)
        # self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.layers, batch_first=True,
        #                     bidirectional=True)
        if use_pos_tags:
            self.pos_embeddings = POSEmbeddings(len(pos_vocab), pos_dim, drop_out_rate)
        if use_char_feature:
            self.char_embeddings = CharEmbeddings(len(char_vocab), char_embed_dim, drop_out_rate)
        if use_sent_boundary_tag:
            self.boundary_embeddings = SentTagEmbeddings(len(sent_boundary_tags), sent_boundary_tag_dim,
                                                         drop_out_rate)
        self.dropout = nn.Dropout(self.drop_rate)

    def forward(self, word_ids, bert_mask, src_pos_tags, chars, bound_tags, is_training=False):
        bert_embeds, cls = self.bert_vec(word_ids, bert_mask, is_training)
        if use_pos_tags:
            src_pos_embeds = self.pos_embeddings(src_pos_tags)
            bert_embeds = torch.cat((bert_embeds, src_pos_embeds), -1)
        if use_char_feature:
            char_feature = self.char_embeddings(chars)
            bert_embeds = torch.cat((bert_embeds, char_feature), -1)
        if use_sent_boundary_tag:
            sent_bound_embeds = self.boundary_embeddings(bound_tags)
            bert_embeds = torch.cat((bert_embeds, sent_bound_embeds), -1)

        # outputs, hc = self.lstm(bert_embeds)
        # outputs = self.dropout(outputs)
        return bert_embeds, cls


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, drop_out_rate, max_length):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.drop_rate = drop_out_rate
        self.max_length = max_length

        if att_type == 0:
            self.dec_att = get_attention(input_dim, input_dim,  mf_count)
            self.lstm = nn.LSTMCell(4 * pointer_net_hidden_size + enc_hidden_size,
                                    self.hidden_dim)
        elif att_type == 1:
            # self.span_w = nn.Linear(2 * pointer_net_hidden_size, self.input_dim)
            self.span_att = get_attention(4 * pointer_net_hidden_size, input_dim, mf_count)
            self.lstm = nn.LSTMCell(4 * pointer_net_hidden_size + enc_hidden_size,
                                    self.hidden_dim)
        else:
            # self.span_w = nn.Linear(2 * pointer_net_hidden_size, self.input_dim)
            self.dec_att = get_attention(input_dim, input_dim, mf_count)
            self.span_att = get_attention(4 * pointer_net_hidden_size, input_dim, mf_count)
            # self.effect_att = get_attention(2 * pointer_net_hidden_size, input_dim, mf_count)
            self.lstm = nn.LSTMCell(4 * pointer_net_hidden_size + 2 * enc_hidden_size,
                                    self.hidden_dim)

        self.ap_first_pointer_lstm = nn.LSTM(enc_hidden_size + dec_hidden_size, int(pointer_net_hidden_size / 2),
                                             1, batch_first=True, bidirectional=True)
        self.op_second_pointer_lstm = nn.LSTM(enc_hidden_size + dec_hidden_size + pointer_net_hidden_size,
                                              int(pointer_net_hidden_size / 2), 1, batch_first=True, bidirectional=True)

        self.ap_start_lin = nn.Linear(pointer_net_hidden_size, 1)
        self.ap_end_lin = nn.Linear(pointer_net_hidden_size, 1)
        self.op_start_lin = nn.Linear(pointer_net_hidden_size, 1)
        self.op_end_lin = nn.Linear(pointer_net_hidden_size, 1)
        self.dropout = nn.Dropout(self.drop_rate)

    def forward(self, prev_causes, prev_effects, h_prev, enc_hs, cls, prev_ctx, src_mask, is_training=False):
        src_time_steps = enc_hs.size()[1]
        repeat_h_prev = h_prev[0].squeeze().unsqueeze(1).repeat(1, src_time_steps, 1)
        repeat_causes = prev_causes.unsqueeze(1).repeat(1, src_time_steps, 1)
        repeat_effects = prev_effects.unsqueeze(1).repeat(1, src_time_steps, 1)
        prev_tuples = torch.cat((repeat_causes, repeat_effects), -1)
        ctx_list = []

        if att_type == 0:
            ctx, attn_weights = self.dec_att(repeat_h_prev, enc_hs, src_mask)
            if use_orthogonality and prev_ctx is not None:
                _, ctx = get_orthogonal(ctx, prev_ctx[0])
            ctx_list.append(ctx)
        elif att_type == 1:
            ctx, attn_weights = self.span_att(prev_tuples, enc_hs, src_mask)
            if use_orthogonality and prev_ctx is not None:
                _, ctx1 = get_orthogonal(ctx, prev_ctx[0])
            ctx_list.append(ctx)
        else:
            ctx1, attn_weights1 = self.dec_att(repeat_h_prev, enc_hs, src_mask)
            ctx2, attn_weights2 = self.span_att(prev_tuples, enc_hs, src_mask)
            if use_orthogonality and prev_ctx is not None:
                _, ctx1 = get_orthogonal(ctx1, prev_ctx[0])
                _, ctx2 = get_orthogonal(ctx2, prev_ctx[1])
            ctx_list.append(ctx1)
            ctx_list.append(ctx2)
            ctx = torch.cat((ctx1, ctx2), -1)

        s_cur = torch.cat((prev_causes, prev_effects, ctx), 1)
        hidden, cell_state = self.lstm(s_cur, h_prev)
        hidden = self.dropout(hidden)

        ap_first_pointer_lstm_input = torch.cat((enc_hs, hidden.unsqueeze(1).repeat(1, src_time_steps, 1)), 2)
        ap_first_pointer_lstm_out, phc = self.ap_first_pointer_lstm(ap_first_pointer_lstm_input)
        ap_first_pointer_lstm_out = self.dropout(ap_first_pointer_lstm_out)

        op_second_pointer_lstm_input = torch.cat((ap_first_pointer_lstm_input, ap_first_pointer_lstm_out), 2)
        op_second_pointer_lstm_out, phc = self.op_second_pointer_lstm(op_second_pointer_lstm_input)
        op_second_pointer_lstm_out = self.dropout(op_second_pointer_lstm_out)

        ap_pointer_lstm_out = ap_first_pointer_lstm_out
        op_pointer_lstm_out = op_second_pointer_lstm_out

        ap_start = self.ap_start_lin(ap_pointer_lstm_out).squeeze()
        ap_start.data.masked_fill_(src_mask.data, -float('inf'))

        ap_end = self.ap_end_lin(ap_pointer_lstm_out).squeeze()
        ap_end.data.masked_fill_(src_mask.data, -float('inf'))

        ap_start_weights = F.softmax(ap_start, dim=-1)
        ap_end_weights = F.softmax(ap_end, dim=-1)

        ap_sv = torch.bmm(ap_start_weights.unsqueeze(1), ap_pointer_lstm_out).squeeze()
        ap_ev = torch.bmm(ap_end_weights.unsqueeze(1), ap_pointer_lstm_out).squeeze()
        ap = torch.cat((ap_sv, ap_ev), -1)

        op_start = self.op_start_lin(op_pointer_lstm_out).squeeze()
        op_start.data.masked_fill_(src_mask.data, -float('inf'))

        op_end = self.op_end_lin(op_pointer_lstm_out).squeeze()
        op_end.data.masked_fill_(src_mask.data, -float('inf'))

        op_start_weights = F.softmax(op_start, dim=-1)
        op_end_weights = F.softmax(op_end, dim=-1)

        op_sv = torch.bmm(op_start_weights.unsqueeze(1), op_pointer_lstm_out).squeeze()
        op_ev = torch.bmm(op_end_weights.unsqueeze(1), op_pointer_lstm_out).squeeze()
        op = torch.cat((op_sv, op_ev), -1)

        if is_training:
            ap_start = F.log_softmax(ap_start, dim=-1)
            ap_end = F.log_softmax(ap_end, dim=-1)
            op_start = F.log_softmax(op_start, dim=-1)
            op_end = F.log_softmax(op_end, dim=-1)

            return ap_start.unsqueeze(1), ap_end.unsqueeze(1), op_start.unsqueeze(1), \
                op_end.unsqueeze(1), (hidden, cell_state), ap, op, ctx_list
        else:
            ap_start = F.softmax(ap_start, dim=-1)
            ap_end = F.softmax(ap_end, dim=-1)
            op_start = F.softmax(op_start, dim=-1)
            op_end = F.softmax(op_end, dim=-1)
            return ap_start.unsqueeze(1), ap_end.unsqueeze(1), op_start.unsqueeze(1), \
                   op_end.unsqueeze(1), (hidden, cell_state), ap, op, ctx_list


class Seq2SeqModel(nn.Module):
    def __init__(self):
        super(Seq2SeqModel, self).__init__()
        self.encoder = Encoder(enc_inp_size, int(enc_hidden_size/2), 1, drop_rate)
        self.decoder = Decoder(dec_inp_size, dec_hidden_size, 1, drop_rate, max_trg_len)
        if use_cardinality:
            self.card_classifier = nn.Linear(bert_hidden_size, 1)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, src_words_seq, bert_mask, src_mask, src_pos_tags, chars, boundary_tags,
                trg_seq_len, is_training=False):
        batch_len = src_mask.size()[0]
        src_seq_len = src_mask.size()[1]

        enc_hs, cls = self.encoder(src_words_seq, bert_mask, src_pos_tags, chars, boundary_tags, is_training)
        if use_cardinality:
            if is_training:
                card_logits = self.card_classifier(cls)
            else:
                card_logits = torch.sigmoid(self.card_classifier(cls))

        h0 = autograd.Variable(torch.FloatTensor(torch.zeros(batch_len, dec_hidden_size))).cuda()
        c0 = autograd.Variable(torch.FloatTensor(torch.zeros(batch_len, dec_hidden_size))).cuda()
        dec_hid = (h0, c0)
        # dec_hid = (cls, cls)

        prev_causes = autograd.Variable(torch.FloatTensor(torch.zeros(batch_len, 2 * pointer_net_hidden_size))).cuda()
        prev_effects = autograd.Variable(torch.FloatTensor(torch.zeros(batch_len, 2 * pointer_net_hidden_size))).cuda()

        # prev_tuples = torch.cat((arg1, arg2), -1)

        if is_training:
            dec_outs = self.decoder(prev_causes, prev_effects, dec_hid, enc_hs, cls, None, src_mask, is_training)
        else:
            dec_outs = self.decoder(prev_causes, prev_effects, dec_hid, enc_hs, cls, None, src_mask, is_training)
        arg1s = dec_outs[0]
        arg1e = dec_outs[1]
        arg2s = dec_outs[2]
        arg2e = dec_outs[3]
        dec_hid = dec_outs[4]
        arg1 = dec_outs[5]
        arg2 = dec_outs[6]
        prev_ctx = dec_outs[7]

        for t in range(1, trg_seq_len):
            if is_training:
                prev_causes = arg1 + prev_causes
                prev_effects = arg2 + prev_effects
                # prev_tuples = torch.cat((arg1, arg2), -1) + prev_tuples
                dec_outs = self.decoder(prev_causes / (t+1), prev_effects / (t+1), dec_hid,
                                        enc_hs, cls, prev_ctx, src_mask, is_training)
            else:
                prev_causes = arg1 + prev_causes
                prev_effects = arg2 + prev_effects
                # prev_tuples = torch.cat((arg1, arg2), -1) + prev_tuples
                dec_outs = self.decoder(prev_causes / (t+1), prev_effects / (t+1), dec_hid,
                                        enc_hs, cls, prev_ctx, src_mask, is_training)

            cur_arg1s = dec_outs[0]
            cur_arg1e = dec_outs[1]
            cur_arg2s = dec_outs[2]
            cur_arg2e = dec_outs[3]
            dec_hid = dec_outs[4]
            arg1 = dec_outs[5]
            arg2 = dec_outs[6]
            prev_ctx = dec_outs[7]

            arg1s = torch.cat((arg1s, cur_arg1s), 1)
            arg1e = torch.cat((arg1e, cur_arg1e), 1)
            arg2s = torch.cat((arg2s, cur_arg2s), 1)
            arg2e = torch.cat((arg2e, cur_arg2e), 1)

        if is_training:
            arg1s = arg1s.view(-1, src_seq_len)
            arg1e = arg1e.view(-1, src_seq_len)
            arg2s = arg2s.view(-1, src_seq_len)
            arg2e = arg2e.view(-1, src_seq_len)
            if use_cardinality:
                return arg1s, arg1e, arg2s, arg2e, card_logits
            else:
                return arg1s, arg1e, arg2s, arg2e
        else:
            if use_cardinality:
                return arg1s, arg1e, arg2s, arg2e, card_logits
            else:
                return arg1s, arg1e, arg2s, arg2e


def get_model(model_id):
    if model_id == 1:
        return Seq2SeqModel()


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 1:
        torch.cuda.manual_seed_all(seed)


def predict(samples, model, model_id):
    pred_batch_size = batch_size
    batch_count = math.ceil(len(samples) / pred_batch_size)
    move_last_batch = False
    if len(samples) - batch_size * (batch_count - 1) == 1:
        move_last_batch = True
        batch_count -= 1
    arg1s = list()
    arg1e = list()
    arg2s = list()
    arg2e = list()
    cardinality = list()
    model.eval()
    set_random_seeds(random_seed)
    start_time = datetime.datetime.now()
    for batch_idx in tqdm(range(0, batch_count)):
        batch_start = batch_idx * pred_batch_size
        batch_end = min(len(samples), batch_start + pred_batch_size)
        if batch_idx == batch_count - 1 and move_last_batch:
            batch_end = len(samples)

        cur_batch = samples[batch_start:batch_end]
        cur_samples_input = get_batch_data(cur_batch, False)

        src_words_seq = torch.from_numpy(cur_samples_input['src_words'].astype('long'))
        bert_words_mask = torch.from_numpy(cur_samples_input['bert_mask'].astype('uint8'))
        src_words_mask = torch.from_numpy(cur_samples_input['src_words_mask'].astype('bool'))
        src_pos_tags = torch.from_numpy(cur_samples_input['src_pos_tags'].astype('long'))
        src_chars_seq = torch.from_numpy(cur_samples_input['src_chars'].astype('long'))
        src_boundary_tags = torch.from_numpy(cur_samples_input['src_boundary_tags'].astype('long'))

        src_words_seq = autograd.Variable(src_words_seq.cuda())
        bert_words_mask = autograd.Variable(bert_words_mask.cuda())
        src_words_mask = autograd.Variable(src_words_mask.cuda())
        src_pos_tags = autograd.Variable(src_pos_tags.cuda())
        src_chars_seq = autograd.Variable(src_chars_seq.cuda())
        src_boundary_tags = autograd.Variable(src_boundary_tags.cuda())

        with torch.no_grad():
            if model_id == 1:
                outputs = model(src_words_seq, bert_words_mask, src_words_mask, src_pos_tags,
                                src_chars_seq, src_boundary_tags, max_trg_len, False)
        if gen_order == gen_orders[0]:
            arg1s += list(outputs[0].data.cpu().numpy())
            arg1e += list(outputs[1].data.cpu().numpy())
            arg2s += list(outputs[2].data.cpu().numpy())
            arg2e += list(outputs[3].data.cpu().numpy())
        else:
            arg1s += list(outputs[2].data.cpu().numpy())
            arg1e += list(outputs[3].data.cpu().numpy())
            arg2s += list(outputs[0].data.cpu().numpy())
            arg2e += list(outputs[1].data.cpu().numpy())
        if use_cardinality:
            cardinality += list(outputs[4].data.cpu().numpy())
        model.zero_grad()

    end_time = datetime.datetime.now()
    custom_print('Prediction time:', end_time - start_time)
    return arg1s, arg1e, arg2s, arg2e, cardinality


def train_model(model_id, train_samples, dev_samples, test_samples, best_model_file):
    train_size = len(train_samples)
    batch_count = int(math.ceil(train_size/batch_size))
    move_last_batch = False
    if len(train_samples) - batch_size * (batch_count - 1) == 1:
        move_last_batch = True
        batch_count -= 1
    custom_print(batch_count)
    model = get_model(model_id)
    # for name, param in model.named_parameters():
    #     print(name)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    custom_print('Parameters size:', pytorch_total_params)

    custom_print(model)
    if torch.cuda.is_available():
        model.cuda()
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    pointer_criterion = nn.NLLLoss(ignore_index=-1)
    pos_weight = torch.ones([1]).cuda()
    pos_weight[0] = 3
    card_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00001)
    if update_bert:
        optimizer = AdamW(model.parameters(), lr=1e-05, weight_decay=1e-05, correct_bias=False)
    else:
        optimizer = optim.Adam(model.parameters(), lr=1e-03, weight_decay=1e-05)
    custom_print(optimizer)

    best_dev_acc = -1.0
    best_epoch_idx = -1
    best_epoch_seed = -1

    for epoch_idx in range(0, num_epoch):
        model.train()
        model.zero_grad()
        custom_print('Epoch:', epoch_idx + 1)
        cur_seed = random_seed + epoch_idx + 1

        set_random_seeds(cur_seed)
        # cur_shuffled_train_data = shuffle_data(train_samples)
        random.shuffle(train_samples)
        start_time = datetime.datetime.now()
        train_loss_val = 0.0
        card_loss_val = 0.0

        for batch_idx in tqdm(range(0, batch_count)):
            batch_start = batch_idx * batch_size
            batch_end = min(len(train_samples), batch_start + batch_size)
            if batch_idx == batch_count - 1 and move_last_batch:
                batch_end = len(train_samples)

            cur_batch = train_samples[batch_start:batch_end]
            cur_samples_input = get_batch_data(cur_batch, True)

            src_words_seq = torch.from_numpy(cur_samples_input['src_words'].astype('long'))
            bert_words_mask = torch.from_numpy(cur_samples_input['bert_mask'].astype('uint8'))
            src_words_mask = torch.from_numpy(cur_samples_input['src_words_mask'].astype('bool'))
            src_pos_tags = torch.from_numpy(cur_samples_input['src_pos_tags'].astype('long'))
            src_chars_seq = torch.from_numpy(cur_samples_input['src_chars'].astype('long'))
            src_boundary_tags = torch.from_numpy(cur_samples_input['src_boundary_tags'].astype('long'))

            arg1s = torch.from_numpy(cur_samples_input['arg1_start'].astype('long'))
            arg1e = torch.from_numpy(cur_samples_input['arg1_end'].astype('long'))
            arg2s = torch.from_numpy(cur_samples_input['arg2_start'].astype('long'))
            arg2e = torch.from_numpy(cur_samples_input['arg2_end'].astype('long'))
            card_target = torch.from_numpy(cur_samples_input['cardinality_target'].astype('float32'))

            src_words_seq = autograd.Variable(src_words_seq.cuda())
            bert_words_mask = autograd.Variable(bert_words_mask.cuda())
            src_words_mask = autograd.Variable(src_words_mask.cuda())
            src_pos_tags = autograd.Variable(src_pos_tags.cuda())
            src_chars_seq = autograd.Variable(src_chars_seq.cuda())
            src_boundary_tags = autograd.Variable(src_boundary_tags.cuda())

            arg1s = autograd.Variable(arg1s.cuda())
            arg1e = autograd.Variable(arg1e.cuda())
            arg2s = autograd.Variable(arg2s.cuda())
            arg2e = autograd.Variable(arg2e.cuda())
            card_target = autograd.Variable(card_target.cuda())
            trg_seq_len = arg1s.size()[1]
            if model_id == 1:
                outputs = model(src_words_seq, bert_words_mask, src_words_mask, src_pos_tags,
                                src_chars_seq, src_boundary_tags, trg_seq_len, True)

            arg1s = arg1s.view(-1, 1).squeeze()
            arg1e = arg1e.view(-1, 1).squeeze()
            arg2s = arg2s.view(-1, 1).squeeze()
            arg2e = arg2e.view(-1, 1).squeeze()

            if gen_order == gen_orders[0]:
                loss = pointer_criterion(outputs[0], arg1s) + pointer_criterion(outputs[1], arg1e) + \
                       pointer_criterion(outputs[2], arg2s) + pointer_criterion(outputs[3], arg2e)
            else:
                loss = pointer_criterion(outputs[2], arg1s) + pointer_criterion(outputs[3], arg1e) + \
                       pointer_criterion(outputs[0], arg2s) + pointer_criterion(outputs[1], arg2e)

            if use_cardinality:
                card_loss = card_criterion(outputs[4].squeeze(), card_target)
                loss += card_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            model.zero_grad()
            train_loss_val += loss.item()
            if use_cardinality:
                card_loss_val += card_loss.item()

        train_loss_val /= batch_count
        card_loss_val /= batch_count

        end_time = datetime.datetime.now()
        custom_print('Training loss:', train_loss_val)
        custom_print('Cardinality loss:', card_loss_val)
        custom_print('Training time:', end_time - start_time)

        if dev_samples is not None:
            custom_print('\nDev Results\n')
            set_random_seeds(random_seed)
            dev_preds = predict(dev_samples, model, model_id)
            dev_acc = get_F1(dev_samples, dev_preds)

            if dev_acc > best_dev_acc:
                best_epoch_idx = epoch_idx + 1
                best_epoch_seed = cur_seed
                custom_print('model saved......')
                best_dev_acc = dev_acc
                torch.save(model.state_dict(), best_model_file)

        if test_samples is not None:
            custom_print('\nTest Results\n')
            set_random_seeds(random_seed)
            test_preds = predict(test_samples, model, model_id)
            test_acc = get_F1(test_samples, test_preds)

        custom_print('\n\n')
        # if epoch_idx + 1 - best_epoch_idx >= early_stop_cnt:
        #     break

    custom_print('*******')
    if train_on_full_data:
        torch.save(model.state_dict(), best_model_file)
    else:
        custom_print('Best Epoch:', best_epoch_idx)
        custom_print('Best Dev F1:', round(best_dev_acc, 3))


def is_match(parts, parts_set):
    for p in parts_set:
        if p[0].strip() == parts[0].strip() and p[1].strip() == parts[1].strip():
            return True
    return False


def get_string(i):
    s = str(i)
    if len(s) == 1:
        return '0000' + s
    if len(s) == 2:
        return '000' + s
    if len(s) == 3:
        return '00' + s
    if len(s) == 4:
        return '0' + s
    return s


def write_fold_data(all_samples):
    # 0001.00026
    # Generate folds
    fincausal_lines = open(sys.argv[5]).readlines()[1:] + open(sys.argv[6]).readlines()[1:]
    sent_id_dct = OrderedDict()
    for line in fincausal_lines:
        line = line.strip()
        parts = line.split(';')
        sent = parts[1].strip()
        if sent not in sent_id_dct:
            sent_id_dct[sent] = [parts[2:]]
        else:
            if not is_match(parts[2:], sent_id_dct[sent]):
                sent_id_dct[sent].append(parts[2:])

    out_dir = sys.argv[7]
    cut_point = int(0.2 * len(all_samples))
    folds = [0, 1, 2, 3, 4]
    for i in folds:
        print('Fold: ', (i + 1))
        dev_samples = all_samples[i * cut_point: (i + 1) * cut_point]
        train_samples = all_samples[0: i * cut_point] + all_samples[(i + 1) * cut_point:]

        writer = open(os.path.join(out_dir, 'test_' + str(i+1)) + '.csv', 'w')
        sent_idx = 1
        for sample in dev_samples:
            index = '0000.' + get_string(sent_idx)
            data = json.loads(sample)
            sent = data['org_sent']
            if len(sent_id_dct[sent]) == 1:
                line = index + '; ' + sent + '; ' + '; '.join(sent_id_dct[sent][0])
                writer.write(line + '\n')
            else:
                index = '0000.' + get_string(sent_idx)
                ce_idx = 1
                for parts in sent_id_dct[sent]:
                    cur_index = index + '.' + str(ce_idx)
                    line = cur_index + '; ' + sent + '; ' + '; '.join(sent_id_dct[sent][ce_idx - 1])
                    writer.write(line + '\n')
                    ce_idx += 1
            sent_idx += 1
        writer.close()

        writer = open(os.path.join(out_dir, 'train_' + str(i + 1)) + '.csv', 'w')
        # sent_idx = 1
        for sample in train_samples:
            index = '0000.' + get_string(sent_idx)
            data = json.loads(sample)
            sent = data['org_sent']
            if len(sent_id_dct[sent]) == 1:
                line = index + '; ' + sent + '; ' + '; '.join(sent_id_dct[sent][0])
                writer.write(line + '\n')
            else:
                index = '0000.' + get_string(sent_idx)
                ce_idx = 1
                for parts in sent_id_dct[sent]:
                    cur_index = index + '.' + str(ce_idx)
                    line = cur_index + '; ' + sent + '; ' + '; '.join(sent_id_dct[sent][ce_idx - 1])
                    writer.write(line + '\n')
                    ce_idx += 1
            sent_idx += 1
        writer.close()
    exit(0)


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
    dataset_name = sys.argv[1]
    config_file = sys.argv[2]
    trg_data_folder = sys.argv[3]
    job_mode = sys.argv[4]

    config = configparser.ConfigParser()
    config.read(config_file)

    if not os.path.exists(trg_data_folder):
        os.mkdir(trg_data_folder)
    model_name = 1
    datasets = ['SemEval', 'FinCausal2021', 'FinCausal2020']
    gen_orders = ['CauseFirst', 'EffectFirst']
    assert dataset_name in datasets
    assert dataset_name in config.sections()

    random_seed = int(config['Model']['random_seed'])
    bert_model_name = config['Model']['bert_model_name']
    bert_tokenizer_name = config['Model']['bert_tokenizer_name']
    bert_hidden_size = int(config['Model']['bert_hidden_size'])
    update_bert = eval(config['Model']['update_bert'])
    do_lower_case = eval(config['Model']['do_lower_case'])
    update_freq = int(config['Model']['update_freq'])
    att_type = int(config['Model']['att_type'])
    use_pos_tags = eval(config['Model']['use_pos_tags'])
    pos_dim = int(config['Model']['pos_dim'])
    use_char_feature = eval(config['Model']['use_char_feature'])
    char_embed_dim = int(config['Model']['char_embed_dim'])
    char_feature_dim = int(config['Model']['char_feature_dim'])
    conv_filter_size = int(config['Model']['conv_filter_size'])
    max_word_len = int(config['Model']['max_word_len'])
    use_orthogonality = eval(config['Model']['use_orthogonality'])
    gen_order = config['Model']['generation_order']

    src_data_folder = config[dataset_name]['data_dir']
    train_file_name = config[dataset_name]['train_file']
    test_file_name = config[dataset_name]['test_file']
    batch_size = int(config[dataset_name]['batch_size'])
    num_epoch = int(config[dataset_name]['num_epoch'])
    drop_rate = float(config[dataset_name]['drop_rate'])
    max_src_len = int(config[dataset_name]['max_src_token'])
    max_trg_len = int(config[dataset_name]['max_trg_steps'])
    dev_percent = float(config[dataset_name]['dev_percent'])
    early_stop_cnt = int(config[dataset_name]['early_stop'])
    train_on_full_data = eval(config[dataset_name]['train_on_full'])
    mf_count = int(config[dataset_name]['mfc'])
    max_span_percent = float(config[dataset_name]['max_span_percent'])
    use_cardinality = eval(config[dataset_name]['use_cardinality'])
    use_sent_boundary_tag = eval(config[dataset_name]['use_sent_boundary_tag'])
    sent_boundary_tag_dim = int(config[dataset_name]['sent_boundary_tag_dim'])
    metric = config[dataset_name]['metric']

    n_gpu = torch.cuda.device_count()
    set_random_seeds(random_seed)
    if train_on_full_data:
        early_stop_cnt = 100
    max_cardinality = 1
    max_span = 1

    bert_tokenizer = AutoTokenizer.from_pretrained(bert_tokenizer_name, do_lower_case=do_lower_case)
    enc_inp_size = bert_hidden_size
    if use_pos_tags:
        enc_inp_size += pos_dim
    if use_char_feature:
        enc_inp_size += char_feature_dim
    if use_sent_boundary_tag:
        enc_inp_size += sent_boundary_tag_dim
    enc_hidden_size = enc_inp_size
    dec_inp_size = enc_hidden_size
    dec_hidden_size = dec_inp_size
    pointer_net_hidden_size = enc_hidden_size

    Sample = recordclass("Sample", "Id SrcLen SrcWords BertLen BertTokens POSTags BoundaryTags "
                                   "TokenMap GT GTLen TrgLen TrgPointers")
    sent_boundary_tags = {'[unused0]': 0, 'B': 1, 'I': 2, 'E': 3, 'O': 4, '[SEP]': 5, '<PAD>': 6}

    # train a model
    if job_mode == 'train':
        logger = open(os.path.join(trg_data_folder, 'training.log'), 'w')
        custom_print('Experiment time: ', datetime.datetime.now())
        custom_print(sys.argv)
        custom_print(dict(config['Model']))
        custom_print(dict(config[dataset_name]))

        custom_print('loading data......')
        train_file = os.path.join(src_data_folder, train_file_name)
        all_samples = open(train_file).readlines()
        custom_print('Total data:', len(all_samples))

        pos_vocab, char_vocab = build_pos_vocab(all_samples)
        output = open(os.path.join(trg_data_folder, 'vocab.pkl'), 'wb')
        pickle.dump([pos_vocab, char_vocab], output)
        output.close()

        random.shuffle(all_samples)
        cut_point = int(dev_percent * len(all_samples))
        dev_data = get_data(all_samples[: cut_point], 2)
        train_data = get_data(all_samples[cut_point:], 1)
        custom_print('Training data size:', len(train_data))
        custom_print('Development data size:', len(dev_data))

        test_file = os.path.join(src_data_folder, test_file_name)
        test_samples = open(test_file).readlines()
        test_data = get_data(test_samples, 3)
        custom_print('Test data size:', len(test_data))

        custom_print("Training started......")
        custom_print('Max cardinality:', max_cardinality)
        custom_print('Max span:', max_span)
        model_file_name = os.path.join(trg_data_folder, 'model.h5py')

        train_model(model_name, train_data, dev_data, test_data, model_file_name)
        custom_print("\n\nPrediction......")
        best_model = get_model(model_name)
        if torch.cuda.is_available():
            best_model.cuda()
        if n_gpu > 1:
            best_model = torch.nn.DataParallel(best_model)
        best_model.load_state_dict(torch.load(model_file_name))

        custom_print('\nDev Results\n')
        dev_preds = predict(dev_data, best_model, model_name)
        dev_f1 = get_F1(dev_data, dev_preds)
        write_test_res(dev_data, dev_preds, os.path.join(trg_data_folder, 'dev_out.json'))

        custom_print('\nTest Results\n')
        test_preds = predict(test_data, best_model, model_name)
        test_f1 = get_F1(test_data, test_preds)
        write_test_res(test_data, test_preds, os.path.join(trg_data_folder, 'test_out.json'))

        logger.close()

    if job_mode == 'train5fold':
        logger = open(os.path.join(trg_data_folder, 'training.log'), 'w')
        custom_print('Experiment time: ', datetime.datetime.now())
        custom_print(sys.argv)
        custom_print(dict(config['Model']))
        custom_print(dict(config[dataset_name]))

        custom_print('loading data......')
        train_file = os.path.join(src_data_folder, train_file_name)
        all_samples = open(train_file).readlines()
        custom_print('Total data:', len(all_samples))
        random.shuffle(all_samples)

        # write_fold_data(all_samples)

        pos_vocab, char_vocab = build_pos_vocab(all_samples)
        output = open(os.path.join(trg_data_folder, 'vocab.pkl'), 'wb')
        pickle.dump([pos_vocab, char_vocab], output)
        output.close()

        cut_point = int(0.2 * len(all_samples))
        best_dev_f1 = -1.0
        folds = [0, 1, 2, 3, 4]
        folds_scores = []
        for i in folds:
            custom_print('\n\nFold: ', (i+1))
            custom_print('\n\n')
            dev_data = get_data(all_samples[i * cut_point: (i + 1) * cut_point], 2)
            train_samples = all_samples[0: i * cut_point] + all_samples[(i + 1) * cut_point:]
            train_data = get_data(train_samples, 1)

            custom_print('Training data size:', len(train_data))
            custom_print('Development data size:', len(dev_data))

            custom_print("Training started......")
            custom_print('Max cardinality:', max_cardinality)
            custom_print('Max span:', max_span)
            model_file_name = os.path.join(trg_data_folder, 'model_fold_' + str(i+1) + '.h5py')

            train_model(model_name, train_data, dev_data, None, model_file_name)
            custom_print("\n\nPrediction......")
            best_model = get_model(model_name)
            if torch.cuda.is_available():
                best_model.cuda()
            if n_gpu > 1:
                best_model = torch.nn.DataParallel(best_model)
            custom_print('loading model......')
            best_model.load_state_dict(torch.load(model_file_name))

            custom_print('\nValidation Results\n')
            dev_preds = predict(dev_data, best_model, model_name)
            # dev_f1 = get_F1(dev_data, dev_preds)
            write_test_res(dev_data, dev_preds, os.path.join(trg_data_folder, 'dev_fold_' + str(i+1) + '_out.json'))
            custom_print('\n\n')

            tk_p, tk_r, tk_f1, em_f1 = get_fincausal_score(os.path.join(trg_data_folder,
                                                                        'dev_fold_' + str(i+1) + '_out.json'))
            scores = [
                "Precision: %f" % tk_p,
                "Recall: %f" % tk_r,
                "F1: %f" % tk_f1,
                "ExactMatch: %f" % em_f1
            ]
            folds_scores.append(scores)
            custom_print('\n')
            custom_print(scores)
            dev_f1 = tk_f1
            if dev_f1 > best_dev_f1:
                custom_print('Prev best:', best_dev_f1)
                custom_print('Cur best:', dev_f1)
                custom_print('\nSaving the model from Fold ' + str(i+1) + '\n')
                best_model_file = os.path.join(trg_data_folder, 'model.h5py')
                best_dev_f1 = dev_f1
                torch.save(best_model.state_dict(), best_model_file)
            os.remove(model_file_name)
        custom_print('\n\n')
        for i in folds:
            custom_print('Scores for fold: ', (i + 1))
            custom_print(folds_scores[i])
        logger.close()

