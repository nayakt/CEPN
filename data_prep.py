import sys
import os
from collections import OrderedDict
import json
from transformers import AutoTokenizer
import nltk
from nltk.tokenize import SpaceTokenizer
import xml.etree.ElementTree as ET
import configparser
from nltk.tokenize import sent_tokenize
from task2_eval import evaluate_lines, evaluate_span_lines
import argparse
import random
import pandas as pd
import pyconll


def generate_bert_json(in_file, out_file, bert_tokenizer_name):
    # bert_model_name = 'bert-large-cased'
    do_lower_case = False
    lines = open(in_file).readlines()
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_tokenizer_name, do_lower_case=do_lower_case)
    writer = open(out_file, 'w')
    for line in lines:
        data = json.loads(line.strip())
        text = data['sent'].strip()
        sentences = sent_tokenize(text)
        bound_tags = []
        bert_bound_tags = []
        for sentence in sentences:
            sentence = sentence.strip()
            bound_tags.append('B')
            toks = sentence.split(' ')
            for tok in toks[1:-1]:
                bound_tags.append('I')
            bound_tags.append('E')
            bert_toks = []
            for tok in toks:
                if do_lower_case:
                    tok = tok.lower()
                sub_tokens = bert_tokenizer.wordpiece_tokenizer.tokenize(tok)
                if len(sub_tokens) == 0:
                    sub_tokens = [tok]
                bert_toks += sub_tokens
            bert_bound_tags.append('B')
            for tok in bert_toks[1:-1]:
                bert_bound_tags.append('I')
            bert_bound_tags.append('E')

        tokens = text.split(' ')
        mod_tokens = []
        for t in tokens:
            if len(t) > 0:
                mod_tokens.append(t)
            else:
                mod_tokens.append('SPACE_TOKEN')
        assert len(tokens) == len(mod_tokens)
        # assert len(tokens) == len(bound_tags)
        pos_seq = list()
        tags = nltk.pos_tag(mod_tokens)
        for t in tags:
            pos_seq.append(t[1])

        bert_pos_seq = []
        bert_tokens = []
        token_map = OrderedDict()
        rev_token_map = OrderedDict()
        bert_idx = 0
        for j in range(len(tokens)):
            cur_token = tokens[j]
            if do_lower_case:
                cur_token = cur_token.lower()
            sub_tokens = bert_tokenizer.wordpiece_tokenizer.tokenize(cur_token)
            if len(sub_tokens) == 0:
                sub_tokens = [cur_token]
            bert_tokens += sub_tokens
            bert_pos_seq += [pos_seq[j] for k in range(len(sub_tokens))]
            token_map[j] = (bert_idx, bert_idx + len(sub_tokens) - 1)
            rev_token_map[bert_idx] = j
            bert_idx += len(sub_tokens)
        assert len(bert_tokens) == len(bert_pos_seq)
        if len(bert_bound_tags) < len(bert_tokens):
            bert_bound_tags += ['O' for i in range(len(bert_tokens) - len(bert_bound_tags))]
        elif len(bert_bound_tags) > len(bert_tokens):
            bert_bound_tags = bert_bound_tags[:len(bert_tokens)]
        assert len(bert_tokens) == len(bert_bound_tags)
        bert_pointers = []
        pointers = data['pointers']
        for p in pointers:
            ap_s, ap_e, op_s, op_e = p.split(' ')
            new_p = [str(token_map[int(ap_s)][0]), str(token_map[int(ap_e)][0]),
                     str(token_map[int(op_s)][0]), str(token_map[int(op_e)][0])]
            bert_pointers.append(' '.join(new_p))
        bert_sent = ' '.join(bert_tokens)
        trg_data = OrderedDict()
        trg_data['org_sent'] = text
        # trg_data['org_boundary_tags'] = bound_tags
        trg_data['bert_sent'] = bert_sent
        trg_data['bert_boundary_tags'] = bert_bound_tags
        trg_data['pos_tags'] = bert_pos_seq
        trg_data['bert_to_org_token_map'] = rev_token_map
        trg_data['original_causality'] = data['original_causality']
        trg_data['modified_causality'] = data['modified_causality']
        trg_data['bert_pointers'] = bert_pointers
        writer.write(json.dumps(trg_data) + '\n')
    writer.close()


def generate_train_json(in_file, out_file):
    lines = open(in_file).readlines()
    print(len(lines))
    index_dct = OrderedDict()
    cnt = 0
    for line in lines:
        line = line.strip()
        parts = line.split(';')
        sent = parts[1].strip()
        cause = parts[2].strip()
        effect = parts[3].strip()
        cause_start = int(float(parts[6].strip()))
        effect_start = int(float(parts[8].strip()))

        char_word_idx = OrderedDict()
        tokens = sent.split(' ')
        word_idx = 0
        char_idx = 0
        for tok in tokens:
            char_word_idx[char_idx] = word_idx
            word_idx += 1
            char_idx += len(tok) + 1
        cstart = char_word_idx[cause_start]
        cend = cstart + len(cause.split(' ')) - 1
        estart = char_word_idx[effect_start]
        eend = estart + len(effect.split(' ')) - 1

        new_cause = ' '.join(tokens[cstart:cend+1])
        new_effect = ' '.join(tokens[estart:eend+1])
        if sent not in index_dct:
            index_dct[sent] = [(cstart, cend, estart, eend, cause, effect, new_cause, new_effect)]
        else:
            index_dct[sent].append((cstart, cend, estart, eend, cause, effect, new_cause, new_effect))
        cnt += 1

    print(len(index_dct))
    print(cnt)
    writer = open(out_file, 'w')
    for key in index_dct:
        data = OrderedDict()
        data['sent'] = key
        # writer.write(key + '\n')
        org_ce = []
        new_ce = []
        pointers = []
        for tup in index_dct[key]:
            org_ce.append((tup[4], tup[5]))
            new_ce.append((tup[6], tup[7]))
            pointers.append(str(tup[0]) + ' ' + str(tup[1]) + ' ' + str(tup[2]) + ' ' + str(tup[3]))
        # writer.write('Original:' + ' | '.join(org_ce) + '\n')
        data['original_causality'] = org_ce
        # writer.write('modified' + ' | '.join(new_ce) + '\n')
        data['modified_causality'] = new_ce
        data['pointers'] = pointers
        writer.write(json.dumps(data) + '\n')
    writer.close()


def convert(in_tsv_file, out_json_file, out_bert_file, tokenizer):
    generate_train_json(in_tsv_file, out_json_file)
    generate_bert_json(out_json_file, out_bert_file, tokenizer)


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])


