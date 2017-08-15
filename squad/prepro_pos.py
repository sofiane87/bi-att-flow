import argparse
import json
import os
from squad.pos_processing import get_pos_one_hot as get_pos
from collections import Counter

from tqdm import tqdm

from squad.utils import process_tokens

import platform

def main():
    args = get_args()
    prepro(args)


def get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~/bi-att-flow")
    if 'sofiane' in platform.node().lower():
        home = os.path.expanduser("~/Code/bi-att-flow")
        
    source_dir = os.path.join(home, "data", "squad")
    target_dir = "data/squad_pos"
    parser.add_argument('-s', "--source_dir", default=source_dir)
    parser.add_argument('-t', "--target_dir", default=target_dir)
    parser.add_argument("--train_name", default='train-v1.1.json')
    parser.add_argument('-d', "--debug", action='store_true')
    parser.add_argument("--train_ratio", default=0.9, type=int)
    parser.add_argument("--mode", default="full", type=str)
    parser.add_argument("--single_path", default="", type=str)
    parser.add_argument("--tokenizer", default="PTB", type=str)
    parser.add_argument("--url", default="vision-server2.corp.ai2", type=str)
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--split", action='store_true')
    parser.add_argument("--suffix", default="")
    # TODO : put more args here
    return parser.parse_args()


def create_all(args):
    out_path = os.path.join(args.source_dir, "all-v1.1.json")
    if os.path.exists(out_path):
        return
    train_path = os.path.join(args.source_dir, args.train_name)
    train_data = json.load(open(train_path, 'r'))
    dev_path = os.path.join(args.source_dir, args.dev_name)
    dev_data = json.load(open(dev_path, 'r'))
    train_data['data'].extend(dev_data['data'])
    print("dumping all data ...")
    json.dump(train_data, open(out_path, 'w'))


def prepro(args):
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)

    if args.mode == 'full':
        prepro_each(args, 'train', out_name='train')
        prepro_each(args, 'dev', out_name='dev')
        prepro_each(args, 'dev', out_name='test')
    elif args.mode == 'all':
        create_all(args)
        prepro_each(args, 'dev', 0.0, 0.0, out_name='dev')
        prepro_each(args, 'dev', 0.0, 0.0, out_name='test')
        prepro_each(args, 'all', out_name='train')
    elif args.mode == 'single':
        assert len(args.single_path) > 0
        prepro_each(args, "NULL", out_name="single", in_path=args.single_path)
    else:
        prepro_each(args, 'train', 0.0, args.train_ratio, out_name='train')
        prepro_each(args, 'train', args.train_ratio, 1.0, out_name='dev')
        prepro_each(args, 'dev', out_name='test')


def save(args, data, shared, data_type):
    data_path = os.path.join(args.target_dir, "data_{}.json".format(data_type))
    shared_path = os.path.join(args.target_dir, "shared_{}.json".format(data_type))
    json.dump(data, open(data_path, 'w'))
    json.dump(shared, open(shared_path, 'w'))

def prepro_each(args, data_type, start_ratio=0.0, stop_ratio=1.0, out_name="default", in_path=None):
    if args.tokenizer == "PTB":
        import nltk
        sent_tokenize = nltk.sent_tokenize
        def word_tokenize(tokens):
            return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]
    elif args.tokenizer == 'Stanford':
        from my.corenlp_interface import CoreNLPInterface
        interface = CoreNLPInterface(args.url, args.port)
        sent_tokenize = interface.split_doc
        word_tokenize = interface.split_sent
    else:
        raise Exception()

    if not args.split:
        sent_tokenize = lambda para: [para]

    if 'squad' in args.source_dir.lower():
        source_path = in_path or os.path.join(args.source_dir, "{}-{}v1.1.json".format(data_type, args.suffix))
    else:
        source_path = in_path or os.path.join(args.source_dir, "{}.json".format(data_type))

    source_data = json.load(open(source_path, 'r'))

    q  = []
    p = []
    ids = []

    start_ai = int(round(len(source_data['data']) * start_ratio))
    stop_ai = int(round(len(source_data['data']) * stop_ratio))
    for ai, article in enumerate(tqdm(source_data['data'][start_ai:stop_ai])):
        pp = []
        p.append(pp)
        for pi, para in enumerate(article['paragraphs']):
            # wordss
            context = para['context']
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')
            xi = list(map(word_tokenize, sent_tokenize(context)))
            xi = [process_tokens(tokens) for tokens in xi]  # process tokens

            # given xi, add chars
            xi_pos = get_pos(xi)
            pp.append(xi_pos)
            ##### Processing Part-of_speech for context 
            # print(xi_pos)
            ##### 

            for qa in para['qas']:
                # get words
                qi = word_tokenize(qa['question'])
                qi = process_tokens(qi)
                qi_pos = get_pos([qi])[0]
                #################### replace by part-of-speech taging ################
                q.append(qi_pos)
                ids.append(qa['id'])

                # yi = []
                # cyi = []
                # answers = []

                # q.append(qi)
                # cq.append(cqi)
                # y.append(yi)
                # cy.append(cyi)
                # rx.append(rxi)
                # rcx.append(rxi)
                # ids.append(qa['id'])
                # idxs.append(len(idxs))
                # answerss.append(answers)

                #################### replace by part-of-speech taging ################


        if args.debug:
            break


    # add context here
    data = {'q': q, 'ids': ids}
    
    # q:    question token list
    # cq:   question token charchater list
    # y:    answer_start(sent_id, word_id), answer_stop+1(sent_id, word_id)
    # cy:   answer_start在token中的id， answer_stop在token中的id
    # rx:   [article_id, paragraph_id]
    # rcx:  [article_id, paragraph_id]
    # ids:  question id list
    # idxs: question id list(start from 0)
    
    shared = {'p': p}

    # x:            context tokens list
    # cx:           context tokens character list
    # p:            context
    # word_counter: context+question word_count
    # lower_word_counter: 
    # char_counter: context+question word_ch_count
    # word2vec:
    # lower_word2vec:
    
    print("saving ...")
    save(args, data, shared, out_name)

if __name__ == "__main__":
    main()