import argparse
import json
import os
import re
# data: q, cq, (dq), (pq), y, *x, *cx
# shared: x, cx, (dx), (px), word_counter, char_counter, word2vec
# no metadata
from collections import Counter

from tqdm import tqdm


def main():
    args = get_args()
    convert(args)

def get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser('~')
    parser.add_argument('-o', '--output_file', type=str)
    parser.add_argument('-i', '--input_file', type=str)
    parser.add_argument('-m', '--multiple_answer', type=int, default=1)
    # TODO : put more args here
    return parser.parse_args()

def convert(args):
    q_id = 10000001
    data_output = []
    with open(args.input_file) as input_file:
        paragraphs = []
        data = json.load(input_file)
        for question in data:
            context = question['support']
            q = question['question']
            answer = question['correct_answer']
            size = len(answer)
            answers = [{'text': context[m.start(0):(m.start(0)+size)], 'answer_start': m.start(0)} for m in re.finditer(re.escape(answer), context, re.IGNORECASE)]

            if len(answers) == 0:
                continue

            if not args.multiple_answer:
                answers = answers[:1]
            paragraphs.append({
                'context': context,
                'qas':[{
                    'id':q_id,
                    'question':q,
                    'answers':answers
                }]
            })

            q_id += 1
        
        data_output = {'data':[{
            'title': 'SciQ dataset',
            'paragraphs': paragraphs
        }]}

    with open(args.output_file, 'w') as output_file:
        json.dump(data_output, output_file)
        
main()