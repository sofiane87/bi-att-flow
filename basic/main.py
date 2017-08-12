import argparse
import json
import math
import os
import shutil
from pprint import pprint

import tensorflow as tf
from tqdm import tqdm
import numpy as np

from basic.evaluator import ForwardEvaluator, MultiGPUF1Evaluator
from basic.graph_handler import GraphHandler
from basic.model import get_multi_gpu_models
from basic.trainer import MultiGPUTrainer
from basic.read_data import read_data, get_squad_data_filter, update_config
from my.tensorflow import get_num_params


def main(config):
    set_dirs(config)
    with tf.device(config.device):
        if config.mode == 'train':
            _train(config)
        elif config.mode == 'test':
            _test(config)
        elif config.mode == 'forward':
            _forward(config)
        else:
            raise ValueError("invalid value for 'mode': {}".format(config.mode))


def set_dirs(config):
    # create directories
    assert config.load or config.mode == 'train', "config.load must be True if not training"
    if not config.load and os.path.exists(config.out_dir):
        shutil.rmtree(config.out_dir)

    config.save_dir = os.path.join(config.out_dir, "save")
    config.log_dir = os.path.join(config.out_dir, "log")
    config.eval_dir = os.path.join(config.out_dir, "eval")
    config.answer_dir = os.path.join(config.out_dir, "answer")
    if not os.path.exists(config.out_dir):
        os.makedirs(config.out_dir)
    if not os.path.exists(config.save_dir):
        os.mkdir(config.save_dir)
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)
    if not os.path.exists(config.answer_dir):
        os.mkdir(config.answer_dir)
    if not os.path.exists(config.eval_dir):
        os.mkdir(config.eval_dir)


def _config_debug(config):
    if config.debug:
        config.num_steps = 2
        config.eval_period = 1
        config.log_period = 1
        config.save_period = 1
        config.val_num_batches = 2
        config.test_num_batches = 2


def _join_dataset(dataset_1, dataset_2, batch_size, num_gpus, num_steps, shuffle=False, cluster=False, model=None):
    stop = False
    gen1 = dataset_1.get_multi_batches(batch_size, num_gpus, num_steps=num_steps, shuffle=shuffle, cluster=cluster)
    gen2 = dataset_2.get_multi_batches(batch_size, num_gpus, num_steps=num_steps, shuffle=shuffle, cluster=cluster)
    while not stop:
        # rand = np.random.randint(2)
        rand = 1
        next_batch = None
        print(rand)
        if rand:
            try:
                next_batch = next(gen1)
                model.model_id_value = 1
            except:
                next_batch = None
        
        if not rand or next_batch is None:
            try:
                next_batch = next(gen2)
                model.model_id_value  = 0
            except:
                next_batch = None
        
        if next_batch is None:
            try:
                next_batch = next(gen1)
                model.model_id_value = 1
            except:
                next_batch = None
        
        if next_batch is None:
            stop = True
        else:
            yield next_batch
        
        

def _train(config):
    data_filter = get_squad_data_filter(config)
    train_data_1 = read_data(config, 'train', config.load, data_filter=data_filter, data_set_id=1)
    dev_data = read_data(config, 'dev', True, data_filter=data_filter, data_set_id=1)

    train_data_2 = read_data(config, 'train', config.load, data_filter=data_filter, data_set_id=2)
    dev_data_2 = read_data(config, 'dev', True, data_filter=data_filter, data_set_id=2)

    update_config(config, [train_data_1, dev_data, train_data_2, dev_data_2])

    _config_debug(config)

    word2vec_dict_1 = train_data_1.shared['lower_word2vec'] if config.lower_word else train_data_1.shared['word2vec']
    word2vec_dict_2 = train_data_2.shared['lower_word2vec'] if config.lower_word else train_data_2.shared['word2vec']
    word2vec_dict = {**word2vec_dict_1, **word2vec_dict_2}
    word2idx_dict = {**train_data_1.shared['word2idx'],**train_data_2.shared['word2idx']}

    idx2vec_dict = {word2idx_dict[word]: vec for word, vec in word2vec_dict.items() if word in word2idx_dict}
    emb_mat = np.array([idx2vec_dict[idx] if idx in idx2vec_dict
                        else np.random.multivariate_normal(np.zeros(config.word_emb_size), np.eye(config.word_emb_size))
                        for idx in range(config.word_vocab_size)])
    config.emb_mat = emb_mat

    # construct model graph and variables (using default graph)
    pprint(config.__flags, indent=2)
    models = get_multi_gpu_models(config)
    model = models[0]
    print("num params: {}".format(get_num_params()))
    trainer = MultiGPUTrainer(config, models)
    evaluator = MultiGPUF1Evaluator(config, models, tensor_dict=model.tensor_dict if config.vis else None)
    graph_handler = GraphHandler(config, model)  # controls all tensors and variables in the graph, including loading /saving

    # Variables
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options = tf.GPUOptions(allow_growth = True)))
    graph_handler.initialize(sess)

    # Begin training
    num_steps = config.num_steps or int(math.ceil((train_data_1.num_examples) / 
        (config.batch_size * config.num_gpus))) * config.num_epochs
    global_step = 0

    suffix = ''

    dataset_name = config.dataset_name 

    if config.use_pos:
        suffix = '_pos'

    save_path = config.log_dir + os.path.sep + dataset_name + os.path.sep
    if not(os.path.exists(save_path)):
        os.makedirs(save_path)

    loss_file = save_path + dataset_name + suffix +'_loss.txt'
    train_file = save_path +  dataset_name + suffix +'_train.txt'
    dev_file = save_path +  dataset_name + suffix + '_dev.txt'
    with open(loss_file, 'w') as f:
        f.write('\t\t\t-Losses-\t\t\t\n')

    with open(train_file, 'w') as f:
        f.write('\t\t\t-train scores-\t\t\t\n')


    with open(dev_file, 'w') as f:
        f.write('\t\t\t-dev scores-\t\t\t\n')

    numpy_loss_file = save_path +  dataset_name + suffix +'_loss'
    numpy_train_file_path = save_path + dataset_name + suffix +'_train'
    numpy_dev_file_path = save_path + dataset_name + suffix + '_dev'

    losses = []
    train_f1_scores = []
    train_exact_scores = []
    dev_f1_scores = []
    dev_exact_scores = []

    for batches in tqdm(_join_dataset(train_data_1, train_data_2, config.batch_size, config.num_gpus,
                                                     num_steps=num_steps//2, shuffle=True, cluster=config.cluster, model=models[0]), total=num_steps):

        global_step = sess.run(model.global_step) + 1  # +1 because all calculations are done after step
        get_summary = global_step % config.log_period == 0
        loss, summary, train_op = trainer.step(sess, batches, get_summary=get_summary)
        with open(loss_file, 'a') as f:
            f.write('step : {}\tloss : {}\n'.format(global_step,loss))

        losses.append(loss)

        if get_summary:
            graph_handler.add_summary(summary, global_step)

        # occasional saving
        if global_step % config.save_period == 0:
            graph_handler.save(sess, global_step=global_step)

        if not config.eval:
            continue
        # Occasional evaluation
        if global_step % config.eval_period == 0:
            models[0].model_id_value=1
            num_steps = math.ceil(dev_data.num_examples / (config.batch_size * config.num_gpus))
            if 0 < config.val_num_batches < num_steps:
                num_steps = config.val_num_batches
            e_train = evaluator.get_evaluation_from_batches(
                sess, tqdm(train_data_1.get_multi_batches(config.batch_size, config.num_gpus, num_steps=num_steps), total=num_steps)
            )
            graph_handler.add_summaries(e_train.summaries, global_step)
            e_dev = evaluator.get_evaluation_from_batches(
                sess, tqdm(dev_data.get_multi_batches(config.batch_size, config.num_gpus, num_steps=num_steps), total=num_steps))
            graph_handler.add_summaries(e_dev.summaries, global_step)

            with open(train_file, 'a') as f:
                f.write(e_train.__repr__() + '\n')

            train_f1_scores.append([e_train.f1,e_train.f1_squad])
            train_exact_scores.append([e_train.acc,e_train.acc_squad])

            with open(dev_file, 'a') as f:
                f.write(e_dev.__repr__() + '\n')

            dev_f1_scores.append([e_dev.f1,e_dev.f1_squad])
            dev_exact_scores.append([e_dev.acc,e_dev.acc_squad])


            print("\n---------------------------------------------")
            print(e_train)
            print(e_dev)
            print("\n+++++++++++++++++++++++++++++++++++++++++++++")

            models[0].model_id_value=0
            num_steps = math.ceil(dev_data.num_examples / (config.batch_size * config.num_gpus))
            if 0 < config.val_num_batches < num_steps:
                num_steps = config.val_num_batches
            e_train = evaluator.get_evaluation_from_batches(
                sess, tqdm(train_data_1.get_multi_batches(config.batch_size, config.num_gpus, num_steps=num_steps), total=num_steps)
            )
            graph_handler.add_summaries(e_train.summaries, global_step)
            e_dev = evaluator.get_evaluation_from_batches(
                sess, tqdm(dev_data.get_multi_batches(config.batch_size, config.num_gpus, num_steps=num_steps), total=num_steps))
            graph_handler.add_summaries(e_dev.summaries, global_step)

            with open(train_file, 'a') as f:
                f.write(e_train.__repr__() + '\n')

            train_f1_scores.append([e_train.f1,e_train.f1_squad])
            train_exact_scores.append([e_train.acc,e_train.acc_squad])

            with open(dev_file, 'a') as f:
                f.write(e_dev.__repr__() + '\n')

            dev_f1_scores.append([e_dev.f1,e_dev.f1_squad])
            dev_exact_scores.append([e_dev.acc,e_dev.acc_squad])

            print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            print(e_train)
            print(e_dev)
            print("\n+++++++++++++++++++++++++++++++++++++++++++++")

            if config.dump_eval:
                graph_handler.dump_eval(e_dev)
            if config.dump_answer:
                graph_handler.dump_answer(e_dev)
    
        if global_step % config.save_period == 0:
            graph_handler.save(sess, global_step=global_step)

    
    np.save(numpy_loss_file,losses)

    np.save(numpy_train_file_path + '_f1',train_f1_scores)
    np.save(numpy_train_file_path + '_exact',train_exact_scores)    

    np.save(numpy_dev_file_path + '_f1',dev_f1_scores)
    np.save(numpy_dev_file_path + '_exact',dev_exact_scores)



def _test(config):
    test_data = read_data(config, 'test', True)
    update_config(config, [test_data])

    _config_debug(config)

    if config.use_glove_for_unk:
        word2vec_dict = test_data.shared['lower_word2vec'] if config.lower_word else test_data.shared['word2vec']
        new_word2idx_dict = test_data.shared['new_word2idx']
        idx2vec_dict = {idx: word2vec_dict[word] for word, idx in new_word2idx_dict.items()}
        new_emb_mat = np.array([idx2vec_dict[idx] for idx in range(len(idx2vec_dict))], dtype='float32')
        config.new_emb_mat = new_emb_mat

    pprint(config.__flags, indent=2)
    models = get_multi_gpu_models(config)
    model = models[0]
    evaluator = MultiGPUF1Evaluator(config, models, tensor_dict=models[0].tensor_dict if config.vis else None)
    graph_handler = GraphHandler(config, model)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options = tf.GPUOptions(allow_growth = True)))
    graph_handler.initialize(sess)
    num_steps = math.ceil(test_data.num_examples / (config.batch_size * config.num_gpus))
    if 0 < config.test_num_batches < num_steps:
        num_steps = config.test_num_batches

    e = None
    for multi_batch in tqdm(test_data.get_multi_batches(config.batch_size, config.num_gpus, num_steps=num_steps, cluster=config.cluster), total=num_steps):
        ei = evaluator.get_evaluation(sess, multi_batch)
        e = ei if e is None else e + ei
        if config.vis:
            eval_subdir = os.path.join(config.eval_dir, "{}-{}".format(ei.data_type, str(ei.global_step).zfill(6)))
            if not os.path.exists(eval_subdir):
                os.mkdir(eval_subdir)
            path = os.path.join(eval_subdir, str(ei.idxs[0]).zfill(8))
            graph_handler.dump_eval(ei, path=path)

    print(e)
    if config.dump_answer:
        print("dumping answer ...")
        graph_handler.dump_answer(e)
    if config.dump_eval:
        print("dumping eval ...")
        graph_handler.dump_eval(e)


def _forward(config):
    assert config.load
    test_data = read_data(config, config.forward_name, True)
    update_config(config, [test_data])

    _config_debug(config)

    if config.use_glove_for_unk:
        word2vec_dict = test_data.shared['lower_word2vec'] if config.lower_word else test_data.shared['word2vec']
        new_word2idx_dict = test_data.shared['new_word2idx']
        idx2vec_dict = {idx: word2vec_dict[word] for word, idx in new_word2idx_dict.items()}
        new_emb_mat = np.array([idx2vec_dict[idx] for idx in range(len(idx2vec_dict))], dtype='float32')
        config.new_emb_mat = new_emb_mat

    pprint(config.__flags, indent=2)
    models = get_multi_gpu_models(config)
    model = models[0]
    print("num params: {}".format(get_num_params()))
    evaluator = ForwardEvaluator(config, model)
    graph_handler = GraphHandler(config, model)  # controls all tensors and variables in the graph, including loading /saving

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True), gpu_options = tf.GPUOptions(allow_growth = True))
    graph_handler.initialize(sess)

    num_batches = math.ceil(test_data.num_examples / config.batch_size)
    if 0 < config.test_num_batches < num_batches:
        num_batches = config.test_num_batches
    e = evaluator.get_evaluation_from_batches(sess, tqdm(test_data.get_batches(config.batch_size, num_batches=num_batches), total=num_batches))
    print(e)
    if config.dump_answer:
        print("dumping answer ...")
        graph_handler.dump_answer(e, path=config.answer_path)
    if config.dump_eval:
        print("dumping eval ...")
        graph_handler.dump_eval(e, path=config.eval_path)


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    return parser.parse_args()


class Config(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)


def _run():
    args = _get_args()
    with open(args.config_path, 'r') as fh:
        config = Config(**json.load(fh))
        main(config)


if __name__ == "__main__":
    _run()
