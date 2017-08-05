# -*- coding: utf-8 -*-
# import spacy
import numpy as np
import logging
import warnings
# from spacy.tokens import Doc
import nltk

logger = logging.getLogger(__name__)


POS_TAGS = ['NN','NNS','NNP','NNPS','PRP$','PRP','WP','WP$',\
			'MD','VB','VBD','VBG','VBN','VBP','VBZ','JJ','JJR',\
			'JJS','RB','RBR','RBS','WRB','DT','PDT','WDT','SYM',\
			'POS','LRB','RRB',',','-',':',';','.','``','"','$','CD',\
			'DAT','CC','EX','FW','IN','RP','TO','UH','URL','USER',\
			'EMAIL','NNPOS','UR',':)','UH','#','!!!','...','\'\'','(',')','LS']

INT_POS_TAGS = list(range(len(POS_TAGS)))



def pos_from_text(text):
	# doc=nlp(text)
	# POS = list(map(lambda x:x.tag_,doc))
	str_text = [(word) for word in text if len(word) >0]
	pos_tuple = nltk.pos_tag(str_text)

	POS = [e[1] for e in nltk.pos_tag(str_text)]
	if len(POS) != len(text):
		print('error - POS length : {} , text length : {}'.format(len(POS),len(text)))
		print('text : {}'.format(text))
		print('Part-of-speech : {}'.format(POS))
		raise 
	return POS

def one_hotify(initial_labels,classes = POS_TAGS):
	number_of_classes = len(classes)
	number_of_samples = len(initial_labels)
	indexed_labels = list(map(lambda x: int(classes.index(x)), initial_labels))
	one_hot_array = np.zeros([number_of_samples,number_of_classes])
	one_hot_array[tuple(range(number_of_samples)), tuple(indexed_labels)] = 1
	return one_hot_array.tolist()

def get_pos(corpus):
	# nlp = load_nlp_model(language=language,tokenizer=None)
	POS_corpus = list(map(lambda x:pos_from_text(x), corpus))
	return POS_corpus

def get_pos_one_hot(corpus):
	POS_corpus = list(map(lambda x:one_hotify(pos_from_text(x)), corpus))
	return POS_corpus


def get_pos_one_hot_simple(text_as_list):
	POS_list = one_hotify(pos_from_text(" ".join(text_as_list)))
	return POS_list

def get_pos_dimension():
	return len(POS_TAGS)

def get_pos_index(corpus):
	# nlp = load_nlp_model(language=language,tokenizer=None)
	POS_corpus = list(map(lambda x:integrify(pos_from_text(x)), corpus))
	return POS_corpus

def get_pos_index_pad(corpus,max_len):
	# nlp = load_nlp_model(language=language,tokenizer=None)
	POS_corpus = list(map(lambda x:integrify(pos_from_text(x)), corpus))
	POS_padded = pad_pos(POS_corpus)
	return POS_padded

def get_pos_pad(corpus,max_len, one_hot=True):
	# nlp = load_nlp_model(language=language,tokenizer=None)
	if one_hot:
		POS_corpus = get_pos_one_hot(corpus)
	else:
		POS_corpus = get_pos_index(corpus)
	POS_padded = pad_pos(POS_corpus)
	return POS_corpus, POS_padded

def save_pos(corpus, save_path, one_hot=True):
	logger.info('processing support part_of_speech for saving')
	support_to_save = flatten_corpus(corpus['support'],one_hot=one_hot)
	logger.info('saving support part_of_speech')
	with open(save_path + '_support.txt', 'w') as support_file:
		support_file.write(support_to_save)
	logger.info('support part_of_speech_saved into {}'.format(save_path + '_support.txt'))

	logger.info('processing question part_of_speech for saving')
	question_to_save = flatten_corpus(corpus['question'],one_hot=one_hot)
	logger.info('saving question part_of_speech')
	with open(save_path + '_question.txt', 'w') as question_file:
		question_file.write(question_to_save)
	logger.info('question part_of_speech_saved into {}'.format(save_path + '_question.txt'))

def flatten_corpus(corpus, one_hot=True):
	flattened_corpus = "\n".join(list(map(lambda x:flatten_text(x,one_hot=one_hot) , corpus)))	
	return flattened_corpus

def flatten_text(text, one_hot=True):
	if one_hot:
		flatten_text = " ".join(list(map(lambda x:str(np.argmax(x)),text)))
	else:
		flatten_text = " ".join(text)
	return flatten_text

def load_pos(save_path,one_hot=True):
	corpus = {"support": [], "question": []} 
	logger.info('loading support_file : {}'.format(save_path + '_support.txt'))
	with open(save_path + '_support.txt', 'r') as support_file:
		support_to_parse = support_file.read()
	corpus['support'] = load_corpus(support_to_parse)

	logger.info('loading question_file : {}'.format(save_path + '_question.txt'))
	with open(save_path + '_question.txt', 'r') as question_file:
		question_to_parse = question_file.read()
	corpus['question'] = load_corpus(question_to_parse,one_hot=one_hot)

	return corpus

def load_corpus(text,one_hot=True):
	parsed_corpus = [load_text(sub_text,one_hot=one_hot) for sub_text in text.split('\n') if len(sub_text) != 0]
	return parsed_corpus

def load_text(text,one_hot=True):
	parsed_text = [int(value) for value in text.split(" ") if len(value) != 0]
	if one_hot:
		return one_hotify(parsed_text, classes = INT_POS_TAGS)
	else:
		return parsed_text


def integerify(initial_labels,classes = POS_TAGS):
	number_of_classes = len(classes)
	number_of_samples = len(initial_labels)
	indexed_labels = list(map(lambda x: int(classes.index(x)), initial_labels))
	return indexed_labels



def pad_pos(corpus,max_len):
	padded_corpus = np.array(map(lambda x:pad_sentence(x,max_len),corpus))
	return padded_corpus

def pad_sentence(sent,max_len):
	if len(sent)>=max_len:
		return sent[0:max_len]
	else:
		if isinstance(sent[-1], int):
			new_sent = sent + [0]*int(max_len - len(sent))
		else:
			temp_array = np.zeros(get_pos_dimension()).astype(int)
			temp_array[0] = 1
			new_sent = sent + [temp_array]*int(max_len - len(sent))
		return sent

