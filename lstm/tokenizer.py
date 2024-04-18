import argparse
import sentencepiece as spm


# 인자값을 받을 수 있는 인스턴스 생성
parser = argparse.ArgumentParser(description='parameter')

# 입력받을 인자값
parser.add_argument('--input', required=False, default='dataset/ptb.train.txt', help='directory of data')
parser.add_argument('--model_prefix', required=False, default='ptb', help='the name of vocabulary model')
parser.add_argument('--vocab_size', required=False, default=1024, help='the vocabulary size')
parser.add_argument('--model_type', required=False, default='unigram', help='the type of tokenizer')
parser.add_argument('--max_sentence_length', required=False, default=9999)

args = parser.parse_args()


# unigram train
spm.SentencePieceTrainer.Train('--input={} --model_prefix={} --vocab_size={} --model_type={} --max_sentence_length={}'
                               .format(args.input, args.model_prefix, args.vocab_size, args.model_type, args.max_sentence_length))
