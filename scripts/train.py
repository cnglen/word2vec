#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""train the word2vec model"""

import argparse

import pandas as pd
import numpy as np
import torchtext
import subprocess
import shlex


def tokenize_to_char(ifile):

    text_field = torchtext.data.Field(lower=True, tokenize=list)
    raw_data = pd.read_csv(ifile, names=["text"], sep="\001")
    raw_data["preprocessed_text"] = raw_data.text.apply(lambda x: " ".join(text_field.preprocess(x.strip())))
    raw_data["preprocessed_text"].to_csv(ifile.replace(".txt", "_char.txt"), index=False, header=False)


def main():

    parser = argparse.ArgumentParser(description='Train word2vec model.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train", help="use text data from <TRAIN> to train the model")
    parser.add_argument(
        "--output", help="Use <OUTPUT> to save the resulting word vectors / word clusters", default="/tmp/output")
    parser.add_argument("--size", help="set sie of word vectors", default=100)
    parser.add_argument("--window", help="set max skip length between words", default=5)
    parser.add_argument(
        "--sample", help="Set threshold for occurrence of words. Those that appear with higher frequency in the training data will be randomly down-sampled, useful range is (0, 1e-5)", default=1e-3)
    parser.add_argument("-hs", help="Use Hierarchical Softmax, 0  not used; 1 use", default=0)
    parser.add_argument(
        "--negative", help="Number of negative examples, common values are 3 - 10 (0=not used)", default=5)
    parser.add_argument("--threads", default=12, help="Use <THEADS> threads")
    parser.add_argument("--iter", default=5, help="Run more training iterations")
    parser.add_argument("--min-count", default=5,
                        help="This will discard words that appears less than <MIN-COUNT> times")
    parser.add_argument(
        "--classes", help="Output word classes rather than word vectors default number of classes is 0 (vectors are written)", default=0)

    parser.add_argument("--debug", help="Set the debug mode(2=more info during training)", default=2)
    parser.add_argument("--binary", help="Save the resulting vectors in binary moded: 0 (off)", default=0)
    parser.add_argument("--save-vocab", help="the vocabulary willbe saved to <SAVE-VOCAB>", default="/tmp/vocab")
    parser.add_argument(
        "--read-vocab", help="The vocabulary will be read from <READ-VOCAB> , not constructed from the training data")
    parser.add_argument("--cbow", help="Use the continuous bag of words model: 0~skip gram; 1 ~ CBOS", default=1)
    parser.add_argument("--alpha", default=None,
                        help="Set the starting learning rate, default is 0.025 for skip-gram and 0.05 for CBOW")

    args = parser.parse_args()

    if args.cbow:
        args.alpha = 0.05
    else:
        args.alhpa = 0.025

    cmd = "../bin/word2vec -train {} -output {} -size {} -window {}    -sample {}   -hs {}   -negative {}   -threads {}    -iter {}   -min-count {} --classes {}  -binary {}   -save-vocab {} -cbow {} ".format(
        args.train, args.output, args.size, args.window, args.sample, args.hs, args.negative, args.threads, args.iter, args.min_count, args.classes, args.binary, args.save_vocab, args.cbow
    )

    subprocess.run(shlex.split(cmd))


if __name__ == '__main__':
    main()
