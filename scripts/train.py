#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""train the word2vec model

input_file: 以 space/tab/eod 作为分词边界
"""

import argparse

import pandas as pd
import numpy as np
import torchtext
import subprocess
import shlex


def main():

    parser = argparse.ArgumentParser(description='Train word2vec model.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--text-data-file", "-tdf",
                        help="use text data from <TEXT_DATA> to train the model, assuming tab/space/EOL spaced tokens")
    parser.add_argument("--alpha", default=0.025,
                        help="Set the starting learning rate: 0.025 suggested for skip-gram;  0.05 suggested for CBOW")
    parser.add_argument("--dim-embedding", "-dm", help="set size of word vectors (dim of embedding)", default=128)
    parser.add_argument("--iter", "-it", default=5, help="Run more training iterations", type=int)
    parser.add_argument("--cbow", help="Use the continuous bag of words model: 0 ~ skip gram; 1 ~ CBOW", default=1)
    parser.add_argument("--window-size", "-ws", help="set max skip length between words", default=5)
    parser.add_argument("--frequency-threshold", "-ft",
                        help="Set threshold for occurrence of words. Those that appear with higher frequency in the training data will be randomly down-sampled, useful range is (0, 1e-5)", default=1e-3)
    parser.add_argument("--use-hierarchical-softmax", "-uhs",
                        help="Use Hierarchical Softmax, 0  not used; 1 use", default=0)
    parser.add_argument("--number-negative", "-nn",
                        help="Number of negative examples, common values are 3 - 10 (0=not used)", default=5)
    parser.add_argument("--threads", default=12, help="Use <THEADS> threads")
    parser.add_argument("--min-count", default=3,
                        help="This will discard words that appears less than <MIN-COUNT> times")
    parser.add_argument("--output-clusters", "-oc",
                        help="number of output clusters: 0 ~ output word vectors, NO kmeans; >0 ~ nubmber of clusters in k-means", default=0, type=int)
    parser.add_argument("--debug", help="Set the debug mode: 2=more info during training", default=2)
    parser.add_argument("--binary", "-b",
                        help="Save the resulting vectors in binary moded: 0 text; 1 binary; 2 both", default=2, type=int)

    # parser.add_argument("--output-file",  "-of",
    #                     help="Use <OUTPUT> to save the resulting word vectors / word clusters", default="/tmp/output")
    # parser.add_argument("--save-vocab", "-sv",
    #                     help="the vocabulary will be saved to <SAVE-VOCAB>. If not setting, save to `text_data_file.vocab`", default=None)
    # parser.add_argument("--read-vocab", "-rv",
    #                     help="The vocabulary will be read from <READ-VOCAB> , not constructed from the training data")

    args = parser.parse_args()

    args.save_vocab = args.text_data_file.replace(".txt", ".vocab")
    if args.output_clusters > 0:
        args.output_file = args.text_data_file.replace(".txt", "_{:04d}.cluster.txt".format(args.iter))
    else:
        if args.binary == 0:
            args.output_file = args.text_data_file.replace(".txt", "_{:04d}.embedding.txt".format(args.iter))
        elif args.binary == 1:
            args.output_file = args.text_data_file.replace(".txt", "_{:04d}.embedding.bin".format(args.iter))
        else:
            args.output_file = args.text_data_file.replace(".txt", "_{:04d}.embedding.txt".format(args.iter))

    cmd = "../bin/word2vec -train {} -output {} -size {} -window {}  -sample {}   -hs {}   -negative {}   -threads {}    -iter {}   -min-count {} -classes {}  -binary {}   -save-vocab {} -cbow {} --alpha {}".format(
        args.text_data_file, args.output_file, args.dim_embedding, args.window_size, args.frequency_threshold, args.use_hierarchical_softmax,
        args.number_negative, args.threads, args.iter, args.min_count, args.output_clusters, args.binary, args.save_vocab, args.cbow, args.alpha
    )
    print(cmd)
    subprocess.run(shlex.split(cmd))
    print("See {} for result".format(args.output_file))


if __name__ == '__main__':
    main()
