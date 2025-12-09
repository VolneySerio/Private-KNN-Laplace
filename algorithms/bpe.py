from collections import defaultdict




def learn_bpe(corpus, num_merges):

    vocab = defaultdict(int)

    for sentence in corpus.split('.').