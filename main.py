import numpy as np
import bisect
import random
import itertools
from collections import Counter
from matplotlib.pyplot import scatter


# Part 1.1
class Split(object):
    def __init__(self, split_path):
        f = open(split_path, 'r')
        header_line_split = f.readline()  # read the header line
        lines_split = {}
        while True:
            text = f.readline()
            if len(text) == 0:
                break
            lines_split[int(text.split(',')[0])] = int(text.split(',')[1])
        self.lines = lines_split


# Part 1.2
class TTData(object):
    def __init__(self, split, sent_path):

        self.train = []
        self.test = []

        f = open(sent_path, 'r')
        header_line = f.readline()   # read the header line
        lines = {}
        while True:
            text = f.readline()
            if len(text) == 0:
                break
            sent = ' '.join(text.split()[1:])
            sent = sent.lower()     # lower case
            sent = ''.join(ch for ch in sent if (ch.isalnum() or ch is ' '))   # alphanumeric only. removes also non-ascii
            long_words = [word for word in sent.split() if len(word) > 3]   # only long words
            ind = int(text.split()[0])
            lines[ind] = long_words
            self.lines = lines
            self.split = split

            # 	1 = train, 2 = test, 3 = dev
            if split.lines[ind] == 1:
                self.train = self.train + [long_words]
            elif split.lines[ind] == 2:
                self.test = self.test + [long_words]


# Part 2.1
class MHyperparams(object):
    def __init__(self, context_win_sz, vec_rep_sz, num_nc_neg_words, alpha, noise_dist, seed):
        """
        :param context_win_sz: context window size
        :param vec_rep_sz: size of the vector representation of each word
        :param num_nc_neg_words: number of non-context negative words (K)
        :param noise_dist: the choice of noise distribution
        :param seed: random seed
        """
        self.context_win_sz = context_win_sz
        self.vec_rep_sz = vec_rep_sz
        self.num_nc_neg_words = num_nc_neg_words
        self.noise_dist = noise_dist
        self.alpha = alpha
        self.seed = seed


# Part 2.2
class MParams(object):
    def __init__(self, mhyper, voc_sz):
        self.mhyper = mhyper
        self.u = np.zeros((voc_sz, mhyper.vec_rep_sz))
        self.v = np.zeros((voc_sz, mhyper.vec_rep_sz))


# Part 2.3
def Init(data, mhyper, alpha=0.01):

    # sets = [set(train[i]) for i in range(len(train))]
    # vocabulary_train = set.union(*sets)

    vocabulary = list(Counter(list(itertools.chain(*data))).keys())

    mparams = MParams(mhyper, len(vocabulary))

    mparams.u = np.random.multivariate_normal(np.zeros((mhyper.vec_rep_sz)),
                                              0.01 * np.identity(mhyper.vec_rep_sz), len(vocabulary))
    mparams.v = np.random.multivariate_normal(np.zeros((mhyper.vec_rep_sz)),
                                              0.01 * np.identity(mhyper.vec_rep_sz), len(vocabulary))

    mparams.u /= np.linalg.norm(mparams.u, axis=1)[:, np.newaxis]
    mparams.v /= np.linalg.norm(mparams.v, axis=1)[:, np.newaxis]

    # For part 2.4
    counts = list(Counter(list(itertools.chain(*data))).values())
    # I'M DOING THE CALCULATION OF THE CUMULATIVE DISTRIBUTION ONCE TO SAVE CPU
    counts = np.array(counts) / np.sum(np.array(counts))
    counts = counts ** alpha / sum(counts ** alpha)
    counts = np.cumsum(counts)

    return mparams, vocabulary, counts


# Part 2.4
def sample_from_unigram(counts):
    uni_rand = np.random.uniform(0, 1)
    ind = bisect.bisect_left(counts, uni_rand)
    return ind


# part 2.5
def sample_random_words(k, counts):
    return [sample_from_unigram(counts) for i in range(k)]


def sigma(z):
    return 1.0 / (1.0 + (np.e ** (-1.0 * z)))


# part 2.6
def log_probability_context_word(context_word_ind, input_word_ind, neg_sample_inds, mparams):

    u_i = mparams.u[input_word_ind]
    v_c = mparams.v[context_word_ind]

    t1 = np.log(sigma(np.dot(np.transpose(u_i), v_c)))

    t2 = 0

    for indx in neg_sample_inds:
        v_k = mparams.v[indx]
        t2 += np.log(1 - sigma(np.dot(np.transpose(u_i),v_k)))

    return t1 + t2



# Part 3 + 4
class AlgHyperparams(object):
    def __init__(self, lr, minibatch_size, num_of_iter):
        """
        :param lr: learning rate
        :param minibatch_size: the mini batch size which will be used to update the parameters
        :param num_of_iter: num of iteration to run all the train data minibatches
        :param lr_update_iter: multiply by 0.5 every N number of iterations
        :param lr_update_iter: multiply by 0.5 every N number of iterations
        """
        self.lr = lr
        self.minibatch_size = minibatch_size
        self.num_of_iterations = num_of_iter
        self.lr_update_iter = int(num_of_iter*0.25)
        self.eval_iter = 10


def create_minibatches(X_train, word_map, mhparams, ahparams):
    pairs = []
    minibatches = []
    random.shuffle(X_train)
    for sentence in X_train:
        if len(sentence) >= 2 * mhparams.context_win_sz + 1:
            for token_idx, token in enumerate(sentence[mhparams.context_win_sz:len(X_train)-mhparams.context_win_sz]):
                real_token_idx = token_idx + mhparams.context_win_sz
                context_start = token_idx
                context_end = real_token_idx + mhparams.context_win_sz + 1
                context = [word_map[s] for s in sentence[context_start:real_token_idx]] + \
                          [word_map[e] for e in sentence[real_token_idx + 1:context_end]]
                target = word_map[token]
                pairs.append((target, context))
                if len(pairs) == ahparams.minibatch_size:
                    minibatches.append(pairs)
                    pairs = []
    return minibatches



def eval_mb(mb, mhparams, uv_params, iter):
    loglike = 0.0
    for target, context in mb:
        for context_word in context:
            k_samples = sample_random_words(mhparams.num_nc_neg_words, noise_distribution)
            loglike += log_probability_context_word(context_word, target, k_samples, uv_params)
    # print("[%d] done, training log-likelihood = %f" % (iter, train_loglike))
    return loglike



def eval_model(mb_test, mhparams, uv_params, iter):
    test_loglike = 0.0  # For evaluation
    for mb_idx, mb in enumerate(mb_test):
        for target, context in mb:
            for context_word in context:
                k_samples = sample_random_words(mhparams.num_nc_neg_words, noise_distribution)
                test_loglike += log_probability_context_word(context_word, target, k_samples, uv_params)
    # print("[%d] done, test log-likelihood = %f" % (iter, test_loglike))
    return test_loglike / mb_idx



def LearnParamsUsingSGD(mb, uv_params, ahparams, mhparams, noise_distribution, iter):
    # for mb_idx, mb in enumerate(mb_train):
    context_grad = np.zeros(uv_params.v.shape)
    target_grad = np.zeros(uv_params.u.shape)
    for target, context in mb:
        for context_word in context:
            k_samples = sample_random_words(mhparams.num_nc_neg_words, noise_distribution)
            cls = [(context_word, 1)] + [(w, 0) for w in k_samples]

            for neg_word, ind in cls:
                z = np.dot(uv_params.v[neg_word], uv_params.u[target])
                p = sigma(z)
                g = ind - p
                context_grad[neg_word] += g * uv_params.u[target]  # Error to backpropagate to v_c and v_nj
                target_grad[target] += g * uv_params.v[neg_word]  # Error to backpropagate to u_t
    # Update context grads
    uv_params.v += ahparams.lr * context_grad
    uv_params.u += ahparams.lr * target_grad
    # Evaluate for a minibatch
    train_loglike = eval_mb(mb, mhparams, uv_params, iter)
    return train_loglike




# Part 5

#5.1
def predict_num_most_likely_context(n, uv_params, mhparams, word):
    input = word_map[word]
    k_samples = sample_random_words(mhparams.num_nc_neg_words, noise_distribution)
    context_argmax = np.array([log_probability_context_word(ctx, input, k_samples, uv_params) for ctx in range(uv_params.v.shape[0])])
    context_argmax = [context_argmax.argsort()[::-1][:n]][0]
    return [index_map[i] for i in context_argmax]


#5.2
def predict_num_most_likely_input(n, uv_params, mhparams, ctx_list):
    k_samples = sample_random_words(mhparams.num_nc_neg_words, noise_distribution)
    context_arr_prob = 0.0
    target_argmax = -1
    for idx, target in enumerate(uv_params.u):
        context_arr_prob += np.sum(
            [log_probability_context_word(ctx, idx, k_samples, uv_params) for ctx in ctx_list])
        target_argmax = context_arr_prob if context_arr_prob > target_argmax else target_argmax


# 5.3
def scatter_plot_words(uv_params, inputs):
    inputs_idx = [word_map[w] for w in inputs]
    x = [u[0] for idx, u in enumerate(uv_params.u) if idx in inputs_idx]
    y = [u[1] for idx, u in enumerate(uv_params.u) if idx in inputs_idx]
    scatter(x, y, marker=">")


# 5.4
def print_most_num_likely_analogy(n, inputs):
    pred = analogy_solver(n, mparams, inputs[0:3])
    print("Most %d likely analogy words for the sequence:" % (n))
    print(*inputs, sep=", ")
    print('is: ')
    print(*pred, sep=", ")


def analogy_solver(n, uv_params, parts):
    u_1 = uv_params.u[word_map[parts[0]]]
    u_2 = uv_params.u[word_map[parts[1]]]
    u_3 = uv_params.u[word_map[parts[2]]]
    analogy_argmax = np.array([np.dot(u_i.T, (u_1 - u_2 + u_3)) for u_i in uv_params.u])
    analogy_argmax = analogy_argmax.argsort()[::-1][:n]
    return [index_map[i] for i in analogy_argmax]



# Usage examples

# Parts 1.1, 1.2
split = Split('datasetSplit.txt')
tt_data = TTData(split, 'datasetSentences.txt')
line_num = 1
print('Parts 1.1, 1.2')
print('Line: ', line_num, 'Words: ', tt_data.lines[line_num], 'Split: ', tt_data.split.lines[line_num])
line_train_test = 0
print('Train/test line num:', line_train_test,
      'train first word:', '"{0}",'.format(tt_data.train[line_train_test][0]),
      'test first word:', '"{0}"'.format(tt_data.test[line_train_test][0]))

# Part 2.1
# TODO: VERIFY THE PARAMS' VALUES
mhyper = MHyperparams(context_win_sz=5, vec_rep_sz=50, num_nc_neg_words=10, alpha=0.0,
                       noise_dist='Unigram', seed=1)

# Parts 2.2, 2.3
mparams, vocabulary, noise_distribution = Init(tt_data.train + tt_data.test, mhyper, )
print('Parts 2.1, 2.2, 2.3')
print('u and v were initialized and normalized:', np.linalg.norm(mparams.u[0]), np.linalg.norm(mparams.v[0]))

# Part 2.4
ind = sample_from_unigram(noise_distribution)
print('Part 2.4')
print('word that was randomly chosen:', '"{0}"'.format(vocabulary[ind]))

# Part 2.5
nc_neg_indcs = sample_random_words(mhyper.num_nc_neg_words, noise_distribution)
print('Part 2.5')
print('K words that were randomly chosen:', [vocabulary[i] for i in nc_neg_indcs])

# Part 2.6
context_word_ind = 100
input_word_ind = 200
log_prob_wc_wi = log_probability_context_word(context_word_ind, input_word_ind, nc_neg_indcs, mparams)
print('Part 2.6')
print('log probability of context word:', '"{0}",'.format(vocabulary[context_word_ind]),
      'given input word:', '"{0}",'.format(vocabulary[input_word_ind]), 'is:', log_prob_wc_wi)


# Part 3+4
scores = [] # Will hold training and test mean log-likelihood for every iteration
word_map = {word : idx for idx, word in enumerate(vocabulary)}
index_map = {v: k for k, v in word_map.items()}
ahparams = AlgHyperparams(lr=0.05, minibatch_size=50, num_of_iter=5000)
print("\n\n\t\t Starting trianing the model")

# Main loop
print("Creating shuffled minibatches")
mb_train = create_minibatches(tt_data.train, word_map, mhyper, ahparams)
mb_test = create_minibatches(tt_data.test, word_map, mhyper, ahparams)
train_size = len(mb_train)
train_loglike = 0.0
for iter in range(ahparams.num_of_iterations):
    # print("[iter %d] Applying SGD on training data" % (iter))
    mb_idx = np.random.randint(train_size)
    train_loglike += LearnParamsUsingSGD(mb_train[mb_idx], mparams, ahparams, mhyper, noise_distribution, iter)

    if iter % ahparams.eval_iter == 0:
        train_loglike /= 1.0 if iter == 0 else ahparams.eval_iter
        print("[iter %d] Train mini-batch mean log-likelihood = %f" % (iter, train_loglike))
        train_loglike = 0.0
        test_loglike = eval_model(mb_test, mhyper, mparams, iter)
        print("[iter %d] Test mean log-likelihood = %f" % (iter, test_loglike))
        print_most_num_likely_analogy(5, ['apple', 'eating', 'juice'])
        scores.append((train_loglike, test_loglike))
        np.savez('model_save/model_iter_%d' % (iter), mparams.u, mparams.v)
    if iter != 0 and iter % ahparams.lr_update_iter == 0:
        ahparams.lr *= 0.5
        print("Leraning rate was deacreased by half. old=%f, new=%f" % (ahparams.lr * 2, ahparams.lr))




# Part 5
n = 10
word = 'king'
print("Most %d likely context words, given input %s" % (n, word))
pred = predict_num_most_likely_context(n, mparams, mhyper, word)
print(*pred, sep = ", ")
inputs = ['king', 'queen', 'husband', 'wife']
scatter_plot_words(mparams, inputs)
print_most_num_likely_analogy(n, inputs)

