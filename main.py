'''
Created on July 1, 2020
@author: Tinglin Huang (tinglin.huang@zju.edu.cn)
'''
__author__ = "huangtinglin"

import random
import os
import torch
import numpy as np

from time import time
from prettytable import PrettyTable

from utils.parser import parse_args
from utils.data_loader import load_data
from modules.KGIN import Recommender
from utils.evaluate import test
from utils.helper import early_stopping
import pickle

n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0

def save_variable(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()
    return filename

def load_variavle(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r

def get_feed_dict(train_entity_pairs, start, end, train_user_set):

    def negative_sampling(user_item, train_user_set):
        neg_items = []
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            while True:
                neg_item = np.random.randint(low=0, high=n_items, size=1)[0]
                if neg_item not in train_user_set[user]:
                    break
            neg_items.append(neg_item)
        return neg_items

    feed_dict = {}
    # entity_pairs = train_entity_pairs[start:end].to(device)
    entity_pairs = train_entity_pairs[start:end].cpu()
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    # feed_dict['neg_items'] = torch.LongTensor(negative_sampling(entity_pairs, train_user_set)).to(device)
    feed_dict['neg_items'] = torch.LongTensor(negative_sampling(entity_pairs, train_user_set)).cpu()
    return feed_dict


if __name__ == '__main__':
    """fix the random seed"""
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    """read args"""
    global args, device
    args = parse_args()
    device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")

    """build dataset"""
    train_cf, test_cf, user_dict, n_params, graph, mat_list = load_data(args)
    adj_mat_list, norm_mat_list, mean_mat_list = mat_list

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']

    """cf data"""
    train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))
    test_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in test_cf], np.int32))

    """define model"""
    model = Recommender(n_params, args, graph, mean_mat_list[0]).to(device)

    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    cur_best_pre_0 = 0
    stopping_step = 0
    should_stop = False

    recall, ndcg, precision, hit_ratio = [], [], [], []
    for epoch in range(210):
        # """training CF"""
        # shuffle training data
        # index = np.arange(len(train_cf))
        # np.random.shuffle(index)
        # train_cf_pairs = train_cf_pairs[index]
        #
        # """training"""
        # loss, s, cor_loss = 0, 0, 0
        # train_s_t = time()
        # while s + args.batch_size <= len(train_cf):
        #     batch = get_feed_dict(train_cf_pairs,
        #                           s, s + args.batch_size,
        #                           user_dict['train_user_set'])
        #     batch_loss, _, _, batch_cor = model(batch)
        #
        #     batch_loss = batch_loss
        #     optimizer.zero_grad()
        #     batch_loss.backward()
        #     optimizer.step()
        #
        #     loss += batch_loss
        #     cor_loss += batch_cor
        #     s += args.batch_size
        #
        # train_e_t = time()

        if epoch % 10 == 9 or epoch == 1:
            """testing"""
            # state_model = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            # torch.save(state_model, os.path.join(os.getcwd(), "result", 'epoch-' + str(epoch) + '.model'))
            load_dir = os.path.join(os.getcwd(), "result", 'last-fm' + 'epoch-' + str(epoch) + '.model')
            # use cpu to load model
            last_model = torch.load(load_dir, map_location='cpu')
            # use gpu to load model
            # last_model = torch.load(load_dir)
            model.load_state_dict(last_model['model_state_dict'])
            optimizer.load_state_dict(last_model['optimizer_state_dict'])

            test_s_t = time()
            ret = test(model, user_dict, n_params)
            test_e_t = time()

            recall.append(ret['recall'])
            ndcg.append(ret['ndcg'])
            precision.append(ret['precision'])
            hit_ratio.append(ret['hit_ratio'])

            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "tesing time", "recall", "ndcg", "precision", "hit_ratio"]
            train_res.add_row(
                [epoch, test_e_t - test_s_t, ret['recall'], ret['ndcg'], ret['precision'], ret['hit_ratio']]
            )
            print(train_res)
    recall = save_variable(recall, 'recall.txt')
    ndcg = save_variable(ndcg, 'ndcg.txt')
    precision = save_variable(precision, 'precision.txt')
    hit_ratio = save_variable(hit_ratio, 'hit_ratio.txt')
    # results = load_variavle('results.txt')
    print("\n The end")