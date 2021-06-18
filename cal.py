'''
Created on July 1, 2020
@author: Tinglin Huang (tinglin.huang@zju.edu.cn)
'''
__author__ = "huangtinglin"

import random

import torch
import numpy as np
import os

from time import time
from prettytable import PrettyTable

from utils.parser import parse_args
from utils.data_loader import load_data
from modules.KGIN import Recommender
from utils.evaluate import test
from utils.helper import early_stopping

n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0


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
    entity_pairs = train_entity_pairs[start:end].to(device)
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = torch.LongTensor(negative_sampling(entity_pairs, train_user_set)).to(device)
    return feed_dict


def get_kg_dict(train_kg_pairs, start, end, relation_dict):
    def negative_sampling(kg_pairs, relation_dict):
        neg_ts = []
        for h, r, _ in kg_pairs.cpu().numpy():
            r = int(r)
            h = int(h)
            while True:
                neg_t = np.random.randint(low=0, high=n_entities, size=1)[0]
                if (h, neg_t) not in relation_dict[r]:
                    break
            neg_ts.append(neg_t)
        return neg_ts

    kg_dict = {}
    kg_pairs = train_kg_pairs[start:end].to(device)
    kg_dict['h'] = kg_pairs[:, 0]
    kg_dict['r'] = kg_pairs[:, 1]
    kg_dict['pos_t'] = kg_pairs[:, 2]
    kg_dict['neg_t'] = torch.LongTensor(negative_sampling(kg_pairs, relation_dict)).to(device)
    return kg_dict



if __name__ == '__main__':
    """fix the random seed"""
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    global args, device
    args = parse_args()
    device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")

    """build dataset"""
    train_cf, test_cf, user_dict, n_params, graph, triplets, relation_dict, mat_list = load_data(args)
    adj_mat_list, norm_mat_list, mean_mat_list = mat_list

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']

    """cf data"""
    train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))
    test_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in test_cf], np.int32))

    """kg data"""
    train_kg_pairs = torch.LongTensor(np.array([[kg[0], kg[1], kg[2]] for kg in triplets], np.int32))



