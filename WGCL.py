# -*- coding: utf-8 -*-
'''
Created on 09 13 14:34:23 2024

@author: Jeffrey Taylor
'''
import torch
from torch import nn
import scipy.sparse as sp
import random
import numpy as np
import logging
import torch.optim as optim
import multiprocessing as mp
import os
from time import strftime
from time import localtime
import argparse
import reckit
from RankingMetrics import *


class WGCL(nn.Module):
    def __init__(self, data_train, data_test, num_users, num_items, embedding_size, temperature,
                 layer, lr, batch_size, eps, noise_type):
        super(WGCL, self).__init__()
        self.train_data = data_train
        self.test_data = data_test
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.n_layers = layer
        self.learning_rate = lr
        self.temperature = temperature
        self.batch_size = batch_size
        self.eps = eps
        self.noise_type = noise_type
        self.train_ui_dict = self.get_ui_dict(self.train_data)
        self.test_ui_dict = self.get_ui_dict(self.test_data)
        self.present_train_size = 0

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_size)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_size)
        self.embedding_user_final = None
        self.embedding_item_final = None

        nn.init.normal_(self.embedding_user.weight, 0, 0.01)
        nn.init.normal_(self.embedding_item.weight, 0, 0.01)

        self.Graph_ui = self.getSparseGraph()

    def getSparseGraph(self):
        Graph_ui = None
        device = 'cuda'
        UserItemMat = sp.dok_matrix((self.num_users, self.num_items), dtype=np.float32)
        adj_mat_ui = sp.dok_matrix((self.num_users + self.num_items, self.num_users + self.num_items), dtype=np.float32)
        adj_mat_ui = adj_mat_ui.tolil()
        R_ui = UserItemMat.tolil()
        for data in self.train_data:
            user, item = data
            R_ui[user, item] = 1
        adj_mat_ui[:self.num_users, self.num_users:] = R_ui
        adj_mat_ui[self.num_users:, :self.num_users] = R_ui.T
        adj_mat_ui = adj_mat_ui.todok()

        norm_adj_ui = self.norm_adj_single(adj_mat_ui)
        Graph_ui = self._convert_sp_mat_to_sp_tensor(norm_adj_ui)
        Graph_ui = Graph_ui.coalesce().to(device)
        return Graph_ui

    def norm_adj_single(self, adj_mat):
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()
        return norm_adj

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def forward(self, users, pos_items, neg_items):
        all_users, all_items = self.computer(self.Graph_ui)
        users_emb = all_users[users]
        pos_items_emb = all_items[pos_items]
        neg_items_emb = all_items[neg_items]
        pos_scores = torch.sum(torch.mul(users_emb, pos_items_emb), dim=1)
        neg_scores = torch.sum(torch.mul(users_emb, neg_items_emb), dim=1)

        self.embedding_user_final = all_users.detach()
        self.embedding_item_final = all_items.detach()

        all_users_1, all_items_1 = self.noise_computer(self.Graph_ui, self.noise_type[0])
        all_users_2, all_items_2 = self.noise_computer(self.Graph_ui, self.noise_type[1])

        users_emb_1 = nn.functional.normalize(all_users_1, dim=1)
        users_emb_2 = nn.functional.normalize(all_users_2, dim=1)
        items_emb_1 = nn.functional.normalize(all_items_1, dim=1)
        items_emb_2 = nn.functional.normalize(all_items_2, dim=1)

        user_emb_1 = users_emb_1[users]
        user_emb_2 = users_emb_2[users]
        item_emb_1 = items_emb_1[pos_items]
        item_emb_2 = items_emb_2[pos_items]

        pos_ratings_user = torch.exp(
            torch.sum(torch.mul(user_emb_1, user_emb_2), dim=1) / self.temperature)  # [batch_size]
        tot_ratings_user = torch.matmul(user_emb_1, torch.transpose(users_emb_2, 0, 1))  # [batch_size,num_user]
        tot_ratings_user = torch.sum(torch.exp(tot_ratings_user / self.temperature), dim=1)  # [batch_size]
        ssl_loss_user = - torch.sum(torch.log(pos_ratings_user / tot_ratings_user))

        pos_ratings_item = torch.exp(torch.sum(torch.mul(item_emb_1, item_emb_2), dim=1) / self.temperature)
        tot_ratings_item = torch.matmul(item_emb_1, torch.transpose(items_emb_2, 0, 1))  # [batch_size,num_item]
        tot_ratings_item = torch.sum(torch.exp(tot_ratings_item / self.temperature), dim=1)  # [batch_size]
        ssl_loss_item = - torch.sum(torch.log(pos_ratings_item / tot_ratings_item))

        wass_loss = self.lwass(user_emb_1, user_emb_2) + self.lwass(item_emb_1, item_emb_2)

        return pos_scores - neg_scores, ssl_loss_user + ssl_loss_item, wass_loss

    def computer(self, Graph):
        users_emb = self.embedding_user.weight
        item_emb = self.embedding_item.weight

        all_emb = torch.cat([users_emb, item_emb])
        embs = []
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(Graph, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def noise_computer(self, Graph, noise_type):
        users_emb = self.embedding_user.weight
        item_emb = self.embedding_item.weight
        random_noise = None
        all_emb = torch.cat([users_emb, item_emb])
        embs = []

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(Graph, all_emb)
            if noise_type == 'uniform_noise':
                random_noise = torch.rand(all_emb.shape)
            if noise_type == 'Gaussian_noise':
                random_noise = torch.randn(size=all_emb.shape)
            random_noise = random_noise.cuda()
            all_emb += torch.mul(torch.sign(all_emb), nn.functional.normalize(random_noise)) * self.eps
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def get_ui_dict(self, data):
        res = {}
        for i in data:
            if i[0] not in res.keys():
                res[i[0]] = []
            res[i[0]].append(i[1])
        return res

    def get_train_batch(self):
        len_data = len(self.train_data)
        if self.present_train_size + self.batch_size > len_data - 1:
            res = self.train_data[self.present_train_size:len_data] + \
                  self.train_data[0:self.present_train_size + self.batch_size - len_data]
        else:
            res = self.train_data[self.present_train_size:self.present_train_size + self.batch_size]
        self.present_train_size += self.batch_size
        self.present_train_size %= len_data
        return res

    def get_feed_dict(self, data):
        user_list = []
        pos_item_list = []
        neg_item_list = []
        for item in data:
            nt = random.sample(range(self.num_items), 1)[0]
            while nt in self.train_ui_dict[item[0]]:
                nt = random.sample(range(self.num_items), 1)[0]
            user_list.append(item[0])
            pos_item_list.append(item[1])
            neg_item_list.append(nt)
        return user_list, pos_item_list, neg_item_list

    def save_model(self, model, args):
        code_name = os.path.basename(__file__).split('.')[0]
        log_path = "model/{}/".format(code_name)
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        if self.embedding_user_final is None:
            print("Error: self.embedding_user_final is None")
        if self.embedding_item_final is None:
            print("Error: self.embedding_item_final is None")

        state_dict = {
            'model': model.state_dict(),
            'embedding_user_final': self.embedding_user_final,
            'embedding_item_final': self.embedding_item_final
        }
        torch.save(state_dict, log_path + \
                   "%s_embed_size%d_lr%0.5f_gamma%.5f_layer%d.pth" % \
                   (args.dataset,
                    args.embedding_size,
                    args.lr,
                    args.gamma,
                    args.layer))


class evaluate:
    def __init__(self, train_ui_dict, test_ui_dic, user_emb, item_emb):
        self.test_ui_dict = test_ui_dic
        self.train_ui_dict = train_ui_dict
        self.user_emb = user_emb
        self.item_emb = item_emb

    def predict(self, user):
        user = torch.from_numpy(np.array(user)).long().to('cuda')
        user_id = self.user_emb[user]
        pred = torch.matmul(user_id, self.item_emb.T)
        return pred

    def evaluator(self, user):
        len_train_list = len(self.train_ui_dict[user])
        rank_list = []
        pred = self.predict(user)
        pred = pred.cpu().detach().numpy()
        tmp = reckit.arg_top_k(pred, 20 + len_train_list)
        for i in tmp:
            if i in self.train_ui_dict[user]:
                continue
            rank_list.append(i)
        test_list = self.test_ui_dict[user]

        p_3, r_3, ndcg_3 = precision_recall_ndcg_at_k(3, rank_list[:3], test_list)
        p_5, r_5, ndcg_5 = precision_recall_ndcg_at_k(5, rank_list[:5], test_list)
        p_10, r_10, ndcg_10 = precision_recall_ndcg_at_k(10, rank_list[:10], test_list)
        p_20, r_20, ndcg_20 = precision_recall_ndcg_at_k(20, rank_list[:20], test_list)
        return [p_3, p_5, p_10, p_20, r_3, r_5, r_10, r_20, ndcg_3, ndcg_5, ndcg_10, ndcg_20]


def load_ui(path):
    num_users = -1
    num_items = -1
    data = []
    with open(path) as f:
        for line in f:
            line = [int(i) for i in line.split('\t')[:2]]
            data.append(line)
            num_users = max(line[0], num_users)
            num_items = max(line[1], num_items)
    num_users, num_items, = num_users + 1, num_items + 1
    return data, num_users, num_items


def load_data(path):
    print('Loading train and test data...', end='')
    data_train, num_users, num_items = load_ui(path + '.train')
    data_test, num_users2, num_items2 = load_ui(path + '.test')
    num_users = max(num_users, num_users2)
    num_items = max(num_items, num_items2)
    print('Done.')
    print()
    print('Number of users: %d' % num_users)
    print('Number of items: %d' % num_items)
    print('Number of train data: %d' % len(data_train))
    print('Number of test data: %d' % len(data_test))

    logging.info('Number of users: %d' % num_users)
    logging.info('Number of items: %d' % num_items)
    logging.info('Number of train data: %d' % len(data_train))
    logging.info('Number of test data: %d' % len(data_test))

    return data_train, data_test, num_users, num_items


def train(train_data, test_data, num_users, num_items):
    device = torch.device('cuda')
    batch_total = int(len(train_data) / args.batch_size)
    wgcl = WGCL(data_train, data_test, num_users, num_items, args.embedding_size, args.temperature, \
                   args.layer, args.lr, args.batch_size, args.eps, args.noise_type).to(device)

    optimizer = optim.Adam(wgcl.parameters(), lr=args.lr, weight_decay=args.reg_rate)

    history_p_at_3 = []
    history_p_at_5 = []
    history_p_at_10 = []
    history_p_at_20 = []
    history_r_at_3 = []
    history_r_at_5 = []
    history_r_at_10 = []
    history_r_at_20 = []
    history_ndcg_at_3 = []
    history_ndcg_at_5 = []
    history_ndcg_at_10 = []
    history_ndcg_at_20 = []
    best_pre_5 = [0] * 12
    for epoch in range(args.epochs):
        wgcl.train()
        sim_gcl_loss = 0
        for k in range(1, batch_total + 1):
            user_list, pos_item_list, neg_item_list = wgcl.get_feed_dict(wgcl.get_train_batch())
            user_id = torch.from_numpy(np.array(user_list)).long().to(device)
            pos_item_id = torch.from_numpy(np.array(pos_item_list)).long().to(device)
            neg_item_id = torch.from_numpy(np.array(neg_item_list)).long().to(device)
            pred, ssl_loss, wass_loss = wgcl(user_id, pos_item_id, neg_item_id)

            batch_loss = -torch.log(torch.sigmoid(pred)).sum() + args.ssl_reg * ssl_loss + args.gamma * wass_loss
            sim_gcl_loss += batch_loss
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        if (epoch + 1) % args.verbose == 0:
            wgcl.eval()
            with torch.no_grad():
                evaluation = evaluate(wgcl.train_ui_dict, wgcl.test_ui_dict, \
                                      wgcl.embedding_user_final, wgcl.embedding_item_final)
                user_id = [key for key in wgcl.test_ui_dict.keys()]
                pool = mp.Pool(processes=1)
                res = pool.map(evaluation.evaluator, user_id)
                pool.close()
                pool.join()
                res = np.array(res)
                res = np.mean(res, axis=0)

                history_p_at_3.append(res[0])
                history_p_at_5.append(res[1])
                history_p_at_10.append(res[2])
                history_p_at_20.append(res[3])
                history_r_at_3.append(res[4])
                history_r_at_5.append(res[5])
                history_r_at_10.append(res[6])
                history_r_at_20.append(res[7])
                history_ndcg_at_3.append(res[8])
                history_ndcg_at_5.append(res[9])
                history_ndcg_at_10.append(res[10])
                history_ndcg_at_20.append(res[11])
                if res[11] > best_pre_5[11]:
                    best_pre_5 = res
                    wgcl.save_model(wgcl, args)
                    print("Find the best, save model.")
                print(
                    " %04d Loss: %.2f \t pre3: %.5f  rec3: %.5f  ndcg3: %.5f  pre5: %.5f  rec5: %.5f ndcg5: %.5f  pre10:  %.5f  rec10:  %.5f ndcg10:  %.5f  pre20:  %.5f  rec20:  %.5f ndcg20:  %.5f" % \
                    (epoch + 1, sim_gcl_loss, res[0], res[4], res[8], res[1], res[5], res[9], res[2], res[6], res[10],
                     res[3], res[7], res[11]))
                logging.info(
                    " %04d Loss: %.2f \t pre3: %.5f  rec3: %.5f  ndcg3: %.5f  pre5: %.5f  rec5: %.5f ndcg5: %.5f  pre10:  %.5f  rec10:  %.5f ndcg10:  %.5f  pre20:  %.5f  rec20:  %.5f ndcg20:  %.5f" % \
                    (epoch + 1, sim_gcl_loss, res[0], res[4], res[8], res[1], res[5], res[9], res[2], res[6], res[10],
                     res[3], res[7], res[11]))

    best_ndcg20_index = np.argmax(history_ndcg_at_20)
    best_pre3 = history_p_at_3[best_ndcg20_index]
    best_pre5 = history_p_at_5[best_ndcg20_index]
    best_pre10 = history_p_at_10[best_ndcg20_index]
    best_pre20 = history_p_at_20[best_ndcg20_index]
    best_rec3 = history_r_at_3[best_ndcg20_index]
    best_rec5 = history_r_at_5[best_ndcg20_index]
    best_rec10 = history_r_at_10[best_ndcg20_index]
    best_rec20 = history_r_at_20[best_ndcg20_index]
    best_ndcg3 = history_ndcg_at_3[best_ndcg20_index]
    best_ndcg5 = history_ndcg_at_5[best_ndcg20_index]
    best_ndcg10 = history_ndcg_at_10[best_ndcg20_index]
    best_ndcg20 = history_ndcg_at_20[best_ndcg20_index]

    print(
        "Best Epochs: pre3: %.5f  rec3: %.5f  ndcg3: %.5f  pre5: %.5f  rec5: %.5f ndcg5: %.5f  pre10:  %.5f  rec10:  %.5f ndcg10:  %.5f  pre20:  %.5f  rec20:  %.5f ndcg20:  %.5f" % \
        (best_pre3, best_rec3, best_ndcg3, best_pre5, best_rec5, best_ndcg5, best_pre10, best_rec10, best_ndcg10,
         best_pre20, best_rec20, best_ndcg20))
    logging.info(
        "Best Epochs: pre3: %.5f  rec3: %.5f  ndcg3: %.5f  pre5: %.5f  rec5: %.5f ndcg5: %.5f  pre10:  %.5f  rec10:  %.5f ndcg10:  %.5f  pre20:  %.5f  rec20:  %.5f ndcg20:  %.5f" % \
        (best_pre3, best_rec3, best_ndcg3, best_pre5, best_rec5, best_ndcg5, best_pre10, best_rec10, best_ndcg10,
         best_pre20, best_rec20, best_ndcg20))
    out_max(best_pre3, best_rec3, best_ndcg3, best_pre5, best_rec5, best_ndcg5, best_pre10, best_rec10, best_ndcg10,
            best_pre20, best_rec20, best_ndcg20)


def out_max(pre3, rec3, ndcg3, pre5, rec5, ndcg5, pre10, rec10, ndcg10, pre20, rec20, ndcg20):
    code_name = os.path.basename(__file__).split('.')[0]
    log_path_ = "log/%s/" % (code_name)
    if not os.path.exists(log_path_):
        os.makedirs(log_path_)
    csv_path = log_path_ + "%s.csv" % (args.dataset)
    log = "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s" % (
        args.embedding_size, args.lr, args.reg_rate, args.ssl_reg, args.layer, args.temperature, args.eps,
        args.noise_type[0], args.noise_type[1],
        pre3, rec3, ndcg3,
        pre5, rec5, ndcg5,
        pre10, rec10, ndcg10,
        pre20, rec20, ndcg20)
    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            f.write(
                "embedding_size,learning_rate,reg_rate,ssl_reg,layer,temperature,eps,noise_type,noise_type,pre3,recall3,ndcg3,pre5,recall5,ndcg5,pre10,recall10,ndcg10,pre20,recall20,ndcg20" + '\n')
            f.write(log + '\n')
            f.close()
    else:
        with open(csv_path, 'a+') as f:
            f.write(log + '\n')
            f.close()


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Go WGCL")
    parser.add_argument('--dataset_path', nargs='?', default='./dataset/amazon-book/',
                        help='Data path.')
    parser.add_argument('--dataset', nargs='?', default='amazon-book',
                        help='Name of the dataset.')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--embedding_size', type=int, default=64,
                        help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int, default=3,  # (1,2,3)
                        help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="the learning rate")
    parser.add_argument('--reg_rate', type=float, default=0.00001,
                        help='Regularization coefficient for user and item embeddings.')
    parser.add_argument('--ssl_reg', type=float, default=0.1,
                        help='Regularization coefficient for ssl.')
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=0.2,
                        help="the hyper-parameter")
    parser.add_argument('--gamma', type=float, default=0.0001,
                        help='the weight 𝛾 controls the desired degree of uniformity.')
    parser.add_argument('--noise_type', type=list, default=['uniform_noise', 'uniform_noise'],
                        help='type of noise. Like [uniform_noise, uniform_noise],[uniform_noise, Gussian_noise],[Gussian_noise, Gussian_noise]')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=100, help='random seed')
    parser.add_argument('--verbose', type=int, default=1, help='Interval of evaluation.')
    return parser.parse_args()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    code_name = os.path.basename(__file__).split('.')[0]
    args = parse_args()
    print(args)
    setup_seed(args.seed)

    log_path = "log/%s_%s/" % (code_name, strftime('%Y-%m-%d', localtime()))
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_path = log_path + "%s_embed_size%.4f_reg%.5f_ssl_reg%.5f_lr%0.5f_eps%.3f_temperature%.3f_layer%.4f_noise_type%s_%s_%s" % (
        args.dataset, args.embedding_size, args.reg_rate, args.ssl_reg, args.lr, args.eps, args.temperature, args.layer,
        args.noise_type[0], args.noise_type[1], strftime('%Y_%m_%d_%H', localtime()))
    logging.basicConfig(filename=log_path, level=logging.INFO)
    logging.info(args)

    data_train, data_test, num_users, num_items = load_data(args.dataset_path + args.dataset)
    train(data_train, data_test, num_users, num_items)