from model.base import AbstractRecommender
import os
import pickle
from typing import List
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from reckit import timer  # reckit==0.2.4
from reckit import pad_sequences
from modules import l2_loss, log_loss
from modules.functions import sp_mat_to_sp_tensor, normalize_adj_matrix
from data.sampler import TimeOrderPairwiseSampler
from .manifold_utils import Manifold, Euclidean, Hyperboloid, Sphere, PoincareBall, StereographicallyProjectedSphere

INIT_MEAN, INIT_STD = 0.0, 0.01


class Module(object):
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Trans(Module):
    def __init__(self, init_radius: float, fixed_curvature: bool=True):
        super(Trans, self).__init__()
        self._radius = tf.Variable(init_radius, dtype=tf.float32, trainable=not fixed_curvature, name="radius")
        self.manifold: Manifold = self._init_manifold(self._radius)

    def _init_manifold(self, init_radius: tf.Variable) -> Manifold:
        raise NotImplementedError

    @property
    def radius(self):
        return self.manifold.radius

    def forward(self, user_embeds, head_embeds, tail_embeds):
        manifold = self.manifold

        r_head_embeds = manifold.exp_map(manifold.expand_proj_dims(head_embeds))
        r_tail_embeds = manifold.exp_map(manifold.expand_proj_dims(tail_embeds))

        v_user_embeds = manifold.expand_proj_dims(user_embeds)
        v_user_embeds_at_x = manifold.parallel_transport_from_mu0(v_user_embeds, r_head_embeds)
        r_trans_embeds = manifold.exp_map(v_user_embeds_at_x, r_head_embeds)

        hat_y = -manifold.distance(tf.expand_dims(r_trans_embeds, axis=1), r_tail_embeds)

        return hat_y

    def predict(self, user_embeds, head_embeds, item_embeddings):
        manifold = self.manifold

        r_head_embeds = manifold.exp_map(manifold.expand_proj_dims(head_embeds))
        r_tail_embeds = manifold.exp_map(manifold.expand_proj_dims(item_embeddings))

        v_user_embeds = manifold.expand_proj_dims(user_embeds)
        v_user_embeds_at_x = manifold.parallel_transport_from_mu0(v_user_embeds, r_head_embeds)
        r_trans_embeds = manifold.exp_map(v_user_embeds_at_x, r_head_embeds)

        ratings = -manifold.distance(tf.expand_dims(r_trans_embeds, axis=1), r_tail_embeds)
        return ratings


class EuclideanTrans(Trans):
    def _init_manifold(self, init_radius: tf.Variable):
        return Euclidean(init_radius)


class HyperboloidTrans(Trans):
    def _init_manifold(self, init_radius: tf.Variable):
        return Hyperboloid(init_radius)


class SphereTrans(Trans):
    def _init_manifold(self, init_radius: tf.Variable):
        return Sphere(init_radius)


class PoincareBallTrans(Trans):
    def _init_manifold(self, init_radius: tf.Variable):
        return PoincareBall(init_radius)


class StereographicallyProjectedSphereTrans(Trans):
    def _init_manifold(self, init_radius: tf.Variable):
        return StereographicallyProjectedSphere(init_radius)


manifold_map = {"h": HyperboloidTrans,
                "p": PoincareBallTrans,
                "s": SphereTrans,
                "d": StereographicallyProjectedSphereTrans,
                "e": EuclideanTrans
                }


def info_fusion(past_emb, future_emb, method='mean'):
    if method == "sum":
        fusion_emb = past_emb + future_emb
    elif method == "mean":
        fusion_emb = (past_emb + future_emb) / 2.0
    elif isinstance(method, float) and 0.0 <= method <= 1.0:
        fusion_emb = past_emb*method + future_emb*(1-method)
    else:
        raise ValueError("'method' must be 'mean', 'sum' or a float.")
    return fusion_emb


class GNN(Module):
    def __init__(self, n_layers, adj_mat, gcn_agg="sum"):
        self.adj_mat = adj_mat
        self.n_layers = n_layers
        self.agg = gcn_agg

    def forward(self, x):
        for _ in range(self.n_layers):
            x1 = tf.sparse.matmul(self.adj_mat, x)
            x = info_fusion(x, x1, self.agg)
        return x


class _BiSGCN(Module):
    def __init__(self, n_users, n_items, p_graph, f_graph, config):
        super(_BiSGCN, self).__init__()
        self.fusion = config["fw_prob"]
        self._manifolds = config["manifolds"]
        self._init_radius = config["init_radius"]
        self.n_dim = config["n_dim"]
        self.long_term = config["long_term"]
        n_layers = config["n_layers"]
        gcn_agg = config["gcn_agg"]
        self.manifolds, self.dim_list = self._parse_manifolds()
        assert sum(self.dim_list) == self.n_dim

        # init embeddings
        init = tf.initializers.random_normal(mean=INIT_MEAN, stddev=INIT_STD)
        self.init_user_embeds = tf.Variable(init([n_users, self.n_dim]))
        self.init_item_embeds = tf.Variable(init([n_items, self.n_dim]))

        zero_init = tf.initializers.zeros()
        self.item_biases = tf.Variable(zero_init([n_items]))

        self.p_gnn = GNN(n_layers, p_graph, gcn_agg)
        self.f_gnn = GNN(n_layers, f_graph, gcn_agg)

        # for test
        self.final_item_embeds = tf.Variable(tf.zeros(tf.shape(self.init_item_embeds)),
                                             dtype=tf.float32, trainable=False)

    def _parse_manifolds(self) -> [List, List]:
        _init_radius = self._init_radius

        manifold_list = []
        dim_list = []
        for space in self._manifolds.split(","):
            alias = space[0]
            n_dim = eval(space[1:])
            dim_list.append(n_dim)
            if alias in manifold_map:
                manifold = manifold_map[alias](_init_radius)
                manifold_list.append(manifold)
            else:
                raise ValueError(f"{alias} is an invalid manifold alias.")
        return manifold_list, dim_list

    def gcn_forward(self):
        p_item_embeds = self.p_gnn(self.init_item_embeds)
        f_item_embeds = self.f_gnn(self.init_item_embeds)
        item_embeds = info_fusion(p_item_embeds, f_item_embeds, self.fusion)
        return item_embeds

    def _mean_history(self, item_embeddings, head_items):
        # fuse to get short-term embeddings
        pad_value = tf.zeros([1, self.n_dim], dtype=tf.float32)
        item_embeddings = tf.concat([item_embeddings, pad_value], axis=0)
        pad_id = tf.shape(item_embeddings)[0]

        item_seq_embeds = tf.nn.embedding_lookup(item_embeddings, head_items)  # (b,l,d)
        mask = tf.cast(tf.not_equal(head_items, pad_id), dtype=tf.float32)  # (b,l)
        his_embeds = tf.reduce_sum(item_seq_embeds, axis=1) / tf.reduce_sum(mask, axis=1, keepdims=True)  # (b,d)/(b,1)
        return his_embeds

    def _forward_head_embed(self, item_embeddings, head_items):
        # embed item sequence
        last_embeds = tf.nn.embedding_lookup(item_embeddings, head_items[:, -1])  # b*d

        if self.long_term is True:
            his_embeds = self._mean_history(item_embeddings, head_items)
            head_embeds = info_fusion(last_embeds, his_embeds, "sum")
        else:
            head_embeds = last_embeds

        return head_embeds

    def forward(self, users, head_items, tail_items):
        # GCN
        user_embeddings = self.init_user_embeds
        item_embeddings = self.gcn_forward()

        user_embeds = tf.nn.embedding_lookup(user_embeddings, users)
        head_embeds = self._forward_head_embed(item_embeddings, head_items)
        tail_embeds = tf.nn.embedding_lookup(item_embeddings, tail_items)
        reg_params = [user_embeds, head_embeds, tail_embeds]

        # Manifold trans
        user_embed_list = tf.split(user_embeds, self.dim_list, axis=-1)
        head_embed_list = tf.split(head_embeds, self.dim_list, axis=-1)
        tail_embed_list = tf.split(tail_embeds, self.dim_list, axis=-1)

        ratings = []
        for manifold, _user_embeds, _head_embeds, _tail_embeds in \
                zip(self.manifolds, user_embed_list, head_embed_list, tail_embed_list):
            sub_rating = manifold(_user_embeds, _head_embeds, _tail_embeds)
            ratings.append(tf.expand_dims(sub_rating, axis=-1))

        item_bias = tf.gather(self.item_biases, tail_items)
        reg_params.append(item_bias)
        train_ratings = tf.reduce_sum(tf.concat(ratings, axis=-1), axis=-1) + item_bias

        # for test
        with tf.control_dependencies([tf.debugging.assert_all_finite(item_embeddings,
                                                                     message="'item_embeddings' are not all finite.")]):
            assign_opt = tf.assign(self.final_item_embeds, item_embeddings)

        head_embeds = self._forward_head_embed(item_embeddings, head_items)
        head_embed_list = tf.split(head_embeds, self.dim_list, axis=-1)
        item_embeds_list = tf.split(self.final_item_embeds, self.dim_list, axis=-1)

        ratings = []
        for manifold, _user_embeds, _head_embeds, _item_embeds in \
                zip(self.manifolds, user_embed_list, head_embed_list, item_embeds_list):
            sub_rating = manifold.predict(_user_embeds, _head_embeds, _item_embeds)
            ratings.append(tf.expand_dims(sub_rating, axis=-1))
        eval_ratings = tf.reduce_sum(tf.concat(ratings, axis=-1), axis=-1) + self.item_biases

        m_embeds_list = []
        for manifold, _item_embeds in zip(self.manifolds, item_embeds_list):
            _manifold = manifold.manifold
            m_embeds_list.append(_manifold.exp_map(_manifold.expand_proj_dims(_item_embeds)))
        return train_ratings, eval_ratings, assign_opt, reg_params


class BiSGCN(AbstractRecommender):
    def __init__(self, config):
        super(BiSGCN, self).__init__(config)
        self.config = config
        self.lr = config["lr"]
        self.reg = config["reg"]
        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]
        self.n_seqs = config["n_seqs"]
        self.n_next = config["n_next"]

        self.n_layers = config["n_layers"]
        self.fw_prob = config["fw_prob"]

        self.manifolds = config["manifolds"]
        self.init_radius = config["init_radius"]

        self.users_num, self.items_num = self.dataset.num_users, self.dataset.num_items
        self._init_constant()
        self.user_group = defaultdict(list)
        for user, item_seq in self.user_pos_train.items():
            self.user_group[len(item_seq)].append(user)

        self._build_model()
        self.sess.run(tf.global_variables_initializer())
        # restore model
        self.save_interval = config["save_interval"]
        self.global_step = -1
        if "restore_model" in config and config["restore_model"]:
            self.global_step = self.restore_model()

    def _create_placeholder(self):
        self.user_ph = tf.placeholder(tf.int32, [None], name="users")
        self.head_ph = tf.placeholder(tf.int32, [None, self.n_seqs], name="item_seqs")  # the previous item
        self.pos_next_ph = tf.placeholder(tf.int32, [None, self.n_next], name="pos_next_item")  # the next item
        self.neg_next_ph = tf.placeholder(tf.int32, [None, self.n_next], name="neg_next_item")  # the next item

    def _init_constant(self):
        dir_name = os.path.dirname(self.config["train_file"])
        dir_name = os.path.join(dir_name, "_"+self.__class__.__name__.lower())
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        pos_train_name = os.path.join(dir_name, "pos_train_seq.pkl")
        if not os.path.exists(pos_train_name):
            self.user_pos_train = self.dataset.get_user_train_dict(by_time=True)
            pickle.dump(self.user_pos_train, open(pos_train_name, 'wb'))
        else:
            self.user_pos_train = pickle.load(open(pos_train_name, 'rb'))

        test_seqs_name = os.path.join(dir_name, f"test_item_seqs_{self.n_seqs}.pkl")
        if not os.path.exists(test_seqs_name):
            self.test_item_seqs = self._process_test()
            pickle.dump(self.test_item_seqs, open(test_seqs_name, 'wb'))
        else:
            self.test_item_seqs = pickle.load(open(test_seqs_name, 'rb'))

        sampler_name = os.path.join(dir_name, "sampler_%d_%d.pkl" % (self.n_seqs, self.n_next))
        if not os.path.exists(sampler_name):
            self.sampler = TimeOrderPairwiseSampler(self.dataset,
                                                    len_seqs=self.n_seqs, len_next=self.n_next,
                                                    pad=self.items_num, num_neg=self.n_next,
                                                    batch_size=self.batch_size,
                                                    shuffle=True, drop_last=False)
            pickle.dump(self.sampler, open(sampler_name, 'wb'))
        else:
            self.sampler = pickle.load(open(sampler_name, 'rb'))

        forward_gname = os.path.join(dir_name, "forward_graph.npz")
        backward_gname = os.path.join(dir_name, "backward_graph.npz")

        if os.path.exists(forward_gname) and os.path.exists(backward_gname):
            forward_adj_mat = sp.load_npz(forward_gname)
            backward_adj_mat = sp.load_npz(backward_gname)
        else:
            item_adj_mat = self._build_item_graph()
            forward_adj_mat = normalize_adj_matrix(item_adj_mat, "left")
            backward_adj_mat = normalize_adj_matrix(item_adj_mat.transpose(copy=True), "left")
            sp.save_npz(forward_gname, forward_adj_mat)
            sp.save_npz(backward_gname, backward_adj_mat)

        self.forward_adj_mat = sp_mat_to_sp_tensor(forward_adj_mat)
        self.backward_adj_mat = sp_mat_to_sp_tensor(backward_adj_mat)

        self.forward_nnz = forward_adj_mat.nnz
        self.backward_nnz = backward_adj_mat.nnz

    @timer
    def _build_item_graph(self):
        th_rs_dict = defaultdict(list)
        for user, pos_items in self.user_pos_train.items():
            for h, t in zip(pos_items[:-1], pos_items[1:]):
                th_rs_dict[(t, h)].append(user)

        th_len_list = [[t, h, len(rs)] for (t, h), rs in th_rs_dict.items()]
        tail_list, head_list, edge_num = list(zip(*th_len_list))

        adj_mat = sp.csr_matrix((edge_num, (tail_list, head_list)), dtype=np.float32,
                                shape=(self.items_num, self.items_num))  # in matrix
        return adj_mat

    def _process_test(self):
        item_seqs = [self.user_pos_train[user][-self.n_seqs:] if user in self.user_pos_train else [self.items_num]
                     for user in range(self.users_num)]

        test_item_seqs = pad_sequences(item_seqs, value=self.items_num, max_len=self.n_seqs,
                                       padding='pre', truncating='pre', dtype=np.int32)
        return test_item_seqs

    def _build_model(self):
        self._create_placeholder()
        rmgcn = _BiSGCN(self.users_num, self.items_num, self.forward_adj_mat, self.backward_adj_mat, self.config)
        tail_items = tf.concat([self.pos_next_ph, self.neg_next_ph], axis=1)
        train_ratings, self.prediction, self.assign_opt, params = rmgcn(self.user_ph, self.head_ph, tail_items)

        yui, yuj = tf.split(train_ratings, 2, axis=1)
        bpr_loss = tf.reduce_sum(log_loss(yui-yuj))
        reg_loss = l2_loss(*params)

        final_loss = bpr_loss + self.reg * reg_loss
        self.train_opt = tf.train.AdamOptimizer(self.lr).minimize(final_loss)

    def train_model(self):
        all_group_result = []
        self.logger.info(self.evaluator.metrics_info())

        update_count = 0.0
        bets_result = 0.0
        stop_counter = 0
        best_str = ""
        fetch_dir = os.path.split(os.path.dirname(self.config["train_file"]))[-1]
        fetch_dir = os.path.join("embeds", fetch_dir)
        if not os.path.exists(fetch_dir):
            os.makedirs(fetch_dir)

        for epoch in range(self.global_step+1, self.epochs):
            for bat_users, bat_item_seq, bat_pos_next, bat_neg_next in self.sampler:
                feed_dict = dict()

                feed_dict.update({self.user_ph: bat_users,
                                  self.head_ph: np.reshape(bat_item_seq, newshape=[-1, self.n_seqs]),
                                  self.pos_next_ph: np.reshape(bat_pos_next, newshape=[-1, self.n_next]),
                                  self.neg_next_ph: np.reshape(bat_neg_next, newshape=[-1, self.n_next])
                                  })
                self.sess.run(self.train_opt, feed_dict=feed_dict)
                update_count += 1

            all_group_result.append(self.evaluate_group())
            result = self.evaluate_model()
            self.logger.info("epoch %d:\t%s" % (epoch, result))
            stop_counter += 1
            if stop_counter > 20:
                self.logger.info("early stop")
                break

            cur_result = float(result.split("\t")[6])
            if cur_result >= bets_result:
                bets_result = cur_result
                best_str = result
                stop_counter = 0

        self.logger.info("best:\t%s" % best_str)
        file_name = self.logger.logger.name
        file_name = file_name.replace(".log", ".pkl")
        with open(file_name, 'wb') as fout:
            pickle.dump(all_group_result, fout)

    def evaluate_model(self):
        self.sess.run(self.assign_opt)
        return self.evaluator.evaluate(self)

    def evaluate_group(self):
        group_result = dict()
        self.sess.run(self.assign_opt)
        for seq_len, users in self.user_group.items():
            result_str = self.evaluator.evaluate(self, users)
            result = eval(result_str.replace('\t', ","))
            result = np.float32(result)
            group_result[(seq_len, len(users))] = result
        return group_result

    def predict_for_eval(self, users):
        last_items = [self.test_item_seqs[u] for u in users]
        feed = {self.user_ph: users, self.head_ph: last_items}
        bat_ratings = self.sess.run(self.prediction, feed_dict=feed)
        return bat_ratings
