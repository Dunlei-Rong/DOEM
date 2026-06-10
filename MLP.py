import os

import numpy
import numpy as np
import pytorch_lightning as pl
import torch
# import wandb

from src.utils.metrics import Precision, NormalizedDCG
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from src.utils.metrics import Precision, NormalizedDCG, Coverage, FaultTolerance, Efficiency, Redundancy, Count
from src.utils.utils import LambdaRankLoss, jaccard_similarity
from itertools import combinations
import itertools
import json
import re
import pickle
import random

class MLP(pl.LightningModule):
    @property
    def device(self):
        return self._device

    def __init__(self, data_dir, topk, embedding_model='openai', fine_tuning=False, enhence=False, negative_sample: int = 5, lr: float = 1e-3,
                 weight_decay=1e-4, hidden_channels=64, alpha=0.5, R=10, mode='', tag_method='manual', dataset='mashup', sample_mode='random'):
        super().__init__()
        self.save_hyperparameters()
        self.device = torch.device('cuda:0')
        self.sample_mode = sample_mode
        self.embedding_model = embedding_model
        self.fine_tuning = fine_tuning
        if dataset == 'Youshu':
            edge_path = '/standard/train_edges.emb'
        else:
            edge_path = '/left_one_data/train_edges.emb'
        if dataset == 'Youshu':
            self.input_channel = 100
            with open(data_dir + '/raw/train_item', 'rb') as f:
                user_item = pickle.load(f)
            self.mashup_embeds = nn.Embedding(user_item.shape[0], self.input_channel)
            self.api_embeds = nn.Embedding(user_item.shape[1], self.input_channel)
            self.hidden_channel = int(self.input_channel / 2)
            self.num_api = self.api_embeds.num_embeddings
            self.num_mashup = self.mashup_embeds.num_embeddings
        if dataset == 'Mashup' or dataset == 'Huggingface':
            if embedding_model == 'openai':
                pre_data_dir = data_dir + '/preprocessed_data/openai_emb/'
                self.api_embeds = torch.stack(torch.load(pre_data_dir + 'api_openai_text_embedding.pt'), dim=0).to(self.device)
                self.mashup_embeds = torch.stack(torch.load(pre_data_dir + 'mashup_openai_text_embedding.pt'), dim=0).to(self.device)
            elif embedding_model == 'bert':
                pre_data_dir = data_dir + '/preprocessed_data/description/'
                self.api_embeds = torch.load(
                    pre_data_dir + 'bert_api_embeddings.pt').to(self.device)
                self.mashup_embeds = torch.load(
                    pre_data_dir + 'bert_mashup_embeddings.pt').to(self.device)
            elif embedding_model == 'bert' and fine_tuning is True:
                pre_data_dir = data_dir + '/preprocessed_data/description/'
                self.api_embeds = torch.load(pre_data_dir + 'bert_apis_embeddings.emb')
                self.mashup_embeds = torch.load(pre_data_dir + 'bert_mashup_embeddings.emb')[1]
            else:
                pre_data_dir = data_dir + '/preprocessed_data/word2vec/'
                self.api_embeds = torch.stack(torch.load(pre_data_dir + 'api_word2vec_text_embedding.pt'), dim=0)
                self.mashup_embeds = torch.stack(torch.load(pre_data_dir + 'mashup_word2vec_text_embedding.pt'), dim=0)
            self.input_channel = self.api_embeds.shape[1]
            self.hidden_channel = int(self.api_embeds.shape[1] / 2)
            self.num_api = self.api_embeds.size(0)
            self.num_mashup = self.mashup_embeds.size(0)

        self.nodes = torch.cat([self.mashup_embeds, self.api_embeds], dim=0).to(self.device)
        # edges = torch.load(data_dir + edge_path)

        # 模型输入输出参数
        self.hidden_channels = hidden_channels
        self.mashup_embeds_channels = self.mashup_embeds.shape[1]
        self.api_embeds_channels = self.api_embeds.shape[1]
        self.num_api = self.api_embeds.shape[0]
        self.num_mashup = self.mashup_embeds.shape[0]

        # 负采样
        self.negative_sample = negative_sample

        # 超参数
        self.lr = lr
        self.weight_decay = weight_decay

        edges = torch.load(data_dir + edge_path)
        graph = torch.zeros(self.num_mashup + self.num_api, self.num_mashup + self.num_api).to(self.device)
        invert_graph = torch.ones(self.num_mashup + self.num_api, self.num_mashup + self.num_api).to(self.device)
        for edge in edges:
            graph[edge[0]][edge[1]] = 1
            graph[edge[1]][edge[0]] = 1
            invert_graph[edge[0]][edge[1]] = 0
            invert_graph[edge[1]][edge[0]] = 0
        self.invert_graph = invert_graph

        # 损失函数
        self.loss_function = th.nn.BCEWithLogitsLoss()

        # 评价指标
        self.topk = topk
        self.max = 0
        # TODO:从这里开始

        self.alpha = alpha
        self.R = R
        self.criterion = torch.nn.MultiLabelSoftMarginLoss()


        if tag_method == 'manual':
            with open(data_dir + '/label/raw/all.json', "r") as f:
                tag_set1 = json.load(f)
            tag_set = [tag_set1[i]['manual'] for i in tag_set1]
        else:
            with open(data_dir + '/label/raw/chatgpt_result.json', "r") as f:
                tag_set1 = json.load(f)
            tag_set = [tag_set1[i]['chatgpt'] for i in tag_set1]
        tag_set_all = set.union(*[set(i) for i in tag_set])

        mashup_path = data_dir + '/label/mashup_tag_embedding.pt'
        if os.path.exists(mashup_path):
            self.mashup_tag_embedding = torch.load(mashup_path)
        else:
            mashup_tag_list = torch.load(data_dir + '/label/label_catagory/new_tags.pt')
            self.mashup_tag_list = []
            for description in mashup_tag_list:
                matches = re.findall(r'\[([^\]]+)\]', description)

                # 将提取的内容用逗号分隔并去除多余的空格，形成列表
                result_list = [item.strip() for item in matches[0].split(',')]
                self.mashup_tag_list.append(result_list)
            self.mashup_tag_embedding = []
            label_to_index = {label: idx for idx, label in enumerate(tag_set_all)}
            for mashup in self.mashup_tag_list:
                vector = torch.zeros(len(tag_set_all), dtype=torch.float32)
                for label in mashup:
                    if label in label_to_index:
                        vector[label_to_index[label]] = 1
                self.mashup_tag_embedding.append(vector)
            self.mashup_tag_embedding = torch.stack(self.mashup_tag_embedding, dim=0)
            torch.save(self.mashup_tag_embedding, mashup_path)

        api_path = data_dir + '/label/api_tag_embedding.pt'
        if os.path.exists(api_path):
            self.api_tag_embedding = torch.load(api_path)
        else:
            self.api_tag_embedding = []  # TODO: 把他们变成one-hot格式
            label_to_index = {label: idx for idx, label in enumerate(tag_set_all)}
            for api in tag_set:
                vector = torch.zeros(len(tag_set_all), dtype=torch.float32)
                for label in api:
                    vector[label_to_index[label]] = 1
                self.api_tag_embedding.append(vector)
            self.api_tag_embedding = torch.stack(self.api_tag_embedding, dim=0)
            torch.save(self.api_tag_embedding, api_path)

        self.P = {}
        self.P_val = {}
        self.DCG = {}
        self.DCG_val = {}
        self.Coverage = {}
        self.Coverage_val = {}
        self.FaultTolerance = {}
        self.FaultTolerance_val = {}
        self.Efficiency = {}
        self.Efficiency_val = {}
        self.Redundancy = {}
        self.Redundancy_val = {}
        # self.Tolerance = {}
        # self.Tolerance_val = {}

        related_metrix_path = 'data/Mashup/label/related_matrix.pt'
        for k in self.topk:
            self.P[k] = Precision(k).to(self.device)
            self.P_val[k] = Precision(k).to(self.device)
            self.DCG[k] = NormalizedDCG(k).to(self.device)
            self.DCG_val[k] = NormalizedDCG(k).to(self.device)
            self.Coverage[k] = Coverage(self.mashup_tag_embedding, self.api_tag_embedding, k).to(self.device)
            self.Coverage_val[k] = Coverage(self.mashup_tag_embedding, self.api_tag_embedding, k).to(self.device)
            self.FaultTolerance[k] = FaultTolerance(self.mashup_tag_embedding, self.api_tag_embedding, k).to(
                self.device)
            self.FaultTolerance_val[k] = FaultTolerance(self.mashup_tag_embedding, self.api_tag_embedding, k).to(
                self.device)
            self.Efficiency[k] = Efficiency(self.mashup_tag_embedding, self.api_tag_embedding, k).to(self.device)
            self.Efficiency_val[k] = Efficiency(self.mashup_tag_embedding, self.api_tag_embedding, k).to(self.device)
            self.Redundancy[k] = Redundancy(self.api_tag_embedding, k).to(self.device)
            self.Redundancy_val[k] = Redundancy(self.api_tag_embedding, k).to(self.device)
        self.Count = Count(self.mashup_tag_embedding, self.api_tag_embedding).to(self.device)
        self.Count_val = Count(self.mashup_tag_embedding, self.api_tag_embedding).to(self.device)

            # self.Tolerance[k] = Tolerance(related_metrix_path, k).to(self.device)
            # self.Tolerance_val[k] = Tolerance(related_metrix_path, k).to(self.device)

        if os.path.exists(related_metrix_path):
            self.related_matrix = torch.load(related_metrix_path)
        else:
            self.related_matrix = torch.zeros((self.num_api, self.num_api), dtype=torch.float32)
            for i, j in itertools.combinations(range(len(tag_set)), 2):
                self.related_matrix[i, j] = jaccard_similarity(set(tag_set[i]), set(tag_set[i]))

                self.related_matrix[i, j] = self.related_matrix[j, i]
            torch.save(self.related_matrix, related_metrix_path)
        self.llambdaloss = LambdaRankLoss(self.api_tag_embedding, mode).to(self.device)

        # TODO：一直到这里
        self._build_layers()

    def _build_layers(self):

        self.mlp = nn.Sequential(
            nn.Linear(self.mashup_embeds_channels * 2, self.hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_channels, 1)
        )

    def forward(self, input):
        # out1 = self.mlp_api(apis.to(self.device))
        # out2 = self.mlp_mashup(mashup.to(self.device))
        out = self.mlp(input)
        return out

    def format_samples(self, Xs, Ys):
        mashup_embedings, api_embedings, labels = [], [], []
        sampled_api_list = []

        for x, y in zip(Xs, Ys):
            pos_apis = torch.nonzero(y).flatten()
            if pos_apis.numel() == 0:
                sampled_api_list.append(torch.tensor([], device=self.device, dtype=torch.long))
                continue

            pos_api = pos_apis[0]
            distribution = torch.ones(self.num_api, device=self.device) - y
            uninvoked_apis = torch.multinomial(
                distribution, num_samples=self.negative_sample, replacement=False
            )

            cur_sampled = [pos_api.item()]

            mashup_embedings.append(self.mashup_embeds[x])
            api_embedings.append(self.api_embeds[pos_api])
            labels.append(1)

            for api in uninvoked_apis:
                mashup_embedings.append(self.mashup_embeds[x])
                api_embedings.append(self.api_embeds[api])
                labels.append(0)
                cur_sampled.append(api.item())

            sampled_api_list.append(torch.tensor(cur_sampled, device=self.device, dtype=torch.long))

        if len(mashup_embedings) == 0:
            return None, None, None

        mashup_embeddings = torch.stack(mashup_embedings, dim=0).to(self.device)
        api_embeddings = torch.stack(api_embedings, dim=0).to(self.device)
        labels = torch.tensor(labels, device=self.device).float()
        input = torch.cat([mashup_embeddings, api_embeddings], dim=1)
        return input, labels, sampled_api_list

    def on_train_start(self) -> None:

        pass

    def training_step(self, batch, batch_idx):
        Xs, pos_item, _ = batch['users'], batch['pos_items'], batch['neg_items']

        input, labels, sampled_apis = self.format_samples(Xs, pos_item)


        # 如果整个 batch 都没有可用样本
        if input is None:
            loss = torch.tensor(0.0, device=Xs.device, requires_grad=True)
            self.log("loss", loss, on_step=True, on_epoch=True)
            self.log("base_loss", loss, on_step=True, on_epoch=True)
            self.log("aux_loss", loss, on_step=True, on_epoch=True)
            return loss

        out = self.forward(input.to(self.dtype)).squeeze(-1)  # [sum_i len(sampled_api_indices[i])]
        base_loss = self.loss_function(out, labels.float())

        mashup = torch.stack(
            [self.mashup_tag_embedding[user1] for user1 in Xs],
            dim=0
        ).to(out.device)

        weight_list = []
        rank_list = []

        selected_indices = None
        if self.sample_mode == 'highest':
            score = torch.matmul(mashup, self.api_tag_embedding.T.to(mashup.device))
            selected_indices = self.select_indices(score, self.R, self.num_api)

        offset = 0
        for i, sample_idx in enumerate(sampled_apis):
            cur_len = len(sample_idx)
            if cur_len == 0:
                continue

            pref_list = out[offset: offset + cur_len]
            offset += cur_len

            global2local = {g.item() if torch.is_tensor(g) else g: l for l, g in enumerate(sample_idx)}

            sorted_local = torch.argsort(pref_list, descending=True)
            rel_topk_local = sorted_local[:min(self.R, cur_len)].tolist()
            rel_topk_global = [sample_idx[l].item() if torch.is_tensor(sample_idx[l]) else sample_idx[l]
                               for l in rel_topk_local]

            gt_global = torch.nonzero(pos_item[i]).flatten().tolist()
            gt_global = [g for g in gt_global if g in global2local]

            rank_global = rel_topk_global + gt_global

            if self.sample_mode == 'random':
                cand = list(set(sample_idx) - set(rank_global))
                rand_num = max(1, self.R // 5)
                rand_global = random.sample(cand, min(rand_num, len(cand)))
                rank_global += rand_global

            elif self.sample_mode == 'highest':
                selected_global = selected_indices[i]
                selected_global = [g for g in selected_global if g in global2local]
                cov_num = max(1, self.R // 5)
                rank_global += selected_global[:cov_num]

            # 去重保序
            seen = set()
            rank_global_unique = []
            for g in rank_global:
                if g not in seen:
                    seen.add(g)
                    rank_global_unique.append(g)

            # 用局部索引从 pref_list 取分数
            rank_local_unique = [global2local[g] for g in rank_global_unique]

            rank_tensor_global = torch.tensor(
                rank_global_unique, device=out.device, dtype=torch.long
            )
            rank_tensor_local = torch.tensor(
                rank_local_unique, device=out.device, dtype=torch.long
            )

            # 防御性检查，避免越界
            if rank_tensor_local.numel() > 0:
                assert rank_tensor_local.max().item() < pref_list.shape[0], (
                    f"local index out of range: max={rank_tensor_local.max().item()}, "
                    f"pref_list_len={pref_list.shape[0]}, sample_idx_len={len(sample_idx)}"
                )

            rank_list.append(rank_tensor_global)
            weight_list.append(pref_list[rank_tensor_local])

        aux_loss = self.llambdaloss(weight_list, rank_list, mashup).to(out.device)
        loss = base_loss + self.alpha * aux_loss

        if torch.isnan(loss) or torch.isinf(loss):
            loss = base_loss
            if torch.isnan(loss) or torch.isinf(loss):
                loss = torch.tensor(0.0, device=out.device, requires_grad=True)

        self.log("loss", loss, on_step=True, on_epoch=True)
        self.log("base_loss", base_loss, on_step=True, on_epoch=True)
        self.log("aux_loss", aux_loss, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # todo:
        Xs, Ys, _ = batch['users'], batch['pos_items'], batch['neg_items']
        if not torch.is_tensor(Xs):
            Xs = torch.tensor(Xs, device=self.mashup_embeds.device, dtype=torch.long)
        else:
            Xs = Xs.to(self.mashup_embeds.device).long()

        # 1. 取出 mashup 表征: [B, D]
        mashup_embeddings = self.mashup_embeds[Xs]

        # 4. 扩展到和每个 API 配对
        # mashup_expand: [B, N, D]
        mashup_expand = mashup_embeddings.unsqueeze(1).expand(-1, self.num_api, -1)

        # api_expand: [B, N, D]
        api_expand = self.api_embeds.unsqueeze(0).expand(mashup_embeddings.size(0), -1, -1)

        # 5. 拼接后送入前向网络
        # input: [B, N, 2D]
        input = torch.cat([mashup_expand, api_expand], dim=-1).to(self.dtype)
        out = self.forward(input.reshape(-1, input.size(-1)))  # [B*N, 1] 或 [B*N]
        out = out.view(mashup_embeddings.size(0), self.num_api)  # [B, N]

        # 7. sigmoid 得到预测
        preds = torch.sigmoid(out)
        # preds = torch.cat(preds, dim=0)
        # target = F.one_hot((Ys - self.num_mashup).to(torch.int64), self.num_api)

        sorted_indice_list = []
        for pref_list, pos_item_list in zip(preds, Ys):
            # nonzero_indices = torch.nonzero(pos_item_list, as_tuple=True)
            # result_list.append([nonzero_indices[1][nonzero_indices[0] == i].tolist() for i in range(pos_item_list.size(0))])
            sorted_indice = torch.argsort(pref_list, descending=True)
            sorted_indice_list.append(sorted_indice)
        if not self.trainer.sanity_checking:
            # self.Count_val.update(sorted_indice_list, Xs)
            # self.log("val/Count", self.Count_val.compute(), on_step=False, on_epoch=True,prog_bar=True)
            for k in self.topk:
                self.P_val[k].update(preds, Ys)
                self.DCG_val[k].update(preds, Ys)
                # self.Tolerance_val[k].update([sorted_list[:k] for sorted_list in sorted_indice_list], nonzero_indices)
                self.Coverage_val[k].update([sorted_list[:k] for sorted_list in sorted_indice_list], Xs)
                self.FaultTolerance_val[k].update([sorted_list[:k] for sorted_list in sorted_indice_list], Xs)
                self.Efficiency_val[k].update([sorted_list[:k] for sorted_list in sorted_indice_list], Xs)
                self.Redundancy_val[k].update([sorted_list[:k] for sorted_list in sorted_indice_list])

                self.log("val/P@" + str(k), self.P_val[k].compute(), on_step=False, on_epoch=True, prog_bar=True)
                self.log("val/DCG@" + str(k), self.DCG_val[k].compute(), on_step=False, on_epoch=True, prog_bar=True)
                # self.log("val/Tolerance@" + str(k), self.Tolerance_val[k].compute(), on_step=False, on_epoch=True, prog_bar=True)
                self.log("val/Coverage@" + str(k), self.Coverage_val[k].compute(), on_step=False, on_epoch=True,
                         prog_bar=True)
                self.log("val/FaultTolerance@" + str(k), self.FaultTolerance_val[k].compute(), on_step=False,
                         on_epoch=True, prog_bar=True)
                self.log("val/Efficiency@" + str(k), self.Efficiency_val[k].compute(), on_step=False, on_epoch=True,
                         prog_bar=True)
                self.log("val/Redundancy@" + str(k), self.Redundancy_val[k].compute(), on_step=False, on_epoch=True,
                         prog_bar=True)

                # wandb.log({"val/P@" + str(k): self.P_val[k].compute(), "val/DCG@" + str(k): self.DCG_val[k].compute()})

        # TODO: 一直到这里
        if self.P_val[self.topk[1]].compute() > self.max:
            self.max = self.P_val[self.topk[1]].compute()
            torch.save(self.state_dict(), 'model.pt')

    def test_step(self, batch, batch_idx):
        Xs, Ys, _ = batch['users'], batch['pos_items'], batch['neg_items']
        if not torch.is_tensor(Xs):
            Xs = torch.tensor(Xs, device=self.mashup_embeds.device, dtype=torch.long)
        else:
            Xs = Xs.to(self.mashup_embeds.device).long()

        # 1. 取出 mashup 表征: [B, D]
        mashup_embeddings = self.mashup_embeds[Xs]

        # 4. 扩展到和每个 API 配对
        # mashup_expand: [B, N, D]
        mashup_expand = mashup_embeddings.unsqueeze(1).expand(-1, self.num_api, -1)

        # api_expand: [B, N, D]
        api_expand = self.api_embeds.unsqueeze(0).expand(mashup_embeddings.size(0), -1, -1)

        # 5. 拼接后送入前向网络
        # input: [B, N, 2D]
        input = torch.cat([mashup_expand, api_expand], dim=-1).to(self.dtype)
        out = self.forward(input.reshape(-1, input.size(-1)))  # [B*N, 1] 或 [B*N]
        out = out.view(mashup_embeddings.size(0), self.num_api)  # [B, N]

        # 7. sigmoid 得到预测
        preds = torch.sigmoid(out)

        sorted_indice_list = []
        for pref_list, pos_item_list in zip(preds, Ys):
            # nonzero_indices = torch.nonzero(pos_item_list, as_tuple=True)
            # result_list.append([nonzero_indices[1][nonzero_indices[0] == i].tolist() for i in range(pos_item_list.size(0))])
            sorted_indice = torch.argsort(pref_list, descending=True)
            sorted_indice_list.append(sorted_indice)
        if not self.trainer.sanity_checking:
            # self.Count.update(sorted_indice_list, Xs)
            # self.log("test/Count", self.Count.compute(), on_step=False, on_epoch=True, prog_bar=True)
            for k in self.topk:
                self.P[k].update(preds, Ys)
                self.DCG[k].update(preds, Ys)
                # self.Tolerance_val[k].update([sorted_list[:k] for sorted_list in sorted_indice_list], nonzero_indices)
                self.Coverage[k].update([sorted_list[:k] for sorted_list in sorted_indice_list], Xs)
                self.FaultTolerance[k].update([sorted_list[:k] for sorted_list in sorted_indice_list], Xs)
                self.Efficiency[k].update([sorted_list[:k] for sorted_list in sorted_indice_list], Xs)
                self.Redundancy[k].update([sorted_list[:k] for sorted_list in sorted_indice_list])

                self.log("test/P@" + str(k), self.P[k].compute(), on_step=False, on_epoch=True, prog_bar=True)
                self.log("test/DCG@" + str(k), self.DCG[k].compute(), on_step=False, on_epoch=True, prog_bar=True)
                # self.log("val/Tolerance@" + str(k), self.Tolerance_val[k].compute(), on_step=False, on_epoch=True, prog_bar=True)
                self.log("test/Coverage@" + str(k), self.Coverage[k].compute(), on_step=False, on_epoch=True,
                         prog_bar=True)
                self.log("test/FaultTolerance@" + str(k), self.FaultTolerance[k].compute(), on_step=False,
                         on_epoch=True, prog_bar=True)
                self.log("test/Efficiency@" + str(k), self.Efficiency[k].compute(), on_step=False, on_epoch=True,
                         prog_bar=True)
                self.log("test/Redundancy@" + str(k), self.Redundancy[k].compute(), on_step=False, on_epoch=True,
                         prog_bar=True)


    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def select_indices(self, tensor1, n, q):
        n_rows = tensor1.size(0)
        q_indices = list(range(q))  # 用于填补的索引
        result = []

        # 找到每一行的最大值
        row_max_values, _ = torch.max(tensor1, dim=1, keepdim=True)

        # 使用 torch.where 找到每一行中最大值的位置
        max_indices = torch.where(tensor1 == row_max_values)

        # 整理每行的索引
        max_indices_list = [[] for _ in range(n_rows)]
        for row, col in zip(max_indices[0], max_indices[1]):
            max_indices_list[row.item()].append(col.item())

        # 为每一行随机选取 n 个索引
        for indices in max_indices_list:
            if len(indices) >= n:
                # 随机选取 n 个索引
                selected_indices = random.sample(indices, n)
            else:
                # 先选择所有最大值的索引
                selected_indices = indices
                # 从 q_indices 中随机选取补足到 n 个
                # selected_indices += random.sample(q_indices, n - len(indices))

            result.append(selected_indices)

        return result

    def update_api_memory(self, mashup_map, pos_items, device):
        positive_user_idx = []
        positive_api_idx = []

        for i, y_item in enumerate(pos_items):
            positive_idx = torch.nonzero(y_item).flatten()
            if positive_idx.numel() == 0:
                continue

            positive_user_idx.append(
                torch.full((len(positive_idx),), i, dtype=torch.long, device=device)
            )
            positive_api_idx.append(
                positive_idx.to(device)
            )

        if len(positive_api_idx) == 0:
            return

        positive_user_idx = torch.cat(positive_user_idx, dim=0)  # [K]
        positive_api_idx = torch.cat(positive_api_idx, dim=0)  # [K]

        # [B, D]
        all_message = self.msg_generation(mashup_map)

        # [K, D]
        pos_message = all_message[positive_user_idx]

        # 聚合到 API
        api_delta = torch.zeros_like(self.api_embeds)
        api_delta.index_add_(0, positive_api_idx, pos_message)

        count = torch.zeros(self.num_api, device=device, dtype=api_delta.dtype)
        count.index_add_(
            0,
            positive_api_idx,
            torch.ones_like(positive_api_idx, dtype=api_delta.dtype)
        )
        count = count.clamp_min(1.0).unsqueeze(1)

        api_delta = api_delta / count

        updated_mask = count.squeeze(1) > 0
        self.api_embeds[updated_mask] = (
                self.api_embeds[updated_mask] + api_delta[updated_mask]
        ).detach()

    @device.setter
    def device(self, value):
        self._device = value

