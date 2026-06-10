import math
import os

import torch
from torchmetrics import Metric
import torch.nn as nn
import numpy as np
from itertools import combinations
from src.utils.utils import jaccard_similarity, LambdaRankLoss
import os
from itertools import combinations

import torch
from torchmetrics import Metric


class Precision(Metric):
    def __init__(self, top_k, propensity_score=None, dist_sync_on_step=False):
        super(Precision, self).__init__(dist_sync_on_step=dist_sync_on_step)
        self.top_k = top_k
        self.propensity_score = propensity_score
        self.add_state('score', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target, model=None):
        top_k, indices = torch.topk(preds, self.top_k, dim=1)
        res = torch.zeros_like(preds).type_as(preds)
        res = res.scatter(1, indices, torch.ones_like(top_k).type_as(top_k))

        score = res * target

        if self.propensity_score is not None:
            score = score / (self.propensity_score)

        score = score.sum(dim=1)
        if model == 'user-based' or model == 'popularity':
            non_zero_mask = target.sum(dim=1) != 0
            score_non_zero = score[non_zero_mask]
            target_non_zero = target[non_zero_mask]
            score = score_non_zero / target_non_zero.sum(dim=1)
        else:
            score = score / target.sum(dim=1)
        self.score += score.sum()
        if model == 'user-based' or model == 'popularity':
            self.total += non_zero_mask.sum()
        else:
            self.total += preds.size(0)

    def compute(self):
        return self.score / self.total


class HIT(Metric):
    def __init__(self, top_k, propensity_score=None, dist_sync_on_step=False):
        super(HIT, self).__init__(dist_sync_on_step=dist_sync_on_step)
        self.top_k = top_k
        self.propensity_score = propensity_score
        self.add_state('score', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        top_k, indices = torch.topk(preds, self.top_k, dim=1)
        res = torch.zeros_like(preds).type_as(preds)
        res = res.scatter(1, indices, torch.ones_like(top_k).type_as(top_k))
        score = res * target

        if self.propensity_score is not None:
            score = score / (self.propensity_score)

        score = score.sum(dim=1)
        self.score += score.sum()
        self.total += preds.size(0)

    def compute(self):
        if self.total == 0:
            return 0
        return self.score / self.total


class NormalizedDCG(Metric):
    def __init__(self, top_k, propensity_score=None, dist_sync_on_step=False):
        super(NormalizedDCG, self).__init__(dist_sync_on_step=dist_sync_on_step)
        self.top_k = top_k
        self.propen_score = propensity_score
        self.add_state('score', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        top_k, indices = torch.topk(preds, self.top_k, dim=1)
        res = torch.zeros_like(preds).type_as(preds)
        res = res.scatter(1, indices, torch.ones_like(top_k).type_as(top_k))
        score = res * target

        log_val = torch.log2(torch.arange(2, self.top_k + 2)).type_as(preds)
        res_log = torch.ones_like(preds).type_as(preds)  # prevent zero
        log_val = log_val.view(1, -1).repeat(preds.size(0), 1)
        res_log = res_log.scatter(1, indices, log_val)

        score = score / res_log
        if self.propen_score is not None:
            score = score / (self.propen_score)

        score = score.sum(dim=1)
        self.score += score.sum()
        self.total += preds.size(0)

    def compute(self):
        return self.score / self.total


import torch
from torchmetrics import Metric


class Efficiency(Metric):
    """
    epsilon(m, R_k) = |Q(R_k) ∩ M(m)| / |Q(R_k)|

    Q(R_k): recommended APIs' union of functionalities
    M(m): target mashup functionality set
    """

    def __init__(self, mashup_tag_embedding, api_tag_embedding, top_k):
        super(Efficiency, self).__init__()
        self.topk = top_k
        self.add_state('score', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

        self.mashup_tag_embedding = mashup_tag_embedding
        self.api_tag_embedding = api_tag_embedding

    def update(self, preds, M_list):
        device = self.api_tag_embedding.device

        for pred_list, M in zip(preds, M_list):
            pred_list = pred_list[:self.topk].to(device)
            M_vec = self.mashup_tag_embedding[M.to(device)]  # 0-1 target vector

            if len(pred_list) == 0:
                continue

            # Q(R_k): 推荐集合功能并集
            Q_vec = torch.max(self.api_tag_embedding[pred_list], dim=0).values

            q_size = torch.sum(Q_vec).float()
            if q_size == 0:
                self.score += torch.tensor(0.0, device=device)
                self.total += 1
                continue

            # Q(R_k) - M(m): 推荐有、目标没有的功能
            useful = torch.sum((Q_vec > 0) & (M_vec > 0)).float()
            efficiency = useful / q_size
            self.score += efficiency

            self.total += 1

    def compute(self):
        if self.total == 0:
            return torch.tensor(0.0, device=self.score.device)
        return self.score / self.total
# class Tolerance(Metric):
#     def __init__(self, tag_set, top_k):
#         super(Tolerance, self).__init__()
#         self.topk = top_k
#         self.add_state('score', default=torch.tensor(0.0), dist_reduce_fx='sum')
#         self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
#         if os.path.exists(tag_set):
#             self.related_matrix = torch.load(tag_set)
#         else:
#             self.related_matrix = torch.zeros((self.num_api, self.num_api), dtype=torch.float32)
#             for i, j in combinations(range(len(tag_set)), 2):
#                 self.related_matrix[i, j] = jaccard_similarity(set(tag_set[i]), set(tag_set[i]))
#                 self.related_matrix[i, j] = self.related_matrix[j, i]
#
#     def update(self, preds, targets):
#         for pred_list, target_list in zip(preds[:self.topk], targets[:self.topk]):
#             target_score = []
#             # sub_matrix = self.related_matrix[target_list, :]
#             # Q = torch.sum(torch.max(sub_matrix, dim=1)[0])
#             for item1 in target_list:
#                 score = torch.sum(self.related_matrix[item1])
#                 target_score.append(score/self.related_matrix.shape[0])
#             # 我需要将target_score中的每个元素除以其中的元素和
#             # target_score = [score /  for index, score in enumerate(target_score)]
#             # total_score = sum(target_score)
#             # target_score = [score / total_score for score in target_score]
#             score = []
#             for index, pred in enumerate(pred_list):
#                 pred_score, index = torch.max(torch.tensor([self.related_matrix[pred, target] for target in target_list]), dim=0)
#                 score.append(float(pred_score) / ((torch.log2(index + 2)) * target_score[index]))
#             self.score += max(score)
#             self.total += 1
#
#     def compute(self):
#         if self.total == 0:
#             return 0
#         return self.score / self.total

class Coverage(Metric):
    def __init__(self, mashup_tag_embedding, api_tag_embedding, top_k):
        super(Coverage, self).__init__()
        self.topk = top_k
        self.add_state('score', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.mashup_tag_embedding = mashup_tag_embedding
        self.api_tag_embedding = api_tag_embedding

    def update(self, preds, M_list):
        device = self.api_tag_embedding.device
        for pred_list, M in zip(preds, M_list):
            pred_list = pred_list[:self.topk].to(device)
            M1 = self.mashup_tag_embedding[M.to(device)]   # 0-1 target vector

            # 推荐集合功能并集
            Q = torch.max(self.api_tag_embedding[pred_list], dim=0).values

            # 目标功能总数
            total = torch.sum(M1).float().clamp(min=1.0)

            # 被覆盖的目标功能数
            covered = torch.sum((M1 > 0) & (Q > 0)).float()

            gamma = covered / total
            self.score += gamma
            self.total += 1

    def compute(self):
        if self.total == 0:
            return torch.tensor(0.0, device=self.score.device)
        return self.score / self.total
# class Coverage(Metric):
#     def __init__(self, mashup_tag_embedding, api_tag_embedding, top_k):
#         super(Coverage, self).__init__()
#         self.topk = top_k
#         self.add_state('score', default=torch.tensor(0.0), dist_reduce_fx='sum')
#         self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
#         self.mashup_tag_embedding = mashup_tag_embedding
#         self.api_tag_embedding = api_tag_embedding
#
#     def update(self, preds, M_list):
#         for pred_list, M in zip(preds, M_list):
#             score = torch.tensor(0.0)
#             for i in range(1, len(pred_list)+1):
#                 Q = torch.max(self.api_tag_embedding[pred_list[:i].to(self.mashup_tag_embedding.device)], dim=0).values
#                 M1 = self.mashup_tag_embedding[M.to(self.mashup_tag_embedding.device)]
#                 score += torch.sum(torch.clamp(M1-Q, min=0)) / ((torch.sum(M1) + 1) * torch.log2(torch.tensor(2+i)))
#             self.score += score / len(pred_list+1)
#             self.total += 1
#
#     def compute(self):
#         if self.total == 0:
#             return 0
#         return self.score / self.total

class Redundancy(Metric):
    def __init__(self, api_tag_embedding, top_k):
        super(Redundancy, self).__init__()
        self.topk = top_k
        self.add_state('score', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.api_tag_embedding = api_tag_embedding

    def update(self, preds):
        device = self.api_tag_embedding.device
        for pred_list in preds:
            pred_list = pred_list[:self.topk].to(device)

            if len(pred_list) <= 1:
                self.score += torch.tensor(0.0, device=device)
                self.total += 1
                continue

            selected = self.api_tag_embedding[pred_list]   # [k, num_tags], 0-1

            # sum_i |Q(r_i)|
            q_sum = torch.sum(selected).float()

            # |Q(R_k)|
            q_union = torch.sum(torch.max(selected, dim=0).values).float()

            rho = (q_sum - q_union) / q_sum.clamp(min=1.0)

            self.score += rho
            self.total += 1

    def compute(self):
        if self.total == 0:
            return torch.tensor(0.0, device=self.score.device)
        return self.score / self.total
# class OverlapRate(Metric):
#     def __init__(self, api_tag_embedding, top_k):
#         super(OverlapRate, self).__init__()
#         self.topk = top_k
#         self.add_state('score', default=torch.tensor(0.0), dist_reduce_fx='sum')
#         self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
#         self.api_tag_embedding = api_tag_embedding
#
#     def update(self, preds):
#         for pred_list in preds:
#             score = torch.tensor(0.0)
#             for i in range(1, len(pred_list)):
#                 Q = torch.sum(torch.max(self.api_tag_embedding[pred_list[:i].to(self.api_tag_embedding.device)], dim=0).values)
#                 Q_sum = torch.sum(self.api_tag_embedding[pred_list.to(self.api_tag_embedding.device)])
#                 score += (Q_sum - Q)/((Q_sum + 1) * torch.log2(torch.tensor(1+i)))
#             self.score += score / len(pred_list)
#             self.total += 1
#
#     def compute(self):
#         if self.total == 0:
#             return 0
#         return self.score / self.total


# class SavingRate(Metric):
#     def __init__(self, mashup_tag_embedding, api_tag_embedding, top_k):
#         super(SavingRate, self).__init__()
#         self.topk = top_k
#         self.add_state('score', default=torch.tensor(0.0), dist_reduce_fx='sum')
#         self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
#         self.mashup_tag_embedding = mashup_tag_embedding
#         self.api_tag_embedding = api_tag_embedding
#
#     def update(self, preds, M_list):
#         for pred_list, M in zip(preds, M_list):
#             score = torch.tensor(0.0)
#             for i in range(1, len(pred_list)):
#                 Q = torch.max(self.api_tag_embedding[pred_list.to(self.mashup_tag_embedding.device)[:i]], dim=0).values
#                 M1 = self.mashup_tag_embedding[M.to(self.mashup_tag_embedding.device)]
#                 score += torch.sum(torch.clamp(Q-M1, min=0)) / ((torch.sum(Q) + 1) * torch.log2(torch.tensor(2 + i)))
#             self.score += score / len(pred_list)
#             self.total += 1
#
#     def compute(self):
#         if self.total == 0:
#             return 0
#         return self.score / self.total




class FaultTolerance(Metric):
    """
    Fault Tolerance:
    tau(m, R_k(m)) =
        | M(m) ∩ union_{i<j, r_i,r_j in R_k}(Q(r_i) ∩ Q(r_j)) | / |M(m)|

    等价实现：
    一个功能只要在推荐集合中被至少两个 API 同时覆盖，
    就属于 union_{i<j}(Q(r_i) ∩ Q(r_j)).
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, mashup_tag_embedding, api_tag_embedding, top_k: int):
        super().__init__()
        self.topk = top_k

        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        self.mashup_tag_embedding = mashup_tag_embedding   # [num_mashup, num_tags], 0-1
        self.api_tag_embedding = api_tag_embedding         # [num_api, num_tags], 0-1

    def update(self, preds, users):
        device = self.api_tag_embedding.device

        for pred_list, user in zip(preds, users):
            pred_list = pred_list[:self.topk].to(device)
            user = user.to(device)

            # M(m): mashup 的目标功能集合
            M_vec = self.mashup_tag_embedding[user].float()

            m_size = torch.sum(M_vec > 0).float()
            if m_size == 0:
                continue

            if len(pred_list) < 2:
                self.score += torch.tensor(0.0, device=device)
                self.total += 1
                continue

            # 选出的 API 标签矩阵: [k, num_tags]
            selected = self.api_tag_embedding[pred_list].float()

            # 被至少两个 API 同时覆盖的功能
            redundant_support = (torch.sum(selected, dim=0) >= 2).float()

            # M(m) ∩ union_{i<j}(Q(r_i) ∩ Q(r_j))
            matched = torch.sum((M_vec > 0) & (redundant_support > 0)).float()

            tau = matched / m_size

            self.score += tau
            self.total += 1

    def compute(self):
        if self.total == 0:
            return torch.tensor(0.0, dtype=torch.float32, device=self.score.device)
        return self.score / self.total
# class FaultTolerance(Metric):
#     def __init__(self, mashup_tag_embedding, api_tag_embedding, tag_num, top_k):
#         super(FaultTolerance, self).__init__()
#         self.topk = top_k
#         self.add_state('score', default=torch.tensor(0.0), dist_reduce_fx='sum')
#         self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
#         self.mashup_tag_embedding = mashup_tag_embedding
#         self.union_matrix = torch.zeros((api_tag_embedding.shape[0], api_tag_embedding.shape[0], tag_num), dtype=torch.float32)
#         for i,j in combinations(range(api_tag_embedding.shape[0]), 2):
#             self.union_matrix[i, j] = torch.min(api_tag_embedding[i], api_tag_embedding[j])
#         self.api_tag_embedding = api_tag_embedding
#
#     def update(self, preds, M_list):
#         for pred_list, M in zip(preds, M_list):
#             score = torch.tensor(0.0)
#             selected_matrix = self.union_matrix[pred_list.to(self.union_matrix.device)][:, pred_list.to(self.union_matrix.device), :]
#             for i in range(1, len(pred_list)+1):
#                 Q = torch.max(selected_matrix[:i, :i], dim=0).values
#                 Q = torch.max(Q, dim=0).values
#                 M1 = torch.stack([self.mashup_tag_embedding[M.to(self.mashup_tag_embedding.device)], Q], dim=0)
#                 score += torch.count_nonzero(torch.min(M1, dim=0).values) / ((torch.sum(self.mashup_tag_embedding[M.to(self.mashup_tag_embedding.device)]) + 1) * torch.log2(torch.tensor(2+i)))
#             self.score += score / len(pred_list)
#             self.total += 1
#
#     def compute(self):
#         if self.total == 0:
#             return 0
#         return self.score / self.total


class Count(Metric):
    def __init__(self, mashup_tag_embedding, api_tag_embedding):
        super(Count, self).__init__()
        self.add_state('score', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.mashup_tag_embedding = mashup_tag_embedding
        self.api_tag_embedding = api_tag_embedding

    def update(self, preds, M_list):
        for pred_list, M in zip(preds, M_list):
            score = torch.tensor(0.0)
            for i in range(1, len(pred_list)):
                Q = torch.max(self.api_tag_embedding[pred_list[:i].to(self.api_tag_embedding.device)], dim=0).values
                # Q = torch.max(Q, dim=1).values
                M1 = self.mashup_tag_embedding[M.to(self.mashup_tag_embedding.device)]
                if torch.sum(torch.clamp(M1 - Q, min=0)) == 0:
                    score = i
                    break
            self.score += score
            self.total += 1

    def compute(self):
        if self.total == 0:
            return 0
        return self.score / self.total


if __name__ == '__main__':
    # 示例
    # 假设 y_true 和 y_pred 是模型的输出
    y_true = torch.tensor([3, 2, 1], dtype=torch.int32).unsqueeze(0)
    y_score = torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float32).unsqueeze(0)
    y_pred = torch.tensor([10, 11, 12, 5], dtype=torch.int32).unsqueeze(0)
    # 初始化损失函数
    a = Efficiency('../../data/Mashup/label/related_matrix.pt', 5)
    a.update([[1, 2, 3]], [[6, 3, 5]])
    criterion = LambdaRankLoss('../../data/Mashup/label/related_matrix.pt')
    loss = criterion(y_score, y_pred, y_true)
    print(f"Loss: {loss.item()}")
