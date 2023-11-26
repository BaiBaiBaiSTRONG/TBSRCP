# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-08-08 14:31:30
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-05-25 09:13:32
# @Email:  cshzxie@gmail.com

# @Author: Hanyang Zhou
# @Date:   2023-02-07 23:41:30
# @Last Modified by:   Hanyang Zhou
# @Last Modified time: 2023-02-07 23:41:30
# @Email:  cshzxie@gmail.com

import logging
import open3d

from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
import torch.nn.functional as F
from torch import argmax
from sklearn.metrics import confusion_matrix

AddPointcloudMatch = True

class Metrics(object):
    ITEMS = [
    #     {
    #     'name': 'F-Score',
    #     'enabled': True,
    #     'eval_func': 'cls._get_Rebuld_f_score',
    #     'is_greater_better': True,
    #     'init_value': 0
    # },
    {
        'name': 'CDL1',
        'enabled': True,
        'eval_func': 'cls._get_chamfer_distancel1',
        'eval_object': ChamferDistanceL1(ignore_zeros=True),
        'is_greater_better': False,
        'init_value': 32767
    }, {
        'name': 'CDL2',
        'enabled': True,
        'eval_func': 'cls._get_chamfer_distancel2',
        'eval_object': ChamferDistanceL2(ignore_zeros=True),
        'is_greater_better': False,
        'init_value': 32767
    },{
        'name':'ReduildAccuracy',
        'enabled': True,
        'eval_func': 'cls._get_rebuild_accuracy',
        'is_greater_better': True,
        'init_value': 0
    },
    # {
    #     'name':'ReduildPrecision',
    #     'enabled': True,
    #     'eval_func': 'cls._get_rebuild_precision',
    #     'is_greater_better': True,
    #     'init_value': 0
    # },{
    #     'name':'ReduildRecall',
    #     'enabled': True,
    #     'eval_func': 'cls._get_rebuild_recall',
    #     'is_greater_better': True,
    #     'init_value': 0
    # },{
    #     'name':'ClsLoss',
    #     'enabled': True,
    #     'eval_func': 'cls._get_clssfication_loss',
    #     'is_greater_better': True,
    #     'init_value': 0
    # },
    {
        'name':'ClsAcc',
        'enabled': True,
        'eval_func': 'cls._get_clssfication_acc',
        'is_greater_better': True,
        'init_value': 0
    }
    # ,{
    #     'name':'ClsFscore',
    #     'enabled': True,
    #     'eval_func': 'cls._get_clssfication_fscore',
    #     'is_greater_better': True,
    #     'init_value': 0
    # },{
    #     'name':'ClsPrecision',
    #     'enabled': True,
    #     'eval_func': 'cls._get_clssfication_precision',
    #     'is_greater_better': True,
    #     'init_value': 0
    # },{
    #     'name':'ClsRecall',
    #     'enabled': True,
    #     'eval_func': 'cls._get_clssfication_recall',
    #     'is_greater_better': True,
    #     'init_value': 0
    # }
    ]

    @classmethod
    def get(cls, pred, gt, cls_pred, y_cls_true):
        _items = cls.items()
        _values = [0] * len(_items)
        for i, item in enumerate(_items):
            eval_func = eval(item['eval_func'])
            if AddPointcloudMatch:
                if item['name'].startswith('Cls'):
                    _values[i] = eval_func(cls_pred, y_cls_true)
                else:
                    _values[i] = eval_func(pred, gt)
            else:
                if item['name'].startswith('Cls'):
                    _values[i] = eval_func(cls_pred, y_cls_true)

        return _values

    @classmethod
    def items(cls):
        return [i for i in cls.ITEMS if i['enabled']]

    @classmethod
    def names(cls):
        _items = cls.items()
        return [i['name'] for i in _items]
    
    # @classmethod
    # def _get_clssfication_loss(cls, cls_loss):
    #     b = cls_loss.size(0)
    #     cls_loss_list=[]
    #     for idx in range(b):
    #         cls_loss_list.append(cls_loss[idx])
    #     return sum(cls_loss_list)/len(cls_loss_list)

    @classmethod
    def _get_clssfication_loss(cls,cls_pred, y_cls_true):
        b = cls_pred.size(0)
        cls_loss_list=[]
        for idx in range(b):
            # 2's 32x28 loss comp
            cls_loss = F.binary_cross_entropy_with_logits(cls_pred[idx], y_cls_true[idx])
            cls_loss_list.append(cls_loss)
        return sum(cls_loss_list)/len(cls_loss_list)
    
    @classmethod
    def _get_clssfication_acc(cls,cls_pred, y_cls_true):
        # print('==== IN ACCURACY TEST ====')
        # print(cls_pred.shape, y_cls_true.shape)
        # print(cls_pred[0])
        # print(y_cls_true[0])

        b = cls_pred.size(0)
        correct = 0

        for idx in range(b):
            y_pred = argmax(cls_pred[idx]) # 2
            y_true = argmax(y_cls_true[idx]) # 5
            if y_pred == y_true:
                correct += 1
        return correct / b

    @classmethod
    def _get_clssfication_recall(cls,cls_pred, y_cls_true):
        b = cls_pred.size(0)
        tp=0
        fn=0
        for idx in range(b):
            y_pred = argmax(cls_pred[idx])
            y_true = argmax(y_cls_true[idx])
            # calculate tp
            if y_pred == y_true:
                tp += 1
            # calculate multicategory fn
            else:
                fn += 1
        return tp/(tp+fn)

    @classmethod
    def _get_clssfication_precision(cls,cls_pred, y_cls_true):
        b = cls_pred.size(0)
        tp=0
        fp=0
        for idx in range(b):
            y_pred = argmax(cls_pred[idx])
            y_true = argmax(y_cls_true[idx])
            if y_pred == y_true:
                tp += 1
            else:
                fp += 1
        return tp/(tp+fp)

    @classmethod
    def _get_clssfication_fscore(cls,cls_pred, y_cls_true, th=0.53):
        return 0

    # @classmethod
    # def _get_clssfication_auc(cls,cls_pred, y_cls_true):

    @classmethod
    def _get_Rebuld_f_score(cls, pred, gt, th=0.01):

        """References: https://github.com/lmb-freiburg/what3d/blob/master/util.py"""
        b = pred.size(0)
        assert pred.size(0) == gt.size(0)
        if b != 1:
            f_score_list = []
            for idx in range(b):
                f_score_list.append(cls._get_Rebuld_f_score(pred[idx:idx+1], gt[idx:idx+1]))
            return sum(f_score_list)/len(f_score_list)
        else:
            pred = cls._get_open3d_ptcloud(pred)
            gt = cls._get_open3d_ptcloud(gt)

            dist1 = pred.compute_point_cloud_distance(gt)
            dist2 = gt.compute_point_cloud_distance(pred)

            recall = float(sum(d < th for d in dist2)) / float(len(dist2))
            precision = float(sum(d < th for d in dist1)) / float(len(dist1))
            return 2 * recall * precision / (recall + precision) if recall + precision else 0

    @classmethod
    def _get_open3d_ptcloud(cls, tensor):
        """pred and gt bs is 1"""
        tensor = tensor.squeeze().cpu().numpy()
        ptcloud = open3d.geometry.PointCloud()
        ptcloud.points = open3d.utility.Vector3dVector(tensor)

        return ptcloud
    @classmethod
    def _get_rebuild_precision(cls, pred, gt, th=0.01):
        """References: https://github.com/lmb-freiburg/what3d/blob/master/util.py"""
        b = pred.size(0)
        assert pred.size(0) == gt.size(0)
        if b != 1:
            precision_list = []
            for idx in range(b):
                precision_list.append(cls._get_rebuild_precision(pred[idx:idx+1], gt[idx:idx+1]))
            return sum(precision_list)/len(precision_list)
        else:
            pred = cls._get_open3d_ptcloud(pred)
            gt = cls._get_open3d_ptcloud(gt)

            dist1 = pred.compute_point_cloud_distance(gt)
            return float(sum(d < th for d in dist1)) / float(len(dist1))
    
    @classmethod
    def _get_rebuild_recall(cls, pred, gt, th=0.01):
        """References: https://github.com/lmb-freiburg/what3d/blob/master/util.py"""
        b = pred.size(0)
        assert pred.size(0) == gt.size(0)
        if b != 1:
            recall_list = []
            for idx in range(b):
                recall_list.append(cls._get_rebuild_recall(pred[idx:idx+1], gt[idx:idx+1]))
            return sum(recall_list)/len(recall_list)
        else:
            pred = cls._get_open3d_ptcloud(pred)
            gt = cls._get_open3d_ptcloud(gt)

            dist2 = gt.compute_point_cloud_distance(pred)
            return float(sum(d < th for d in dist2)) / float(len(dist2))

    @classmethod
    def _get_rebuild_accuracy(cls, pred, gt, th=0.01):
        b= pred.size(0)
        assert pred.size(0) == gt.size(0)
        if b != 1:
            accuracy_list = []
            for idx in range(b):
                accuracy_list.append(cls._get_rebuild_accuracy(pred[idx:idx+1], gt[idx:idx+1]))
            return sum(accuracy_list)/len(accuracy_list)
        else:
            pred = cls._get_open3d_ptcloud(pred)
            gt = cls._get_open3d_ptcloud(gt)

            dist1 = pred.compute_point_cloud_distance(gt)
            dist2 = gt.compute_point_cloud_distance(pred)
            
            return float(sum(d < th for d in dist1) + sum(d < th for d in dist2)) / float(len(dist1) + len(dist2))
           

    @classmethod
    def _get_chamfer_distancel1(cls, pred, gt):
        chamfer_distance = cls.ITEMS[0]['eval_object']
        return chamfer_distance(pred, gt).item() * 1000

    @classmethod
    def _get_chamfer_distancel2(cls, pred, gt):
        chamfer_distance = cls.ITEMS[1]['eval_object']
        return chamfer_distance(pred, gt).item() * 1000

    def __init__(self, metric_name, values):
        self._items = Metrics.items()
        self._values = [item['init_value'] for item in self._items]
        self.metric_name = metric_name

        if type(values).__name__ == 'list':
            self._values = values
        elif type(values).__name__ == 'dict':
            metric_indexes = {}
            for idx, item in enumerate(self._items):
                item_name = item['name']
                metric_indexes[item_name] = idx
            for k, v in values.items():
                if k not in metric_indexes:
                    logging.warn('Ignore Metric[Name=%s] due to disability.' % k)
                    continue
                self._values[metric_indexes[k]] = v
        else:
            raise Exception('Unsupported value type: %s' % type(values))

    def state_dict(self):
        _dict = dict()
        for i in range(len(self._items)):
            item = self._items[i]['name']
            value = self._values[i]
            _dict[item] = value

        return _dict

    def __repr__(self):
        return str(self.state_dict())

    def better_than(self, other):
        if other is None:
            return True

        _index = -1
        for i, _item in enumerate(self._items):
            if _item['name'] == self.metric_name:
                _index = i
                break
        if _index == -1:
            raise Exception('Invalid metric name to compare.')

        _metric = self._items[i]
        _value = self._values[_index]
        other_value = other._values[_index]
        return _value > other_value if _metric['is_greater_better'] else _value < other_value
