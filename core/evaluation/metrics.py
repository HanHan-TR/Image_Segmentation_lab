# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Sequence
from unittest import result

import numpy as np
import torch
import torch.nn.functional as F
import sys
import os
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
RANK = int(os.getenv('RANK', -1))

from core.fileio import mkdir_or_exist
from PIL import Image
from prettytable import PrettyTable


class SegEvaluator():
    """ IoU evaluation metric.

    Args:
        ignore_index (int): Index that will be ignored in evaluation.
            Default: 255.
        iou_metrics (list[str] | str): Metrics to be calculated, the options
            includes 'mIoU', 'mDice' and 'mFscore'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        beta (int): Determines the weight of recall in the combined score.
            Default: 1.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        output_dir (str): The directory for output prediction. Defaults to
            None.
        format_only (bool): Only format result for results commit without
            perform evaluation. It is useful when you want to save the result
            to a specific format and submit it to the test server.
            Defaults to False.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    def __init__(self,
                 ignore_index: int = 255,
                 iou_metrics: List[str] = ['mIoU', 'mDice', 'mFscore'],
                 nan_to_num: Optional[int] = None,
                 beta: int = 1,  # beta=1，默认计算 F1-score
                 # collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__()

        self.ignore_index = ignore_index
        self.metrics = iou_metrics  # 选择计算的指标类型 （mIOU/mDice/mFscore）
        self.nan_to_num = nan_to_num
        self.beta = beta  # F-score计算中的精确率与召回率的权重比
        self.output_dir = output_dir  # 预测结果输出路径 （用于可视化）
        self.prefix = prefix
        if self.output_dir:
            mkdir_or_exist(self.output_dir)
        self.format_only = format_only

        self.results = defaultdict(list)  # 用于缓存计算结果  area_intersect, area_union, area_pred_label, area_label

    def process(self, pred_batch: dict, labels_batch: Sequence[torch.tensor]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        num_classes = 2  # len(self.dataset_meta['classes'])  # 目标类别数
        for key, value in pred_batch.items():
            for i in range(len(value)):
                pred_batch[key][i] = F.softmax(value[i], dim=0).argmax(dim=0)

        # labels_batch = [label.squeeze(0) for label in labels_batch]

        for key, value in pred_batch.items():
            # if key not in results:
            #     results[key] = []

            self.results[key].append(self.intersect_and_union(pred_labels=value,
                                                              labels_gt=labels_batch.copy(),
                                                              num_classes=num_classes,
                                                              ignore_index=self.ignore_index))
        # for data_sample in data_samples:
        #     pred_label = data_sample['pred_sem_seg']['data'].squeeze()
        #     # format_only always for test dataset without ground truth
        #     if not self.format_only:
        #         label = data_sample['gt_sem_seg']['data'].squeeze().to(pred_label)
        #         self.results.append(self.intersect_and_union(pred_label,
        #                                                      label,
        #                                                      num_classes,
        #                                                      self.ignore_index))
        #     # format_result
        #     if self.output_dir is not None:
        #         basename = osp.splitext(osp.basename(data_sample['img_path']))[0]
        #         png_filename = osp.abspath(osp.join(self.output_dir, f'{basename}.png'))
        #         output_mask = pred_label.cpu().numpy()
        #         # The index range of official ADE20k dataset is from 0 to 150.
        #         # But the index range of output is from 0 to 149.
        #         # That is because we set reduce_zero_label=True.
        #         if data_sample.get('reduce_zero_label', False):
        #             output_mask = output_mask + 1
        #         output = Image.fromarray(output_mask.astype(np.uint8))
        #         output.save(png_filename)

    def compute_metrics(self):
        if isinstance(self.results, list):
            metrics_results = self.seg_metrics(self.results)
        elif isinstance(self.results, dict):
            metrics_results = dict()
            for key, value in self.results.items():
                assert isinstance(value, list), "the values in the results dict of SegEvaluator must be a list"
                print('-------------------------' + key + '-------------------------')
                metrics_results[key] = self.seg_metrics(value)
        else:
            raise TypeError('the results of SegEvaluator must be a list or dict')
        return metrics_results

    def seg_metrics(self, results: list) -> Dict[str, float]:  # 在运行完所有batch后使用
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
                mRecall.
        """
        # 根据中间结果计算最终指标（抽象方法，子类必须实现）
        # logger: MMLogger = MMLogger.get_current_instance()
    #    results = self.results
        if self.format_only:
            # logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()
        # convert list of tuples to tuple of lists, e.g.
        # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to ([A_1, ..., A_n], ..., [D_1, ..., D_n])
        results = tuple(zip(*results))  # 解压所有batch的结果，将[(a,b,c,d),...]转为([a...],[b...],...)
        assert len(results) == 4

        # 累加所有batch的统计量
        total_area_intersect = sum(results[0])  # 交集
        total_area_union = sum(results[1])  # 并集
        total_area_pred_label = sum(results[2])  # 预测区域
        total_area_label = sum(results[3])  # 标签区域

        # 调用指标转换方法
        ret_metrics = self.total_area_to_metrics(total_area_intersect,
                                                 total_area_union,
                                                 total_area_pred_label,
                                                 total_area_label,
                                                 self.metrics,
                                                 self.nan_to_num,
                                                 self.beta)

        class_names = self.dataset_meta['classes']

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        metrics = dict()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                metrics[key] = val
            else:
                metrics['m' + key] = val

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)

        # 生成表格输出
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        # print_log('per class results:', logger)
        # print_log('\n' + class_table_data.get_string(), logger=logger)

        return metrics

    @staticmethod
    def intersect_and_union(pred_labels: list, labels_gt: list,
                            num_classes: int, ignore_index: int):
        """Calculate Intersection and Union.

        Args:
            pred_label (torch.tensor): Prediction segmentation map
                or predict result filename. The shape is (H, W).
            label (torch.tensor): Ground truth segmentation map
                or label filename. The shape is (H, W).
            num_classes (int): Number of categories.
            ignore_index (int): Index that will be ignored in evaluation.

        Returns:
            torch.Tensor: The intersection of prediction and ground truth
                histogram on all classes.
            torch.Tensor: The union of prediction and ground truth histogram on
                all classes.
            torch.Tensor: The prediction histogram on all classes.
            torch.Tensor: The ground truth histogram on all classes.
        """

        # mask = (label != ignore_index)  # 创建有效区域掩码
        # pred_label = pred_label[mask]  # 过滤掉预测结果中的被忽略的区域
        # label = label[mask]  # 过滤掉标签中的被忽略的区域

        assert len(pred_labels) == len(labels_gt)
        for idx, label in enumerate(labels_gt):
            mask = (label != ignore_index)
            # masks.append(mask)
            labels_gt[idx] = label[mask]  # 过滤掉标签中的被忽略的区域
            pred_labels[idx] = pred_labels[idx][mask]

        area_intersect, area_pred_label, area_label, area_union = [], [], [], []

        for i in range(len(labels_gt)):
            gt = labels_gt[i].cuda()
            pred_label = pred_labels[i]
            intersect = pred_label[pred_label == gt]
            area_inter = torch.histc(intersect.float(),
                                     bins=(num_classes),
                                     min=0,
                                     max=num_classes - 1).cpu()
            area_intersect.append(area_inter)

            # 计算预测结果直方图
            area_pred = torch.histc(pred_label.float(),
                                    bins=(num_classes),
                                    min=0,
                                    max=num_classes - 1).cpu()
            area_pred_label.append(area_pred)

            area_gt = torch.histc(label.float(),
                                  bins=(num_classes),
                                  min=0,
                                  max=num_classes - 1).cpu()
            area_label.append(area_gt)

            area_union.append(area_gt + area_pred - area_inter)
        return area_intersect, area_union, area_pred_label, area_label

    @staticmethod
    def total_area_to_metrics(total_area_intersect: np.ndarray,
                              total_area_union: np.ndarray,
                              total_area_pred_label: np.ndarray,
                              total_area_label: np.ndarray,
                              metrics: List[str] = ['mIoU'],
                              nan_to_num: Optional[int] = None,
                              beta: int = 1):
        """Calculate evaluation metrics
        Args:
            total_area_intersect (np.ndarray): The intersection of prediction
                and ground truth histogram on all classes.
            total_area_union (np.ndarray): The union of prediction and ground
                truth histogram on all classes.
            total_area_pred_label (np.ndarray): The prediction histogram on
                all classes.
            total_area_label (np.ndarray): The ground truth histogram on
                all classes.
            metrics (List[str] | str): Metrics to be evaluated, 'mIoU' and
                'mDice'.
            nan_to_num (int, optional): If specified, NaN values will be
                replaced by the numbers defined by the user. Default: None.
            beta (int): Determines the weight of recall in the combined score.
                Default: 1.
        Returns:
            Dict[str, np.ndarray]: per category evaluation metrics,
                shape (num_classes, ).
        """

        def f_score(precision, recall, beta=1):
            """calculate the f-score value.

            Args:
                precision (float | torch.Tensor): The precision value.
                recall (float | torch.Tensor): The recall value.
                beta (int): Determines the weight of recall in the combined
                    score. Default: 1.

            Returns:
                [torch.tensor]: The f-score value.
            """
            score = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
            return score

        if isinstance(metrics, str):
            metrics = [metrics]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metrics).issubset(set(allowed_metrics)):
            raise KeyError(f'metrics {metrics} is not supported')

        all_acc = total_area_intersect.sum() / total_area_label.sum()  # aAcc 全局准确率 （计算结果是一个浮点数，综合了所有类别）
        ret_metrics = OrderedDict({'aAcc': all_acc})

        # m 表示‘mean’，平均，这里的平均不是不同类别上的平均，而是每个类别对应的指标在整个数据集上的平均。
        # 因此以下所有指标的计算结果均为一个长度对于类别数的向量
        # 例如：mIoU的意思是，对于特定类别的分割目标，每个分割结果都有一个IOU，mIOU即是该类别所有分割结果IOU的平均值
        for metric in metrics:
            if metric == 'mIoU':
                iou = total_area_intersect / total_area_union
                acc = total_area_intersect / total_area_label
                ret_metrics['IoU'] = iou
                ret_metrics['Acc'] = acc
            elif metric == 'mDice':
                dice = 2 * total_area_intersect / (total_area_pred_label + total_area_label)
                acc = total_area_intersect / total_area_label
                ret_metrics['Dice'] = dice
                ret_metrics['Acc'] = acc
            elif metric == 'mFscore':
                precision = total_area_intersect / total_area_pred_label  # Precision：
                recall = total_area_intersect / total_area_label
                f_value = torch.tensor([f_score(x[0], x[1], beta) for x in zip(precision, recall)])
                ret_metrics['Fscore'] = f_value
                ret_metrics['Precision'] = precision
                ret_metrics['Recall'] = recall

        ret_metrics = {
            metric: value.numpy()
            for metric, value in ret_metrics.items()
        }
        if nan_to_num is not None:
            ret_metrics = OrderedDict({
                metric: np.nan_to_num(metric_value, nan=nan_to_num)
                for metric, metric_value in ret_metrics.items()
            })
        return ret_metrics
