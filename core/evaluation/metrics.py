# Copyright (c) OpenMMLab. All rights reserved.
import math
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Sequence, Union
import cv2 as cv
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
                 epoch: int,
                 num_classes: int,
                 class_names: List[str],
                 palette: Sequence[Sequence[int]],
                 ignore_index: int = 255,
                 iou_metrics: List[str] = ['mIoU', 'mDice', 'mFscore'],
                 nan_to_num: Optional[int] = None,
                 beta: int = 1,  # beta=1，默认计算 F1-score
                 # collect_device: str = 'cpu',
                 show_result: bool = True,
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__()
        self.epoch = epoch
        self.num_classes = num_classes
        self.class_names = class_names
        self.palette = palette
        self.ignore_index = ignore_index
        self.metrics = iou_metrics  # 选择计算的指标类型 （mIOU/mDice/mFscore）
        self.nan_to_num = nan_to_num
        self.beta = beta  # F-score计算中的精确率与召回率的权重比
        self.show_result = show_result
        self.output_dir = output_dir  # 预测结果输出路径 （用于可视化）
        self.prefix = prefix
        if self.output_dir:
            mkdir_or_exist(self.output_dir)
        self.format_only = format_only

        self.results = dict()  # 用于缓存计算结果  area_intersect, area_union, area_pred_label, area_label

    def process(self,
                batch_idx: int,
                pred_batch: Union[torch.tensor, List[torch.tensor]],
                batch_infos: dict) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        num_classes = self.num_classes
        labels_batch = batch_infos['ori_gt']

        for key, value in pred_batch.items():
            for i in range(len(value)):  # list
                if self.num_classes == 1:
                    pred = F.sigmoid(value[i]).argmax(dim=1)
                else:
                    pred = F.softmax(value[i], dim=1).argmax(dim=1)
                pred_batch[key][i] = pred.squeeze(0)  # tensor: (H, W)

        if self.show_result and batch_idx < 4:
            self.plot_results(batch_idx=batch_idx,
                              pred_batch=pred_batch,
                              batch_infos=batch_infos)

        for key, value in pred_batch.items():
            if key not in self.results:
                self.results[key] = [[], [], [], []]
            area_intersect, area_union, area_pred_label, area_label = self.intersect_and_union(pred_labels=value,
                                                                                               labels_gt=labels_batch.copy(),
                                                                                               num_classes=num_classes,
                                                                                               ignore_index=self.ignore_index)
            self.results[key][0].extend(area_intersect)
            self.results[key][1].extend(area_union)
            self.results[key][2].extend(area_pred_label)
            self.results[key][3].extend(area_label)

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
        # results = self.results
        # if self.format_only:
        #     # logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
        #     return OrderedDict()
        # # convert list of tuples to tuple of lists, e.g.
        # # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to ([A_1, ..., A_n], ..., [D_1, ..., D_n])
        # results = tuple(zip(*results))  # 解压所有batch的结果，将[(a,b,c,d),...]转为([a...],[b...],...)
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
        ret_metrics_class.update({'Class': self.class_names})
        ret_metrics_class.move_to_end('Class', last=False)

        # 生成表格输出
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            if key != self.class_names[self.ignore_index]:
                class_table_data.add_column(key, val)

        # print_log('per class results:', logger)
        print('\n' + class_table_data.get_string())

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
            intersect = pred_label[pred_label == gt]  # intersect 包含的是预测标签中与真实标签匹配的所有像素值
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

            area_gt = torch.histc(gt.float(),
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

    def plot_results(self,
                     batch_idx,
                     pred_batch,
                     batch_infos,
                     opacity=0.5):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor): The semantic segmentation results to draw over
                `img`.
            palette (list[list[int]]] | np.ndarray | None): The palette of
                segmentation map. If None is given, random palette will be
                generated. Default: None
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.
            opacity(float): Opacity of painted segmentation map.
                Default 0.5.
                Must be in (0, 1] range.
        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """

        if self.palette is None:
            state = np.random.get_state()
            np.random.seed(42)
            # random palette
            palette = np.random.randint(
                0, 255, size=(len(self.num_classes), 3))
            np.random.set_state(state)
        else:
            palette = self.palette

        palette = np.array(palette)
        assert palette.shape[0] == self.num_classes
        assert palette.shape[1] == 3
        assert len(palette.shape) == 2
        assert 0 < opacity <= 1.0

        for key, value in pred_batch.items():
            out_dir = Path(self.output_dir) / 'pred_results'
            out_dir.mkdir(parents=True, exist_ok=True)
            processed_images = []
            for i in range(min(len(value), 16)):
                img_name = Path(batch_infos['img_file_path'][i]).name
                img = cv.imread(batch_infos['img_file_path'][i], cv.COLOR_BGR2RGB)
                seg = value[i].cpu()

                color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
                for label, color in enumerate(palette):
                    color_seg[seg == label, :] = color
                # convert to BGR
                color_seg = color_seg[..., ::-1]

                img = img * (1 - opacity) + color_seg * opacity
                img = img.astype(np.uint8)

                processed_images.append(img)

            MAX_SIZE = 1920  # 最终图像最大尺寸

            # 计算所有图像的平均尺寸作为基准
            avg_width = sum(img.shape[1] for img in processed_images) // len(processed_images)
            avg_height = sum(img.shape[0] for img in processed_images) // len(processed_images)

            # 自动计算行列布局（基于图像数量）
            num_images = len(processed_images)
            aspect_ratio = avg_width / avg_height

            # 动态计算最优行列数
            if aspect_ratio > 1:  # 宽大于高
                num_cols = min(4, num_images)
                num_rows = math.ceil(num_images / num_cols)
            else:  # 高大于宽
                num_rows = min(4, num_images)
                num_cols = math.ceil(num_images / num_rows)

            # 计算单图最大允许尺寸
            max_cell_width = min(MAX_SIZE // num_cols, avg_width)
            max_cell_height = min(MAX_SIZE // num_rows, avg_height)

            # 调整所有图像到统一尺寸（保持宽高比）
            resized_images = []
            for img in processed_images:
                h, w = img.shape[:2]
                scale = min(max_cell_width / w, max_cell_height / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                resized = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_AREA)
                resized_images.append(resized)

            # 使用统一尺寸创建画布
            cell_height = max(img.shape[0] for img in resized_images)
            cell_width = max(img.shape[1] for img in resized_images)

            # 最终画布尺寸计算（不超过最大尺寸）
            total_width = min(num_cols * cell_width, MAX_SIZE)
            total_height = min(num_rows * cell_height, MAX_SIZE)

            combined_img = np.zeros((total_height, total_width, 3), dtype=np.uint8)

            # 拼接图像（自动居中放置）
            for idx, img in enumerate(resized_images):
                row = idx // num_cols
                col = idx % num_cols
                y_offset = row * cell_height
                x_offset = col * cell_width

                # 居中放置
                y_center = (cell_height - img.shape[0]) // 2
                x_center = (cell_width - img.shape[1]) // 2

                combined_img[
                    y_offset + y_center: y_offset + y_center + img.shape[0],
                    x_offset + x_center: x_offset + x_center + img.shape[1]
                ] = img

            # 保存图像
            combined_file = out_dir / f"pred_epoch_{self.epoch}_batch_{batch_idx}_{key}.jpg"
            cv.imwrite(str(combined_file), combined_img)
