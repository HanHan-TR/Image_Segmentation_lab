# Copyright (c) OpenMMLab. All rights reserved.
from .class_names import get_classes, get_palette
# from .eval_hooks import DistEvalHook, EvalHook
from .metrics import SegEvaluator

__all__ = [
    # 'EvalHook', 'DistEvalHook',
    # 'mean_dice', 'mean_iou', 'mean_fscore', 'eval_metrics',
    'get_classes', 'get_palette',
    # 'pre_eval_to_metrics','intersect_and_union',
    'SegEvaluator'
]
