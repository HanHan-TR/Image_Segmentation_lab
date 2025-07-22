# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import sys
import os
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
RANK = int(os.getenv('RANK', -1))

from utils import add_prefix, resize
from models.builder import BACKBONE, SEGMENTOR, NECK, DECODEHEAD, build_module_from_cfg
from .base import BaseSegmentor


@SEGMENTOR.register()
class EncoderDecoder(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone: dict,
                 decode_head: dict,
                 neck: dict = None,
                 auxiliary_head: dict = None,
                 with_aux: bool = True,
                 train_cfg: dict = None,
                 test_cfg: dict = None,
                 pretrained: str = None,
                 init_cfg: dict = None):
        super(EncoderDecoder, self).__init__(init_cfg)
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if pretrained is not None:
            if isinstance(pretrained, str):
                self.init_cfg = dict(type='PretrainedInit', checkpoint=pretrained)
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            # backbone.pretrained = pretrained
        self.backbone = build_module_from_cfg(cfg=backbone, registry=BACKBONE)
        if neck is not None:
            self.neck = build_module_from_cfg(cfg=neck, registry=NECK)
        self._init_decode_head(decode_head)
        if with_aux and auxiliary_head:
            self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head, 'EncoderDecoder Segmentor must have the decode head.'
        self.init_weights()

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = build_module_from_cfg(decode_head, registry=DECODEHEAD)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(build_module_from_cfg(head_cfg, registry=DECODEHEAD))
            else:
                self.auxiliary_head = build_module_from_cfg(auxiliary_head, registry=DECODEHEAD)

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, img):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        out = self._decode_head_forward_test(x)
        out = resize(input=out,
                     size=img.shape[2:],  # 将输出resize到与模型输入一样的尺寸
                     mode='bilinear',
                     align_corners=self.align_corners)
        return out

    def _decode_head_forward_train(self, inputs, gt_semantic_seg, meta_infos, rescale=False):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        seg_logits, loss_decode = self.decode_head.forward_train(inputs,
                                                                 gt_semantic_seg,
                                                                 meta_infos,
                                                                 rescale=rescale)

        losses.update(add_prefix(loss_decode, 'decode'))
        return seg_logits, losses

    def _decode_head_forward_test(self, inputs):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(inputs)
        return seg_logits

    def _auxiliary_head_forward_train(self, inputs, gt_semantic_seg, meta_infos=dict(), rescale=False):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            seg_logits = dict()
            for idx, aux_head in enumerate(self.auxiliary_head):
                seg_logit_aux, loss_aux = aux_head.forward_train(inputs,
                                                                 gt_semantic_seg,
                                                                 meta_infos,
                                                                 rescale=rescale)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
                seg_logits[idx] = seg_logit_aux
        else:
            seg_logits, loss_aux = self.auxiliary_head.forward_train(inputs,
                                                                     gt_semantic_seg,
                                                                     meta_infos,
                                                                     rescale=rescale)
            losses.update(add_prefix(loss_aux, 'aux'))

        return seg_logits, losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def forward_train(self, img, gt_semantic_seg, meta_infos=dict(), rescale=False):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(img)

        seg_logits = dict()
        losses = dict()

        decode_seg_logit, loss_decode = self._decode_head_forward_train(inputs=x,
                                                                        gt_semantic_seg=gt_semantic_seg,
                                                                        meta_infos=meta_infos,
                                                                        rescale=rescale)
        seg_logits['decode'] = decode_seg_logit
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            aux_seg_logit, loss_aux = self._auxiliary_head_forward_train(inputs=x,
                                                                         gt_semantic_seg=gt_semantic_seg,
                                                                         meta_infos=meta_infos,
                                                                         rescale=rescale)
            seg_logits['aux'] = aux_seg_logit
            losses.update(loss_aux)

        return seg_logits, losses

    # TODO refactor
    def slide_inference(self, img, ori_img_size=None, rescale=True):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            # remove padding area
            # resize_shape = img_meta[0]['img_shape'][:2]
            # preds = preds[:, :, :resize_shape[0], :resize_shape[1]]
            preds = resize(preds,
                           size=ori_img_size,  # img_meta[0]['ori_shape'][:2],
                           mode='bilinear',
                           align_corners=self.align_corners,
                           warning=False)
        return preds

    def whole_inference(self, img, ori_img_size=None, rescale=True):
        """Inference with full image."""

        seg_logit = self.encode_decode(img)  # 与模型的输入尺寸相同
        if rescale and ori_img_size:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                # remove padding area
                # resize_shape = img_meta[0]['img_shape'][:2]
                # seg_logit = seg_logit[:, :, :resize_shape[0], :resize_shape[1]]
                size = ori_img_size

            seg_logit = resize(seg_logit,
                               size=size,
                               mode='bilinear',
                               align_corners=self.align_corners,
                               warning=False)

        return seg_logit

    def inference(self, img, ori_img_size=None, rescale=True, mode: str = 'whole'):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        # assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = ori_img_size
        # assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if mode == 'slide':
            seg_logit = self.slide_inference(img, ori_shape, rescale)
        else:
            seg_logit = self.whole_inference(img, ori_shape, rescale)
        if self.out_channels == 1:
            # σ(x) = 1 / (1 + e⁻ˣ) 将任意实数映射到 (0,1) 区间，输出值表示概率或置信度
            output = F.sigmoid(seg_logit)
        else:
            # softmax(xᵢ) = eˣⁱ / Σⱼeˣʲ 将多个输出值转换为概率分布，所有通道的概率和为一
            output = F.softmax(seg_logit, dim=1)

        return output

    def simple_test(self, img, ori_img_size=None, rescale=True):
        """Simple test with single image."""

        seg_logit = self.inference(img, ori_img_size=ori_img_size, rescale=rescale)
        if self.out_channels == 1:
            seg_pred = (seg_logit > self.decode_head.threshold).to(seg_logit).squeeze(1)
        else:
            seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        # seg_pred = list(seg_pred)
        # return seg_pred
        return seg_logit

    def simple_test_logits(self, img, img_metas, rescale=True):
        """Test without augmentations.

        Return numpy seg_map logits.
        """
        seg_logit = self.inference(img[0], img_metas[0], rescale)
        seg_logit = seg_logit.cpu().numpy()
        return seg_logit

    def batch_test(self, imgs, ori_img_size=None, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        # assert rescale
        # to save memory, we get augmented seg logit inplace
        # seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        seg_logits = []
        for i in range(1, len(imgs)):
            single_seg_logit = self.simple_test(imgs[i], ori_img_size=ori_img_size[i], rescale=rescale)
            seg_logits.append(single_seg_logit)
        # seg_logit /= len(imgs)
        # if self.out_channels == 1:
        #     seg_pred = (seg_logit > self.decode_head.threshold).to(seg_logit).squeeze(1)
        # else:
        #     seg_pred = seg_logit.argmax(dim=1)
        # seg_pred = seg_pred.cpu().numpy()
        # # unravel batch dim
        # seg_pred = list(seg_pred)
        return seg_logits

    def aug_test_logits(self, img, img_metas, rescale=True):
        """Test with augmentations.

        Return seg_map logits. Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale

        imgs = img
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit

        seg_logit /= len(imgs)
        seg_logit = seg_logit.cpu().numpy()
        return seg_logit
