# 训练策略选择
# 当数据集中明确指定了Train、val文件夹时，就按照固定的train、val文件夹去进行训练和训练过程中的验证
# 当数据集中只提供了train文件夹，或者说并没有通过文件夹等确定的形式对数据集进行训练集与验证集的明确划分时，可以支持两种训练形式：
# 1. 支持用户手动指定一个比例，然后按照比例首先将数据随机划分成一个训练集和一个验证集，并生成train.txt与val.txt文件
#    然后再分别在训练集与验证集上进行训练与测试。整个过程中保持训练数据集与测试数据集不改变；
# 2. 支持 K折交叉验证。 用户可以指定折数K，假设K=5，则会将整个数据平均分成5份，并随机挑选一份作为验证集. 当训练进行了一段时间之后，在选择另外一份作为验证集
#    将其余的部分作为训练集。如此得以提高数据利用率
from collections import OrderedDict
from tqdm import tqdm
from cv2 import log, mean
import torch
import torch.distributed as dist
import torch.nn.functional as F
from typing import Optional
import sys
import os
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
RANK = int(os.getenv('RANK', -1))

from utils.ops import add_prefix
from core.initialize import weights_to_cpu
TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'  # tqdm bar format


def parse_losses(losses):
    """Parse the raw outputs (losses) of the network.

    Args:
        losses (dict): Raw output of the network, which usually contain
            losses and other necessary information.

    Returns:
        tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
            which may be a weighted sum of all losses, log_vars contains
            all the variables to be sent to the logger.
    """
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                f'{loss_name} is not a tensor or list of tensors')

    loss = sum(_value for _key, _value in log_vars.items()
               if 'loss' in _key)

    # If the loss_vars has different length, raise assertion error
    # to prevent GPUs from infinite waiting.
    if dist.is_available() and dist.is_initialized():
        log_var_length = torch.tensor(len(log_vars), device=loss.device)
        dist.all_reduce(log_var_length)
        message = (f'rank {dist.get_rank()}' + f' len(log_vars): {len(log_vars)}' + ' keys: '
                   + ','.join(log_vars.keys()) + '\n')
        assert log_var_length == len(log_vars) * dist.get_world_size(), \
            'loss log variables are different across GPUs!\n' + message

    log_vars['loss'] = loss
    for loss_name, loss_value in log_vars.items():
        # reduce loss when distributed training
        if dist.is_available() and dist.is_initialized():
            loss_value = loss_value.data.clone()
            dist.all_reduce(loss_value.div_(dist.get_world_size()))
        log_vars[loss_name] = loss_value.item()

    return loss, log_vars


def train_one_epoch(epoch, model, data_loader, optimizer, scaler, schedule_cfg):
    model.train()
    mean_log_vars = OrderedDict()

    process_bar = tqdm(enumerate(data_loader), total=len(data_loader), bar_format=TQDM_BAR_FORMAT)
    for i, (images, labels, infos) in process_bar:
        images = images.to('cuda')
        labels = labels.to('cuda')
        with torch.cuda.amp.autocast(enabled=schedule_cfg.get('amp', False)):
            seg_logits, losses = model(images, labels, infos, rescale=False, return_loss=True)
            losses, log_var = parse_losses(losses)
        optimizer.zero_grad()
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        current_lr = optimizer.param_groups[0]['lr']

        for name, value in log_var.items():
            if name not in mean_log_vars:
                mean_log_vars[name] = 0.0

            mean_log_vars[name] = (i * mean_log_vars[name] + value) / (i + 1)    # 实时计算并更新平均值

        log_var_str = ", ".join(f"mean {name}:{value :.3f}" for name, value in mean_log_vars.items())
        process_bar.desc = f"epoch[{epoch + 1}/{schedule_cfg.get('epochs', 50)}] {log_var_str} lr:{current_lr}"

    return mean_log_vars


def validate_one_epoch(epoch, model, data_loader, evaluator, schedule_cfg):
    model.eval()
    mean_log_vars = OrderedDict()

    process_bar = tqdm(enumerate(data_loader), total=len(data_loader), bar_format=TQDM_BAR_FORMAT)
    for i, (images, labels, infos) in process_bar:
        images = images.to('cuda')
        labels = labels.to('cuda')
        with torch.no_grad():
            seg_logits, losses = model(images, labels, infos, rescale=True, return_loss=True)
            _, log_var = parse_losses(losses)

        for name, value in log_var.items():
            if name not in mean_log_vars:
                mean_log_vars[name] = 0.0

            mean_log_vars[name] = (i * mean_log_vars[name] + value) / (i + 1)    # 实时计算并更新平均值

        log_var_str = ", ".join(f"mean {name}:{value :.3f}" for name, value in mean_log_vars.items())
        process_bar.desc = f"val epoch[{epoch + 1}/{schedule_cfg.get('epochs', 50)}] {log_var_str}"
        # val_seg_logits = dict()

        # 计算模型性能指标
        # 模型预测结果与标签值有两种:
        # 1. tensor:
        # pred: (N, C, H, W), labels: (N, H, W)
        # 2. list of tensor:
        # len(list) == N, pred[i]: (1, C, H, W), labels[i]: (H, W)
        evaluator.process(batch_idx=i, pred_batch=seg_logits, batch_infos=infos)

    metrics = evaluator.compute_metrics()
    return mean_log_vars, metrics


def pth_metadata(metadata: Optional[dict],
                 epoch: int,
                 fits: float,
                 train_log: Optional[dict] = None,
                 val_log: Optional[dict] = None,
                 metric: Optional[dict] = None):
    metadata.update(epoch=epoch,
                    fits=fits)
    train_log = add_prefix(inputs=train_log, prefix='train')
    metadata.update(train_log)
    val_log = add_prefix(inputs=val_log, prefix='val')
    metadata.update(val_log)

    for key, value in metric.items():
        sub_metric = add_prefix(value, prefix='metric.' + key)
        metadata.update(sub_metric)

    return metadata


def save_model(model,
               metadata,
               save_path: str = 'model.pth'):
    checkpoint = {'metadata': metadata,
                  'state_dict': weights_to_cpu(model.state_dict())}
    torch.save(checkpoint, save_path)
