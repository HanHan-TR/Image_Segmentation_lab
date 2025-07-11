import argparse
import shutil
import torch
from datetime import datetime
from torch.utils.data import DataLoader
import sys
import os
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
RANK = int(os.getenv('RANK', -1))

from models import build_segmentor
from core.initialize import init_random_seed, set_random_seed
from core.builder import DATASET, LR_SCHEDULER, build_optimizer, build_from_cfg
from core.evaluation import SegEvaluator
from core.fileio import parse_and_backup_config, increment_path
from utils.train_utils import train_one_epoch, validate_one_epoch, save_model
from tools.logger.metadata import get_environment_info


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    # Step 1 ------------------- 设置配置文件 ------------------------------------------------------------------------------
    parser.add_argument('--network-cfg',
                        default=ROOT / 'configs/network/deeplabv3/deeplabv3_r50-d8.py', help='network config file path')
    parser.add_argument('--dataset-cfg',
                        default=ROOT / 'configs/dataset/KvasirSEG.py', help='dataset config file path')
    parser.add_argument('--schedule-cfg',
                        default=ROOT / 'configs/schedule/kvasir_training_schedule.py', help='dataset config file path')
    # Step 2 ------------------- 设置训练结果保存路径 ------------------------------------------------------------------------------
    parser.add_argument('--work-dir',
                        default=ROOT / 'res/train', help='the dir to save logs and models')
    parser.add_argument('--project', type=str,
                        default='CarinaShifting', help='save to work-dir/project')
    parser.add_argument('--name', default='exp', help='save to work-dir/project/name')
    parser.add_argument('--deterministic', action='store_true',
                        default=False,
                        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--device', default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--load-from',
                        help='the checkpoint file to load weights from')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--no-validate',
                        action='store_true',
                        help='whether not to evaluate the checkpoint during training')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--auto-resume',
                        action='store_true',
                        help='resume from the latest checkpoint automatically.')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    # Step 1 ---------------------------------------------- 设置输出保存文件夹 ----------------------------------------------
    args = parse_args()
    save_dir = increment_path(work_dir=args.work_dir, project=args.project, name=args.name)
    weights_dir, config_dir = Path(save_dir) / 'weights', Path(save_dir) / 'config'
    weights_dir.mkdir(parents=True, exist_ok=True)  # make dir
    config_dir.mkdir(parents=True, exist_ok=True)
    last_pth, best_pth = weights_dir / 'last.pth', weights_dir / 'best.pth'
    # last_pt, best_pt = weights_dir / 'last.pt', weights_dir / 'best.pt'
    # Step 2 ---------------------------------------------- 读取配置文件与命令行参数 -----------------------------------------
    metadata = dict()
    metadata.update(time=datetime.now().strftime(format="%Y-%m-%d %H:%M:%S"))
    metadata.update(get_environment_info())
    network_cfg = parse_and_backup_config(filename=args.network_cfg,
                                          backup_dir=config_dir,
                                          metadata=metadata).pop('model')
    dataset_cfg = parse_and_backup_config(filename=args.dataset_cfg,
                                          backup_dir=config_dir,
                                          metadata=metadata).pop('dataset')
    schedule_cfg = parse_and_backup_config(filename=args.schedule_cfg,
                                           backup_dir=config_dir,
                                           metadata=metadata)
    # Step 3 ---------------------------------------------- 设备选择，设置随机种子----------------------------------------------
    seed = init_random_seed(seed=schedule_cfg.get('seed', 0), device=args.device)
    metadata.update(seed=seed)
    set_random_seed(seed=seed, deterministic=schedule_cfg.get('deterministic', False))
    # Step 4 搭建语义分割网络模型 （模型权重初始化）
    model = build_segmentor(cfg=network_cfg)
    print(model)
    model = model.to('cuda')
    # 设置优化区与学习率调度器 这部分设置的一些配置需要具体手动修改代码
    optimizer = build_optimizer(cfg=schedule_cfg.pop('optimizer'),
                                params=model.parameters())
    lr_scheduler = build_from_cfg(cfg=schedule_cfg.pop('lr_config'),
                                  registry=LR_SCHEDULER,
                                  default_args=dict(optimizer=optimizer))
    # Step 5 读取数据集 并设置数据加载器
    train_dataset = build_from_cfg(cfg=dataset_cfg['train'], registry=DATASET)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=schedule_cfg.get('train_batch_size', 8),
                                  shuffle=False,
                                  num_workers=schedule_cfg.get('num_workers', 4),
                                  pin_memory=False,
                                  collate_fn=train_dataset.collate_fn)

    val_dataset = build_from_cfg(cfg=dataset_cfg['val'], registry=DATASET)
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=schedule_cfg.get('val_batch_size', 4),
                                shuffle=False,
                                num_workers=schedule_cfg.get('num_workers', 4),
                                pin_memory=False,
                                collate_fn=val_dataset.collate_fn)

    metadata.update(CLASSES=train_dataset.CLASSES,
                    PALETTE=train_dataset.PALETTE)
    scaler = torch.cuda.amp.GradScaler(enabled=schedule_cfg.get('amp', False))  # 混合精度训练

    for epoch in range(schedule_cfg.get('epochs', 50)):
        optimizer.zero_grad()
        train_log_vars = train_one_epoch(epoch, model, train_dataloader, optimizer, scaler, schedule_cfg)
        # # 更新学习率
        lr_scheduler.step()
        # # 保存模型

        # todo：
        # 1. 模型验证相关代码：验证的推理过程，模型评价指标计算；
        evaluator = SegEvaluator(epoch,
                                 num_classes=val_dataset.num_classes,
                                 class_names=val_dataset.CLASSES,
                                 palette=val_dataset.PALETTE,
                                 ignore_index=val_dataset.ignore_index,
                                 output_dir=save_dir)
        # 2. 保存模型相关代码：保存模型权重，日志文件，配置文件等；
        # 3. 日志记录相关代码：记录训练过程中的相关信息，如loss，lr，时间等；
        # 4. 模型推理结果的可视化相关代码：可视化模型推理结果，如预测的mask，原始图像，预测结果等；
        # 5. 训练过程的可视化相关代码：可视化训练过程，如loss曲线，lr变化曲线等；
        # 6. CAM可视化相关代码：可视化CAM结果，如原始图像，CAM结果等；
        val_log_vars, metrics = validate_one_epoch(epoch=epoch,
                                                   model=model,
                                                   data_loader=val_dataloader,
                                                   evaluator=evaluator,
                                                   schedule_cfg=schedule_cfg)

        # 保存模型，保存成pth
        save_model(model,
                   metadata=metadata,
                   train_log=train_log_vars,
                   val_log=val_log_vars,
                   metric=metrics,
                   with_aux=False,
                   save_path=last_pth)
        print('Done')


if __name__ == '__main__':
    main()
