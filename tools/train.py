# Copyright 2021 colaYang.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import datetime
import warnings

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ProgressBar
from tqdm import tqdm

warnings.simplefilter("ignore", UserWarning)
import shutil

from package.data.collate import collate_function
from package.data.dataset import build_dataset
from package.evaluator import build_evaluator
from package.trainer.task import TrainingTask
from package.util import Logger, cfg, convert_old_model, load_config, mkdir

os.environ['MASTER_PORT'] = '12355'
os.environ['MASTER_ADDR'] = 'localhost'
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed')
    args = parser.parse_args()
    return args


def main(args):
    start = datetime.datetime.now()

    load_config(cfg, args.config)
    if cfg.model.arch.head.num_classes != len(cfg.class_names):
        raise ValueError('cfg.model.arch.head.num_classes must equal len(cfg.class_names), '
                         'but got {} and {}'.format(cfg.model.arch.head.num_classes, len(cfg.class_names)))
    local_rank = int(args.local_rank)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    mkdir(local_rank, cfg.save_dir)
    logger = Logger(local_rank, cfg.save_dir)

    if args.seed is not None:
        logger.log('Set random seed to {}'.format(args.seed))
        pl.seed_everything(args.seed)

    logger.log('Setting up data...')
    train_dataset = build_dataset(cfg.data.train, 'train')
    val_dataset = build_dataset(cfg.data.val, 'test')

    evaluator = build_evaluator(cfg, val_dataset)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.device.batchsize_per_gpu,
                                                   shuffle=True, num_workers=cfg.device.workers_per_gpu,
                                                   pin_memory=True, collate_fn=collate_function, drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.device.batchsize_per_gpu,
                                                 shuffle=False, num_workers=cfg.device.workers_per_gpu,
                                                 pin_memory=True, collate_fn=collate_function, drop_last=False)

    logger.log('Creating model...')
    task = TrainingTask(cfg, evaluator)
    
    if 'load_model' in cfg.schedule:
        ckpt = torch.load(cfg.schedule.load_model)
        if 'pytorch-lightning_version' not in ckpt:
            warnings.warn('Warning! Old .pth checkpoint is deprecated. '
                          'Convert the checkpoint with tools/convert_old_checkpoint.py ')
            ckpt = convert_old_model(ckpt)
        task.load_state_dict(ckpt['state_dict'], strict=False)
        logger.log('Loaded model weight from {}'.format(cfg.schedule.load_model))

    model_resume_path = os.path.join(cfg.save_dir, 'model_last.ckpt') if 'resume' in cfg.schedule else None
    if cfg.data.train.readall is True:
        for i in tqdm(range(len(train_dataloader))):
            dataloaderIter = iter(train_dataloader)
            next(dataloaderIter)
    
    
    trainer = pl.Trainer(default_root_dir=cfg.save_dir,
                         max_epochs=cfg.schedule.total_epochs,
                         gpus=cfg.device.gpu_ids,
                         check_val_every_n_epoch=cfg.schedule.val_intervals,
                         accelerator='dp',
                         #amp_backend='apex', amp_level=cfg.device.amp_level,
                         precision=16,
                         log_every_n_steps=cfg.log.interval,
                         num_sanity_val_steps=0,
                         resume_from_checkpoint=model_resume_path,
                         callbacks=[ProgressBar(refresh_rate=0)],  # disable tqdm bar
                         benchmark=True,
                         )
    
    trainer.fit(task, train_dataloader, val_dataloader)
    logger.log("copy config")
    listdir_info = os.listdir(os.path.join(cfg.save_dir,'lightning_logs'))
    existing_versions = []
    for listing in listdir_info:

        bn = os.path.basename(listing)
        if bn.startswith("version_"):
            dir_ver = bn.split("_")[1].replace('/', '')
            existing_versions.append(int(dir_ver))
        if len(existing_versions) == 0:
                existing_versions.append(0)
    strint = str('lightning_logs/version_'+ str(max(existing_versions)))
    shutil.copy(args.config, os.path.join(cfg.save_dir,strint))
    end = datetime.datetime.now()
    runtimee = str("執行時間：", end - start)
    logger.log(runtimee)

if __name__ == '__main__':
    args = parse_args()
    main(args)
