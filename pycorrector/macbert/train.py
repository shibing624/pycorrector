# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com), Abtion(abtion@outlook.com)
@description: 
"""
import os
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import BertTokenizer

sys.path.append('../..')

from pycorrector.macbert.reader import make_loaders, get_csc_loader, DataCollator
from pycorrector.macbert.macbert4csc import MacBert4Csc
from pycorrector.macbert import preprocess, config
from pycorrector.utils.logger import logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, loaders, ckpt_callback=None):
    """
    训练
    Args:
        model: 模型
        loaders: 各个数据的loader，包含train，valid，test
        ckpt_callback: 按需保存模型的callback，如为空则默认每个epoch保存一次模型。
    Returns:
        None
    """
    train_loader, valid_loader, test_loader = loaders
    trainer = pl.Trainer(max_epochs=config.epochs,
                         gpus=None if device == torch.device('cpu') else config.gpu_ids,
                         accumulate_grad_batches=4,
                         callbacks=[ckpt_callback])
    # 进行训练
    # train_loader中有数据
    if train_loader and len(train_loader) > 0:
        if valid_loader and len(valid_loader) > 0:
            trainer.fit(model, train_loader, valid_loader)
        else:
            trainer.fit(model, train_loader)
    logger.info('train model done.')
    # 进行测试的逻辑同训练
    if test_loader and len(test_loader) > 0:
        if ckpt_callback and len(ckpt_callback.best_model_path) > 0:
            ckpt_path = ckpt_callback.best_model_path
        elif config.ckpt_path:
            ckpt_path = config.ckpt_path
        else:
            ckpt_path = None
        logger.info(f'ckpt_path: {ckpt_path}')
        if ckpt_path and os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path)['state_dict'])
        trainer.test(model, test_loader)


def main():
    # 如果不存在训练文件则先处理数据
    if not os.path.exists(config.train_path):
        logger.debug('preprocess data')
        preprocess.main()
    logger.info('load model')
    tokenizer = BertTokenizer.from_pretrained(config.pretrained_model)
    collator = DataCollator(tokenizer=tokenizer)
    model = MacBert4Csc(tokenizer, pretrained_model=config.pretrained_model)

    # 热启动
    if config.ckpt_path and os.path.exists(config.ckpt_path):
        model.load_from_checkpoint(checkpoint_path=config.ckpt_path, map_location=device, tokenizer=tokenizer)
    # 加载数据
    loaders = make_loaders(get_csc_loader, train_path=config.train_path, valid_path='', test_path=config.test_path,
                           batch_size=config.batch_size, test_batch_size=config.test_batch_size, num_workers=4,
                           _collate_fn=collator)
    ckpt_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=config.model_dir,
        filename='{epoch:02d}-{val_loss:.5f}',
        save_top_k=1,
        mode='min'
    )
    # 训练模型
    logger.info('train model ...')
    train(model, loaders, ckpt_callback)


if __name__ == '__main__':
    main()
