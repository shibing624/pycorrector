# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import sys
import os
import tensorflow as tf

sys.path.append('../..')

from pycorrector.transformer import config
from pycorrector.transformer.model import translate, model, checkpoint

if __name__ == "__main__":
    if config.gpu_id > -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    data_config = {
        "source_vocabulary": config.src_vocab_path,
        "target_vocabulary": config.tgt_vocab_path
    }

    model.initialize(data_config)
    # Load model
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, config.model_dir, max_to_keep=5)
    if checkpoint_manager.latest_checkpoint is not None:
        tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
        checkpoint.restore(checkpoint_manager.latest_checkpoint)

    translate(config.src_test_path,
              batch_size=config.batch_size,
              beam_size=config.beam_size)

# result:
# input:由我起开始做。
# output:我开始做。
# input:没有解决这个问题，
# output:没有解决的问题，
# input:由我起开始做。
