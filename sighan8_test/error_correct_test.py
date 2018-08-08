# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
import sys
sys.path.append("../")
from pycorrector import corrector

error_sentences = [
    # '他戴着眼镜跟袜子入睡了。', # *!!!!!!!!!!
    # '看电影时候，我都觉得这个电影很有意思，可是现在我把什么事都不济的。',   ##16.095643 -> 12.051
    # '今天下了课，我打算跟我的奴朋友去看电影，所以我有一点儿领张，六点半我就起床了。',  ## 12.331646. -> 7.87076
    # '我要跟我的朋友一起去市大夜市吃晚饭。', # 12.463. ->. 16.21970
    # '小祥有女朋友。他的女朋友是同班同学，而且他们两个是邻住。' # 32.034513 --> 23.3937, 23.393771 --> 16.83796
    '我今天二十三个小时的考试，热后我应该回家到下个星期，所以我觉得我们没有办法见面了。',
    # '我听说这个礼拜六你要开一个误会。可是那天我会很忙，因为我男朋友要回国来看我。',  #20.163218 -> 20.20958738

]
for line in error_sentences:
    print("starting correction...")
    correct_sent = corrector.correct(line)
    print("original sentence:{} => correct sentence:{}".format(line, correct_sent))




