# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('../..')

from pycorrector.mucgec_bart.mucgec_bart_corrector import MuCGECBartCorrector

if __name__ == "__main__":
    m = MuCGECBartCorrector()
    result = m.correct_batch(['这洋的话，下一年的福气来到自己身上。',
                              '在拥挤时间，为了让人们尊守交通规律，派至少两个警察或者交通管理者。',
                              '随着中国经济突飞猛近，建造工业与日俱增',
                              "北京是中国的都。",
                              "他说：”我最爱的运动是打蓝球“",
                              "我每天大约喝5次水左右。",
                              "今天，我非常开开心。"])
    print(result)

# [{'source': '这洋的话，下一年的福气来到自己身上。', 'target': '这样的话，下一年的福气就会来到自己身上。', 'errors': [('洋', '样', 1), ('', '就会', 11)]},
# {'source': '在拥挤时间，为了让人们尊守交通规律，派至少两个警察或者交通管理者。', 'target': '在拥挤时间，为了让人们遵守交通规则，应该派至少两个警察或者交通管理者。', 'errors': [('尊', '遵', 11), ('律', '则', 16), ('', '应该', 18)]},
# {'source': '随着中国经济突飞猛近，建造工业与日俱增', 'target': '随着中国经济突飞猛进，建造工业与日俱增', 'errors': [('近', '进', 9)]},
# {'source': '北京是中国的都。', 'target': '北京是中国的首都。', 'errors': [('', '首', 6)]},
# {'source': '他说：”我最爱的运动是打蓝球“', 'target': '他说：“我最爱的运动是打篮球”', 'errors': [('”', '“', 3), ('蓝', '篮', 12), ('“', '”', 14)]},
# {'source': '我每天大约喝5次水左右。', 'target': '我每天大约喝5杯水左右。', 'errors': [('次', '杯', 7)]},
# {'source': '今天，我非常开开心。', 'target': '今天，我非常开心。', 'errors': [('开', '', 7)]}]
