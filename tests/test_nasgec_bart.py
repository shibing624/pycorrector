import sys
import unittest

sys.path.append('..')
from pycorrector import NaSGECBartCorrector
from pycorrector.utils.sentence_utils import is_not_chinese_error


m = NaSGECBartCorrector()


class MyTestCase(unittest.TestCase):
    def test1(self):
        sents = ["北京是中国的都。", "他说：”我最爱的运动是打蓝球“", "我每天大约喝5次水左右。", "今天，我非常开开心。"]
        res = m.correct_batch(sents)

        self.assertEqual(res[0]['target'], '北京是中国的首都。')
        self.assertEqual(res[1]['target'], '他说：“我最爱的运动是打篮球”')
        self.assertEqual(res[2]['target'], '我每天大约喝5次水左右。')
        self.assertEqual(res[3]['target'], '今天，我非常开心。')

    
    def test2(self):
        long_text = "在一个充满生活热闹和忙碌的城市中，有一个年轻人名叫李华。他生活在北京，这座充满着现代化建筑和繁忙街道的都市。每天，他都要穿行在拥挤的人群中，追逐着自己的梦想和生活节奏。\n\n李华从小就听祖辈讲述关于福气和努力的故事。他相信，“这洋的话，下一年的福气来到自己身上”。因此，尽管每天都很忙碌，他总是尽力保持乐观和积极。\n\n某天早晨，李华骑着自行车准备去上班。北京的交通总是非常繁忙，尤其是在早高峰时段。他经过一个交通路口，看到至少两个交警正在维持交通秩序。这些交警穿着整齐的制服，手势有序而又果断，让整个路口的车辆有条不紊地行驶着。这让李华想起了他父亲曾经告诫过他的话：“在拥挤的时间里，为了让人们遵守交通规则，至少要派两个警察或者交通管理者。”\n\n李华心中感慨万千，他想要在自己的生活中也如此积极地影响他人。他虽然只是一名普通的白领，却希望能够通过自己的努力和行动，为这座城市的安全与和谐贡献一份力量。\n\n随着时间的推移，中国的经济不断发展，北京的建设也日益繁荣。李华所在的公司也因为他的努力和创新精神而蓬勃发展。他喜欢打篮球，每周都会和朋友们一起去运动场，放松身心。他也十分重视健康，每天都保持适量的饮水量，大约喝五次左右。\n\n今天，李华觉得格外开心。他意识到，自己虽然只是一个普通人，却通过日复一日的努力，终于在生活中找到了属于自己的那份福气。他明白了祖辈们口中的那句话的含义——“这洋的话，下一年的福气来到自己身上”，并且深信不疑。\n\n在这个充满希望和机遇的时代里，李华将继续努力工作，为自己的梦想而奋斗，也希望能够在这座城市中留下自己的一份足迹，为他人带来更多的希望和正能量。\n\n这就是李华的故事，一个在现代城市中追寻梦想和福气的普通青年。"
        result = m.correct(long_text, ignore_function=is_not_chinese_error)
        for e in result["errors"]:
            self.assertEqual(result["source"][e[2]], e[0])


if __name__ == '__main__':
    unittest.main()