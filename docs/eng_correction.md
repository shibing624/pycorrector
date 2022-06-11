# 英文纠错

### 错误样本生成器

```python
import nlpaug.augmenter.char as nac
def ocr_augment_chars(text, **kwargs):
    aug = nac.OcrAug(**kwargs)
    augmented_data = aug.augment(text)
    return augmented_data

text_list = ['i am s student.', 'i am a teacher.']
ocr_text = ocr_augment_chars(text_list, aug_char_p=0.4, aug_word_p=0.6)
```

