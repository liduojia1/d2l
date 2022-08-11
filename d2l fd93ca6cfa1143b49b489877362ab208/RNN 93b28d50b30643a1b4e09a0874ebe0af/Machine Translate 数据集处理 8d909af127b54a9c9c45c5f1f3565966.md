# Machine Translate 数据集处理

## 预处理数据集

- DownLoad

数据集中的每一行都是制表符分隔的文本序列对， 序列对由英文文本序列和翻译后的法语文本序列组成。 请注意，每个文本序列可以是一个句子， 也可以是包含多个句子的一个段落。 在这个将英语翻译成法语的机器翻译问题中， 英语是*源语言*（source language）， 法语是*目标语言*（target language）。

```python
#@save
d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')

#@save
def read_data_nmt():
    """载入“英语－法语”数据集"""
    data_dir = d2l.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r',
             encoding='utf-8') as f:
        return f.read()

raw_text = read_data_nmt()
print(raw_text[:75])
```

![Untitled](Machine%20Translate%20%E6%95%B0%E6%8D%AE%E9%9B%86%E5%A4%84%E7%90%86%208d909af127b54a9c9c45c5f1f3565966/Untitled.png)

- 预处理

用空格代替*不间断空格*（non-breaking space）， 使用小写字母替换大写字母，并在单词和标点符号之间插入空格。

```python
#@save
def preprocess_nmt(text):
    """预处理“英语－法语”数据集"""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 使用空格替换不间断空格
    # 使用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)

text = preprocess_nmt(raw_text)
print(text[:80])
```

![Untitled](Machine%20Translate%20%E6%95%B0%E6%8D%AE%E9%9B%86%E5%A4%84%E7%90%86%208d909af127b54a9c9c45c5f1f3565966/Untitled%201.png)

## 词元化

下面的`tokenize_nmt`函数对前`num_examples`个文本序列对进行词元化，其中每个词元要么是一个词，要么是一个标点符号。 此函数返回两个词元列表：`source`和`target`； `source[i]`是源语言（这里是英语）第i个文本序列的词元列表， `target[i]`是目标语言（这里是法语）第i个文本序列的词元列表。

```python
#@save
def tokenize_nmt(text, num_examples=None):
    """词元化“英语－法语”数据数据集"""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target

source, target = tokenize_nmt(text)
source[:10], target[:10]
```

![Untitled](Machine%20Translate%20%E6%95%B0%E6%8D%AE%E9%9B%86%E5%A4%84%E7%90%86%208d909af127b54a9c9c45c5f1f3565966/Untitled%202.png)

绘制词元数量的直方图

```python
def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """绘制列表长度对的直方图"""
    d2l.set_figsize()
    _, _, patches = d2l.plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]])
    d2l.plt.xlabel(xlabel)
    d2l.plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    d2l.plt.legend(legend)

show_list_len_pair_hist(['source', 'target'], '# tokens per sequence',
                        'count', source, target);
```

![Untitled](Machine%20Translate%20%E6%95%B0%E6%8D%AE%E9%9B%86%E5%A4%84%E7%90%86%208d909af127b54a9c9c45c5f1f3565966/Untitled%203.png)

## 构建词表

由于机器翻译数据集由语言对组成， 因此我们可以分别为源语言和目标语言构建两个词表。 使用单词级词元化时，词表大小将明显大于使用字符级词元化时的词表大小。 为了缓解这一问题，这里我们将出现次数少于2次的低频率词元 视为相同的未知（“<unk>”）词元。 除此之外，我们还指定了额外的特定词元， 例如在小批量时用于将序列填充到相同长度的填充词元（“<pad>”）， 以及序列的开始词元（“<bos>”）和结束词元（“<eos>”）。 这些特殊词元在自然语言处理任务中比较常用。

```python
src_vocab = d2l.Vocab(source, min_freq=2,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
len(src_vocab)    #结果大小为10012
```

## 加载数据集

- 截断和填充

语言模型中的序列样本都有一个固定的长度， 无论这个样本是一个句子的一部分还是跨越了多个句子的一个片断。 这个固定长度是由 `num_steps`（时间步数或词元数量）参数指定的。 在机器翻译中，每个样本都是由源和目标组成的文本序列对， 其中的每个文本序列可能具有不同的长度。

为了提高计算效率，我们仍然可以通过*截断*（truncation）和 *填充*（padding）方式实现一次只处理一个小批量的文本序列。 假设同一个小批量中的每个序列都应该具有相同的长度`num_steps`， 那么如果文本序列的词元数目少于`num_steps`时， 我们将继续在其末尾添加特定的“<pad>”词元， 直到其长度达到`num_steps`； 反之，我们将截断文本序列时，只取其前`num_steps` 个词元， 并且丢弃剩余的词元。这样，每个文本序列将具有相同的长度， 以便以相同形状的小批量进行加载。

```python
#@save
def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充

truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>'])
```

其中

![Untitled](Machine%20Translate%20%E6%95%B0%E6%8D%AE%E9%9B%86%E5%A4%84%E7%90%86%208d909af127b54a9c9c45c5f1f3565966/Untitled%204.png)

上面函数的结果为`[47, 4, 1, 1, 1, 1, 1, 1, 1, 1]`

如果把<pad>换成<bos>或<eos>，则结果为`[47, 4, 2, 2, 2, 2, 2, 2, 2, 2]`，`[47, 4, 3, 3, 3, 3, 3, 3, 3, 3]`

- 将文本序列转换为小批量数据

我们将特定的“<eos>”词元添加到所有序列的末尾， 用于表示序列的结束。 当模型通过一个词元接一个词元地生成序列进行预测时， 生成的“<eos>”词元说明完成了序列输出工作。 此外，我们还记录了每个文本序列的长度（有效长度）， 统计长度时排除了填充词元。

```python
#@save
def build_array_nmt(lines, vocab, num_steps):
    """将机器翻译的文本序列转换成小批量"""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len
```

## 加载返回数据迭代器

```python
#@save
def load_data_nmt(batch_size, num_steps, num_examples=600):
    """返回翻译数据集的迭代器和词表"""
    #读取数据集 大小写空格的预处理操作
    text = preprocess_nmt(read_data_nmt())
    #词元化对：前num_examples个文本序列对进行词元化
    source, target = tokenize_nmt(text, num_examples)
    #构建词表 频率少于2的为unk未知词元
    src_vocab = d2l.Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = d2l.Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    #将文本数据转为小批量数据
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    #加载为数据迭代器
    data_iter = d2l.load_array(data_arrays, batch_size)
    #返回整体(source,target)的数据迭代器，以及(source,target)的词表
    return data_iter, src_vocab, tgt_vocab
```

调用

```python
train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
for X, X_valid_len, Y, Y_valid_len in train_iter:
    print('X:', X.type(torch.int32))
    print('X的有效长度:', X_valid_len)
    print('Y:', Y.type(torch.int32))
    print('Y的有效长度:', Y_valid_len)
    break
```

![Untitled](Machine%20Translate%20%E6%95%B0%E6%8D%AE%E9%9B%86%E5%A4%84%E7%90%86%208d909af127b54a9c9c45c5f1f3565966/Untitled%205.png)