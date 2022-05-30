# webdataset封装接口
pytorch大规模数据读取dataset
## 说明
对于tensorflow，读取大规模数据可以使用tfrecord进行存储，而pytorch则只能通过读取小文件、或者通过pickle文件打包，IterableDataset进行读取。
在这类场景下，webdataset是一个比较好的工具。这里我将其进行了封装，可以通过继承，复用已有接口

## How to use
- 接口说明：
- Args:
    - tar_pattern: 格式path/video-tfr-{000000..000003}.tar
    - length: 数据样本长度
    - local_rank: reduce操作所在机器
    - world_size: 单台机器内GPU卡数
    - keys：tar文件内的数据的key的子集，需要用来解析的key
    - decode_key_funcs: key对应的数据decode操作，可以为编解码等相关操作
    - process_key_funcs: 每个key对应的处理操作
    - shuffle_buffer: 打乱数据对应的缓存区空间


### 例子
#### 生成tar文件
将数据写入tar，见write2tar.py中例子

#### 继承并实现解析逻辑
这里想实现一个功能：通过读取text，将word以及对应词向量存储至tar文件中；需要如下步骤：
1. 构建解析函数
2. 构建pipe_fn函数，主要用来整理输出格式

```python

class TextPairDistilTarDataset(BaseTarDataset):
    def __init__(self, filename, split, ratio, length, local_rank=0, world_size=2):
        identity_func = lambda x: x
        embedd_process_fn = lambda x: np.array([eval(item) for item in str(x, encoding='utf-8').split("\t")], dtype=np.float32)

        keys = ["text", "embedding"]
        decode_key_funcs = [identity_func, embedd_process_fn]
        process_key_funcs = [identity_func, identity_func]
        
        data_pattern = list(braceexpand.braceexpand(filename))
        indexes = list(range(len(data_pattern)))
        random.shuffle(indexes)
        
        selected_indexes = indexes[:int(ratio * len(indexes))] if split == 'train' else indexes[int(ratio * len(indexes)):]
        tar_pattern = [data_pattern[i] for i in selected_indexes]
        print("[INFO]:{} mode, curr filelist,".format(split), len(tar_pattern))
        super().__init__(
            tar_pattern=tar_pattern,
            length=length,
            local_rank=local_rank,
            world_size=world_size,
            keys=keys,
            decode_key_funcs=decode_key_funcs,
            process_key_funcs=process_key_funcs,
            shuffle_buffer=2000,
        )
        self.length = length

    def __len__(self):
        return self.length
    
    def find_chinese(self, file):
        pattern = re.compile(r'[^\u4e00-\u9fa5]')
        chinese = re.sub(pattern, '', file)
        return chinese
    
    def pipe_fn(self, data):
        for sample in data:
            sample = list(sample)
            text = str(sample[0], encoding='utf-8')
            text = self.find_chinese(text)
            if len(text) == 0:
                text = "我的"
            length = len(text)
            if len(text) > 256:
                start_index = np.random.randint(0, length - 256)
                text = text[start_index: start_index + 256]
            embedding = sample[1]
            yield {'text': text, 'embedding': embedding}

```
#### 在dataloader中使用
```
dataset_class = tar_class_dataset_init()
dataloader = torch.utils.data.DataLoader(dataset_class.dataset, batch_size=xx, num_worker=xx)
```

**TIPS: webdataset属于IterableDataset, 因此需要指定数据length，否则会报错**
