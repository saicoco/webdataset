import torch
import sys
import warnings
import webdataset as wds

show_splits = False


def split_by_node(urls, dist_rank, dist_size):

    """Split urls for each node.
    This uses the rank and world size. Note that it is invoked in each worker,
    so the results need to be consistent between multiple invocations."""

    if dist_rank >= 0 and dist_size > 0:
        result = urls[dist_rank::dist_size]
        if show_splits:
            print(
                f"split_by_node {dist_rank}/{dist_size} len={len(result)}",
                file=sys.stderr,
            )
        return result
    else:
        return urls


def split_by_worker(urls):

    """Split urls for each worker."""

    urls = [url for url in urls]
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        wid = worker_info.id
        num_workers = worker_info.num_workers
        if wid == 0 and len(urls) < num_workers:
            warnings.warn(f"num_workers {num_workers} > num_shards {len(urls)}")
        result = urls[wid::num_workers]
        if show_splits:
            print(
                f"split_by_worker {wid}/{num_workers} len={len(result)}",
                file=sys.stderr,
            )
        return result
    else:
        return urls


class BaseTarDataset(object):
    def __init__(
        self,
        tar_pattern,
        length=None,
        local_rank=0,
        world_size=2,
        keys=[],
        decode_key_funcs=[],
        process_key_funcs=[],
        shuffle_buffer=2000,
        **kwargs,
    ):
        """
        Args:
            tar_pattern: 格式path/video-tfr-{000000..000003}.tar
            length: 数据样本长度
            local_rank: reduce操作所在机器
            world_size: 单台机器内GPU卡数
            keys：tar文件内的数据的key的子集，需要用来解析的key
            decode_key_funcs: key对应的数据decode操作，可以为编解码等相关操作
            process_key_funcs: 每个key对应的处理操作
            shuffle_buffer: 打乱数据对应的缓存区空间
        """
        decode_list = []
        for k, kf in zip(keys, decode_key_funcs):
            decode_list.append(wds.handle_extension(".{}".format(k), kf))

        tuple_str = " ".join(keys)

        self.dataset = (
            wds.WebDataset(
                tar_pattern,
                length=length,
                nodesplitter=lambda x: split_by_node(x, local_rank, world_size),
                splitter=split_by_worker,
            )
            .shuffle(shuffle_buffer)
            .decode(*decode_list)
            .to_tuple(tuple_str)
            .map_tuple(*process_key_funcs)
            .pipe(self.pipe_fn)
        )

        self.keys = keys

    def pipe_fn(self, data):
        for sample in data:
            sample = list(sample)
            sample[0] = str(sample[0], encoding='utf-8')
            yield sample
