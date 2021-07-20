# coding=utf-8
import tensorflow as tf
import numpy as np
import base64
from absl import logging
import webdataset as wds


if __name__ == "__main__":
    import sys, glob, os

    record_filename = sys.argv[1]
    output_dir = sys.argv[2]
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
        
    index = 0
    num_per_dir = 2000
    
    pattern = os.path.join(output_dir, f"eng_zh-%06d.tar")
    sink = wds.ShardWriter(pattern, maxsize=int(6e9), maxcount=int(num_per_dir))
    
    all_lines = []
    with open(record_filename, 'r') as f:
        for line in f:
            items = line.strip('\n').split('\t')
            text = items[0]
            embedding = "\t".join(items[1:])
            xkey = "%07d" % index
            sample = {
                "__key__": xkey, 
                "text": text,
                "embedding": embedding
            }
            # Write the sample to the sharded tar archives.
            sink.write(sample)
            index += 1
            
        sink.close()
        
