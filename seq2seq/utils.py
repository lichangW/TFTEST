# -*- coding:utf-8 -*-

def get_gpus(cuda_only=True):
    """
    code from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/platform/test.py
    Returns whether TensorFlow can access a GPU.
    Args:
      cuda_only: limit the search to CUDA gpus.
    Returns:
      True iff a gpu device of the requested kind is available.
    """
    from tensorflow.python.client import device_lib as _device_lib

    if cuda_only:
        return [x.name for x in _device_lib.list_local_devices() if x.device_type == 'GPU']
    else:
        return [x.name for x in _device_lib.list_local_devices() if x.device_type == 'GPU' or x.device_type == 'SYCL']


local_gpus= get_gpus()

def get_gpu_str(device_id):

    if len(local_gpus)==0:
        return "/cpu:0"

    return local_gpus[device_id %len(local_gpus)]