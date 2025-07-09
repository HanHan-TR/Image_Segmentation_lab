# parse: 解析；剖析；分析
import os
import sys
import importlib

from io import StringIO
import types
import platform
import torch
import shutil
from pathlib import Path, PosixPath

from collections import abc
from typing import Any, Optional, Union, Type

from core.fileio.backend import HardDiskBackend


def list_from_file(filename,
                   prefix='',
                   offset=0,
                   max_num=0,
                   encoding='utf-8',
                   # file_client_args=None
                   ):
    """Load a text file and parse the content as a list of strings.

    Note:
        In v1.3.16 and later, ``list_from_file`` supports loading a text file
        which can be storaged in different backends and parsing the content as
        a list for strings.

    Args:
        filename (str): Filename.
        prefix (str): The prefix to be inserted to the beginning of each item.
        offset (int): The offset of lines.
        max_num (int): The maximum number of lines to be read,
            zeros and negatives mean no limitation.
        encoding (str): Encoding used to open the file. Default utf-8.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmcv.fileio.FileClient` for details.
            Default: None.

    Examples:
        >>> list_from_file('/path/of/your/file')  # disk
        ['hello', 'world']
        >>> list_from_file('s3://path/of/your/file')  # ceph or petrel
        ['hello', 'world']

    Returns:
        list[str]: A list of strings.
    """
    cnt = 0
    item_list = []
    # file_client = FileClient.infer_client(file_client_args, filename)
    with StringIO(HardDiskBackend.get_text(filename, encoding)) as f:
        for _ in range(offset):
            f.readline()
        for line in f:
            if 0 < max_num <= cnt:
                break
            item_list.append(prefix + line.rstrip('\n\r'))
            cnt += 1
    return item_list


def dict_from_file(filename,
                   key_type=str,
                   encoding='utf-8',
                   # file_client_args=None
                   ):
    """Load a text file and parse the content as a dict.

    Each line of the text file will be two or more columns split by
    whitespaces or tabs. The first column will be parsed as dict keys, and
    the following columns will be parsed as dict values.

    Note:
        In v1.3.16 and later, ``dict_from_file`` supports loading a text file
        which can be storaged in different backends and parsing the content as
        a dict.

    Args:
        filename(str): Filename.
        key_type(type): Type of the dict keys. str is user by default and
            type conversion will be performed if specified.
        encoding (str): Encoding used to open the file. Default utf-8.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmcv.fileio.FileClient` for details.
            Default: None.

    Examples:
        >>> dict_from_file('/path/of/your/file')  # disk
        {'key1': 'value1', 'key2': 'value2'}
        >>> dict_from_file('s3://path/of/your/file')  # ceph or petrel
        {'key1': 'value1', 'key2': 'value2'}

    Returns:
        dict: The parsed contents.
    """
    mapping = {}
    # file_client = FileClient.infer_client(file_client_args, filename)
    with StringIO(HardDiskBackend.get_text(filename, encoding)) as f:
        for line in f:
            items = line.rstrip('\n').split()
            assert len(items) >= 2
            key = key_type(items[0])
            val = items[1:] if len(items) > 2 else items[1]
            mapping[key] = val
    return mapping


def select_device(device='', batch_size=0, newline=True):
    # device = None or 'cpu' or 0 or '0' or '0,1,2,3'
    s = f'Python-{platform.python_version()} torch-{torch.__version__} '
    # 处理device参数。它将其转换为小写，并移除了'cuda:'和'none'。例如，'cuda:0'将被转换为'0'
    device = str(device).strip().lower().replace('cuda:', '').replace('none', '')  # to string, 'cuda:0' to '0'
    cpu = device == 'cpu'
    mps = device == 'mps'  # Apple Metal Performance Shaders (MPS)
    if cpu or mps:
        # 如果device是'cpu'或'mps'，则将环境变量CUDA_VISIBLE_DEVICES设置为'-1'，以强制torch.cuda.is_available()返回False
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device:
        # 如果device不是'cpu'或'mps'，它会将环境变量CUDA_VISIBLE_DEVICES设置为device的值，并确保CUDA设备可用。
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', '')), \
            f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

    if not cpu and not mps and torch.cuda.is_available():  # prefer GPU if available
        # 如果不是CPU或MPS，并且CUDA设备可用，它会优先选择GPU。它会检查设备数量和批量大小，确保批量大小是设备数量的倍数。
        # 然后，它遍历设备列表，获取每个设备的属性（例如名称和总内存），并将其添加到字符串s中
        devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size > 0:
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # bytes to MB
        arg = 'cuda:0'
    elif mps and getattr(torch, 'has_mps', False) and torch.backends.mps.is_available():  # prefer MPS if available
        s += 'MPS\n'
        arg = 'mps'
    else:  # revert to CPU
        s += 'CPU\n'
        arg = 'cpu'

    if not newline:
        s = s.rstrip()
    print(s)
    return torch.device(arg)


def parse_and_backup_config(filename: PosixPath, backup_dir: PosixPath = None, metadata: dict = None):  # ? OK
    (path, file) = os.path.split(filename)
    if backup_dir:
        backup_file = backup_dir / file
        shutil.copy(str(filename), str(backup_dir))

        meta_key = filename.parts[1] + '_config'
        metadata[meta_key] = str(backup_file)

    abspath = os.path.abspath(os.path.expanduser(path))
    sys.path.insert(0, abspath)
    mod = importlib.import_module(file.split('.')[0])
    sys.path.pop(0)
    cfg_dict = {
        name: value
        for name, value in mod.__dict__.items()
        if not name.startswith('__')
        and not isinstance(value, types.ModuleType)
        and not isinstance(value, types.FunctionType)
    }
    return cfg_dict


def add_prefix(inputs, prefix):
    """Add prefix for dict.

    Args:
        inputs (dict): The input dict with str keys.
        prefix (str): The prefix to add.

    Returns:

        dict: The dict with keys updated with ``prefix``.
    """

    outputs = dict()
    for name, value in inputs.items():
        outputs[f'{prefix}.{name}'] = value

    return outputs


def is_seq_of(seq: Any,
              expected_type: Union[Type, tuple],
              seq_type: Optional[Type] = None) -> bool:
    """检查它是否为某种类型的序列。

    Args:
        seq (Sequence): 要检查的序列。
        expected_type (type 或 tuple)：序列项的预期类型。
        seq_type (type, 可选)：预期的序列类型。默认为 None。

    Returns:
        bool：如果 ``seq`` 有效则返回 True，否则返回 False。

    Examples:
        >>> from mmengine.utils import is_seq_of
        >>> seq = ['a', 'b', 'c']
        >>> is_seq_of(seq, str)
        True
        >>> is_seq_of(seq, int)
        False
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def is_list_of(seq, expected_type):
    """检查它是否为某种类型的列表。

    :func:`is_seq_of` 的部分方法。
    """
    return is_seq_of(seq, expected_type, seq_type=list)


def is_tuple_of(seq, expected_type):
    """检查它是否为某种类型的元组。

    这是 :func:`is_seq_of` 的一个部分方法。
    """
    return is_seq_of(seq, expected_type, seq_type=tuple)


def str_from_dict(input: dict, hyphen: str = '\n'):
    """将字典转换成可以打印输出的字符串"""
    max_key_len = max(len(k) for k in input.keys()) + 5
    result = hyphen.join(f"{key + ':':<{max_key_len}} {value}" for key, value in input.items())
    return result
