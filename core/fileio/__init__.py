# Copyright (c) OpenMMLab. All rights reserved.
# from .file_client import BaseStorageBackend, FileClient
from .handlers import BaseFileHandler, JsonHandler, PickleHandler, YamlHandler
from .backend import HardDiskBackend
from .parse import dict_from_file, list_from_file, is_list_of, is_seq_of, is_tuple_of, file2dict
from .path import increment_path, is_filepath, mkdir_or_exist
from .io import dump, load, register_handler  # , HardDiskBackend


__all__ = [
    'BaseFileHandler', 'JsonHandler', 'PickleHandler', 'YamlHandler',
    'HardDiskBackend',
    'dump', 'load', 'register_handler',
    'dict_from_file', 'list_from_file', 'is_list_of', 'is_seq_of', 'is_tuple_of', 'file2dict',
    'increment_path', 'is_filepath', 'mkdir_or_exist'
]
