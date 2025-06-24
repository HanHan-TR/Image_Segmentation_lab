import os
import os.path as osp
from pathlib import Path, PosixPath
from typing import Union


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def is_filepath(x):
    return isinstance(x, str) or isinstance(x, Path)


def increment_path(work_dir: Union[str, PosixPath] = 'res',
                   project: str = 'project',
                   name: str = 'exp',
                   sep: str = '',
                   exist_ok: bool = False,  # 是否允许路径覆盖，默认不允许
                   mkdir: bool = False
                   ) -> PosixPath:
    """生成递增路径以防止命名冲突。

    当发生冲突时，自动创建顺序的目录/文件路径，
    例如 'work_dir/project/exp' -> 'work_dir/project/exp2', 'work_dir/project/exp3' 等。

    参数：
        work_dir (str/Path): 基础工作目录。默认为 'res'
        project (str): 项目子目录名称。默认为 'project'
        name (str): 实验名称。默认为 'exp'
        sep (str): 名称与递增数字之间的分隔符。默认为空字符串 ''
        exist_ok (bool): 允许覆盖现有路径。默认为 False
        mkdir (bool): 立即创建目录。默认为 False

    返回：
        Path: 生成的具有顺序递增的路径对象
    """
    path = Path(work_dir) / project / name  # Platform-independent path
    if path.exists() and not exist_ok:  # 当路径已存在且不允许覆盖时，进行递增处理
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        # Method 1
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path
