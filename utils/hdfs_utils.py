# coding: utf-8
"""
Support Hadoop fs commands
"""
import os
import shlex
import subprocess
import sys
import logging

if sys.version_info[0] >= 3:
    _STRING_TYPES = (str,)
else:
    _STRING_TYPES = (str, unicode)

# supported HDFS FS prefix
_SUPPORTED_HDFS_PATH_PREFIXES = ('hdfs://', 'ufs://')


def has_hdfs_path_prefix(filepath):
    """
    Check if a filepath has hdfs path prefix.
    :param filepath: str, filepath.
    :return: bool, if a filepath has hdfs path prefix.
    """
    for prefix in _SUPPORTED_HDFS_PATH_PREFIXES:
        if filepath.startswith(prefix):
            return True
    return False


def is_hdfs_path_pattern(filepath):
    """
    Check if a filepath is a hdfs path pattern.
    :param filepath: str, filepath.
    :return: bool, if a filepath is of hdfs path pattern.
    """
    return filepath.find('*') != -1 or (filepath.find('[') != -1 and filepath.find(']') != -1)


def exists_or_islink(filepath):
    """
    Check file exists or not, allow symbol links.
    :param filepath: str, filepath.
    :return: bool, if a file exists or is a symbol link.
    """
    return os.path.exists(filepath) or os.path.islink(filepath)


def check_call_hdfs_command(command, hadoop_binary='hdfs'):
    """
    Check call hdfs command with hadoop_binary.
    :param command: str, hadoop command.
    :param hadoop_binary: str, hadoop binary.
    :return None
    """
    hdfs_command = '{} dfs {}'.format(hadoop_binary, command)
    subprocess.check_call(shlex.split(hdfs_command))


def popen_hdfs_command(command, hadoop_binary='hdfs'):
    """
    Call hdfs command with popen and get stdout result.
    :param command: str, hadoop command.
    :param hadoop_binary: str, hadoop binary.
    :return stdout result.
    """
    hdfs_command = '{} dfs {}'.format(hadoop_binary, command)
    p = subprocess.Popen(shlex.split(hdfs_command), stdout=subprocess.PIPE)
    stdout, _ = p.communicate()
    return stdout


def is_hdfs_file(filepath, hadoop_binary='hdfs'):
    """
    Check if input filepath is hdfs file.
    :param filepath: str, filepath.
    :param hadoop_binary: str, hadoop binary.
    :return bool, if input filepath is hdfs file.
    """
    if exists_or_islink(filepath):
        # is local path, return False
        return False
    elif is_hdfs_path_pattern(filepath):
        # this is a hdfs path pattern, not a file
        return False
    cmd = '-test -f {}'.format(filepath)
    try:
        check_call_hdfs_command(cmd, hadoop_binary=hadoop_binary)
        return True
    except Exception:
        return False


def is_hdfs_dir(filepath, hadoop_binary='hdfs'):
    """
    Check if input filepath is hdfs directory.
    :param filepath: str, filepath.
    :param hadoop_binary: str, hadoop binary.
    :return bool, if input filepath is hdfs directory.
    """
    if exists_or_islink(filepath):
        # is local path, return False
        return False
    elif is_hdfs_path_pattern(filepath):
        # this is a hdfs path pattern, not a directory
        return False
    cmd = '-test -d {}'.format(filepath)
    try:
        check_call_hdfs_command(cmd, hadoop_binary=hadoop_binary)
        return True
    except Exception:
        return False


def hdfs_mkdir(filepath, hadoop_binary='hdfs'):
    """
    Make hdfs directory.
    :param filepath: str, filepath.
    :param hadoop_binary: str, hadoop binary.
    :return if making directory successes.
    """
    try:
        cmd = '-mkdir -p {}'.format(filepath)
        check_call_hdfs_command(cmd, hadoop_binary=hadoop_binary)
        return True
    except Exception:
        return False

def hdfs_mkdirs(filepaths, hadoop_binary='hdfs'):
    """
    Make hdfs directories.
    :param filepaths: str, filepaths.
    :param hadoop_binary: str, hadoop binary.
    :return if making directory successes.
    """
    try:
        cmd = '-mkdir -p '
        for file in filepaths:
            cmd+='{}'.format(file)
        check_call_hdfs_command(cmd, hadoop_binary=hadoop_binary)
        return True
    except Exception:
        return False
        



def hdfs_put(src, dst, overwrite=True, output_to_dir=False, hadoop_binary='hdfs'):
    """
    Upload src files/directories to dst path.
    :param src: (str, List(str), Tuple(str)), source of downloading.
    :param dst: str, destination of uploading.
    :param overwrite: bool, whether overwrite exist files or not.
    :param output_to_dir: bool, if dst is a dir.
        will be set True if src is a (list, tuple) with more than one element.
    :param hadoop_binary: str, hadoop binary.
    :return: bool, whether uploading successes.
    """
    require_dst_dir = True if output_to_dir else False
    # TODO: change assertion to type_check() after Yanyu merges.
    assert isinstance(src, (list, tuple) + _STRING_TYPES), \
        "Input src path must be a str or a list of str, got {}".format(
            type(src))
    assert src, "Input src path is empty"
    if isinstance(src, (list, tuple)):
        if len(src) > 1:
            # dst path must be a directory
            require_dst_dir = True
    else:
        src = [src]

    # check output dst path
    if not has_hdfs_path_prefix(dst):
        raise ValueError('Input dst path is not a hdfs path: {}'.format(dst))
    if require_dst_dir:
        if is_hdfs_file(dst):
            raise IOError('Required dst path {} as a directory for uploading, got a file'.format(
                dst))
        elif not is_hdfs_dir(dst):
            # mkdir
            hdfs_mkdir(dst)
    else:
        dst_dir = os.path.dirname(dst)
        if not is_hdfs_dir(dst_dir):
            # mkdir
            if is_hdfs_path_pattern(dst_dir):
                # this is a hdfs path pattern, cannot make this directory
                raise OSError("HDFS destination directory cannot be a wildcard pattern, got {}".format(
                    dst_dir
                ))
            hdfs_mkdir(dst_dir)

    hdfs_cmd = '-put -f' if overwrite else '-put'

    cmd = '{} {} {}'.format(
        hdfs_cmd, ' '.join(src), dst)
    try:
        check_call_hdfs_command(cmd, hadoop_binary=hadoop_binary)
        return True
    except Exception:
        logging.error('HDFS put command exception caught: type {}, msg: {}'.format(
            type(Exception).__name__, str(Exception)
        ))
        return False




def hdfs_get(src, dst, output_to_dir=False, hadoop_binary='hdfs'):
    """
    Download src files/directories to dst path.
    :param src: (str, List(str), Tuple(str)), source of downloading.
    :param dst: str, destination of downloading.
    :param output_to_dir: bool, if dst is a dir.
        will be set True if src is a (list, tuple) with more than one element.
    :param hadoop_binary: str, hadoop binary.
    :return: bool, whether downloading successes.
    """
    require_dst_dir = True if output_to_dir else False
    # TODO: change assertion to type_check() after Yanyu merges.
    assert isinstance(src, (list, tuple) + _STRING_TYPES), \
        "Input src path must be a str or a list of str, got {}".format(
            type(src))
    assert src, "Input src path is empty"
    if isinstance(src, (list, tuple)):
        if len(src) > 1:
            # dst path must be a directory
            require_dst_dir = True
    else:
        src = [src]

    # check output dst path
    if require_dst_dir:
        if os.path.exists(dst) and os.path.isfile(dst):
            raise IOError('Required dst path {} as a directory for multiple hdfs paths downloading, got a file'.format(
                dst))
        elif not os.path.exists(dst):
            os.makedirs(dst)
    else:
        dst_dir = os.path.dirname(dst)
        if dst_dir and not os.path.isdir(dst_dir):
            os.makedirs(dst_dir)

    hdfs_cmd = '-get {} {}'.format(
        ' '.join(src), dst)
    try:
        check_call_hdfs_command(hdfs_cmd, hadoop_binary=hadoop_binary)
        return True
    except Exception:
        logging.error('HDFS get command exception caught: type {}, msg: {}'.format(
            type(Exception).__name__, str(Exception)
        ))
        return False


def hdfs_ls(filepath, hadoop_binary='hdfs'):
    """
    List hdfs path pattern.
    :param filepath: str, filepath.
    :param hadoop_binary: str, hadoop binary.
    :return List(str), the result of listing.
    """
    try:
        cmd = '-ls {}'.format(filepath)
        stdout = popen_hdfs_command(cmd, hadoop_binary=hadoop_binary)
        lines = stdout.splitlines()
        if lines:
            # decode bytes string in python3 runtime
            lines = [line.decode('utf-8') for line in lines]
            return [line.split(' ')[-1] for line in lines][1:]
        else:
            return []
    except Exception:
        return []








def hdfs_rm(filepath, recursive=True, force=True, hadoop_binary='hadoop'):
    """
    Remove files from hdfs filepath.
    :param filepath: str, filepath.
    :param recursive: bool, whether do recursive removal.
    :param force: bool, whether do force removal.
    :param hadoop_binary: str, hadoop binary.
    :return: bool, whether removing successes.
    """
    # TODO: change assertion to type_check() after Yanyu merges.
    assert isinstance(filepath, _STRING_TYPES), \
        "Input filepath must be a str, got {}".format(
            type(filepath))
    hdfs_cmd = '-rm '
    if recursive:
        hdfs_cmd += '-r '
    if force:
        hdfs_cmd += '-f '
    cmd = '{}{}'.format(
        hdfs_cmd, filepath)
    try:
        check_call_hdfs_command(cmd, hadoop_binary=hadoop_binary)
        return True
    except Exception:
        logging.error('HDFS put command exception caught: type {}, msg: {}'.format(
            type(Exception).__name__, str(Exception)
        ))
        return False


def test_hdfs_curd():
    """HDFS CURD tester."""

    # You may not have right of access to the paths below.
    # You can change the _HDFS_TEST_ADDR to your own path.
    _HDFS_TEST_ADDR = "hdfs://haruna/home/byte_arnold_lq_vc/user/yujun.seu/Datasets/xigua_long_video_summarization_data"
    
    content = hdfs_ls(_HDFS_TEST_ADDR)
    print(content)
    # _HDFS_FILE_NAME = "put"
    # _HDFS_PUT_FILE_LOCAL_PATH = os.path.realpath(_HDFS_FILE_NAME)
    # _HDFS_PUT_FILE_HDFS_PATH = os.path.join(_HDFS_TEST_ADDR, _HDFS_FILE_NAME)
    # _HDFS_FILE_NAME2 = "put2"
    # _HDFS_PUT_FILE_LOCAL_PATH2 = os.path.realpath(_HDFS_FILE_NAME2)
    # _HDFS_PUT_FILE_HDFS_PATH2 = os.path.join(_HDFS_TEST_ADDR, _HDFS_FILE_NAME2)
    # _HDFS_GET_FILE_LOCAL_DIR = os.path.join(os.path.dirname(_HDFS_PUT_FILE_HDFS_PATH), 'get/')
    # _HDFS_GET_FILE_LOCAL_PATH = os.path.join(_HDFS_GET_FILE_LOCAL_DIR, _HDFS_FILE_NAME)
    # _HDFS_GET_FILE_LOCAL_PATH2 = os.path.join(_HDFS_GET_FILE_LOCAL_DIR, _HDFS_FILE_NAME2)
    # _HDFS_GET_SOURCE = [_HDFS_PUT_FILE_HDFS_PATH,
    #                     _HDFS_PUT_FILE_HDFS_PATH2]
    # _HDFS_TEST_FILE_STRING = "Hello HDFS!"
    # _HDFS_TEST_FILE_STRING2 = "Hello HDFS! Again!"


    # if not is_hdfs_dir(_HDFS_TEST_ADDR):
    #     hdfs_mkdir(_HDFS_TEST_ADDR)



    # # test hdfs put
    # with open(_HDFS_PUT_FILE_LOCAL_PATH, 'w') as f:
    #     f.write(_HDFS_TEST_FILE_STRING)
    #     print("HDFS test file string is: {}".format(
    #         _HDFS_TEST_FILE_STRING
    #     ))
    # hdfs_put(_HDFS_PUT_FILE_LOCAL_PATH, _HDFS_TEST_ADDR, output_to_dir=True)
    # print ("HDFS put: put {} to {}".format(
    #     _HDFS_PUT_FILE_LOCAL_PATH, _HDFS_TEST_ADDR
    # ))

    # with open(_HDFS_PUT_FILE_LOCAL_PATH2, 'w') as f:
    #     f.write(_HDFS_TEST_FILE_STRING2)
    #     print("HDFS test file string is: {}".format(
    #         _HDFS_TEST_FILE_STRING2
    #     ))
    # hdfs_put(_HDFS_PUT_FILE_LOCAL_PATH2, _HDFS_TEST_ADDR, output_to_dir=True)
    # print ("HDFS put: put {} to {}".format(
    #     _HDFS_PUT_FILE_LOCAL_PATH2, _HDFS_TEST_ADDR
    # ))


    # # test hdfs rm
    # hdfs_rm(_HDFS_PUT_FILE_HDFS_PATH)
    # hdfs_rm(_HDFS_PUT_FILE_HDFS_PATH2)


    # # clean up local test files
    # os.remove(_HDFS_PUT_FILE_LOCAL_PATH)
    # os.remove(_HDFS_PUT_FILE_LOCAL_PATH2)
    # os.remove(_HDFS_GET_FILE_LOCAL_PATH)
    # os.remove(_HDFS_GET_FILE_LOCAL_PATH2)
    # os.rmdir(_HDFS_GET_FILE_LOCAL_DIR)


if __name__ == '__main__':
    test_hdfs_curd()
