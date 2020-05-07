'''
something-something dataset
https://20bn.com/datasets/something-something/v2#download
'''

import hashlib
from os import listdir
from os.path import isfile, join

VERIFIED_FOLDER_JESTER = 'C:/DOC/Vid compress/Dataset/Jester'
VERIFIED_FOLDER_SOMETHING_SOMETHING = 'C:\DOC\Vid compress\Dataset\Something_something'

def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def getFiles(dir_path):
    return [f for f in listdir(dir_path) if isfile(join(dir_path, f))]

def read_original_md5_checksums(dir_path):
    result = {}
    for file in getFiles(dir_path):
        if file.endswith('md5'):
            file_no_extension = file.split('.')[0]
            checksum = open(join(dir_path, file), 'r').read(32)
            result[file_no_extension] = checksum
    return result

def compute_checksums(dir_path):
    result = {}
    for file in getFiles(dir_path):
        if not file.endswith('md5'):
            result[file] = md5(join(dir_path, file))
            print('Computed checksum for file', file)
    return result

def md5_verify_dir(dir_path):
    unmatching = []
    original = read_original_md5_checksums(dir_path)
    computed = compute_checksums(dir_path)
    for file, original_checksum in original.items():
        if not original_checksum == computed[file]:
            unmatching.append(file)
    print('Results for', dir_path)
    if len(unmatching) > 0:
        print('>>> Unmatching MD5 Checksums <<<')
        for file in unmatching:
            print(file)
    else:
        print('>>> All checksums matched <<<')
    return unmatching

if __name__ == '__main__':
    md5_verify_dir(VERIFIED_FOLDER_JESTER)
    md5_verify_dir(VERIFIED_FOLDER_SOMETHING_SOMETHING)
