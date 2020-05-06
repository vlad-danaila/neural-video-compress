import os
from os.path import join, exists
import random
import numpy as np
import shutil

class VidDataset():
    def __init__(self, dir, fps):
        self.dir = dir
        self.fps = fps

class VideoInfo():
    def __init__(self, path, fps):
        self.path = path
        self.fps = int(fps)

    def __repr__(self):
        return self.path + ' ' + self.fps

DATASET_ROOT = 'C:\\DOC\\Vid compress\\Dataset\\Experimental'
RAW_VIDEO_PATH = join(DATASET_ROOT, 'raw')
FRAMES_PATH = join(DATASET_ROOT, 'frames')

TRAIN_PATH = join(FRAMES_PATH, 'train')
TEST_PATH = join(FRAMES_PATH, 'test')
VALIDATION_PATH = join(FRAMES_PATH, 'validation')

TRAIN_LIST_PATH = join(FRAMES_PATH, 'train_list')
TEST_LIST_PATH = join(FRAMES_PATH, 'test_list')
VALIDATION_LIST_PATH = join(FRAMES_PATH, 'validation_list')

SOMETHING_SOMETHING_DIR = 'Something_something'
PATH_SOMETHING_SOMETHING_RAW = join(RAW_VIDEO_PATH, SOMETHING_SOMETHING_DIR)
FPS_SOMETHING_SOEMTHING = 12
SOMETHING_SOMETHING = VidDataset(PATH_SOMETHING_SOMETHING_RAW, FPS_SOMETHING_SOEMTHING)

CHARDES_DIR = 'Chardes'
PATH_CHARDES_RAW = join(RAW_VIDEO_PATH, CHARDES_DIR)
FPS_CHARDES = 30
CHARDES = VidDataset(PATH_CHARDES_RAW, FPS_CHARDES)

EGO_DIR = 'ChardesEgo'
PATH_EGO_RAW = join(RAW_VIDEO_PATH, EGO_DIR)
FPS_EGO = 30
CHARDES_EGO = VidDataset(PATH_EGO_RAW, FPS_EGO)

SCALE_BIG = 300
SCALE_SMALL = 150

FRAME_COUNT_TRESHOLD_FOR_VIDEO_SPLIT = 100

TRAIN_TEST_VALIDATION = 0.9, 0.05, 0.05

def check_error(status_code, error_msg):
    if status_code > 0:
        raise Exception(error_msg)

def make_dir_if_missing(dir):
    if not exists(dir):
        os.mkdir(dir)

def make_frames_dir():
    assert exists(PATH_SOMETHING_SOMETHING_RAW)
    make_dir_if_missing(FRAMES_PATH)
    make_dir_if_missing(TRAIN_PATH)
    make_dir_if_missing(TEST_PATH)
    make_dir_if_missing(VALIDATION_PATH)

def extract_frames(in_file, out_file, scale, fps):
    cmd = 'ffmpeg -hide_banner -loglevel panic -i "{in_file}" -vf scale=w={scale}:h={scale}:force_original_aspect_ratio=decrease -r {fps}/1 "{out_file}{separator}%d.jpg"'\
        .format(in_file = in_file, out_file = out_file, scale = scale, fps = fps, separator = os.path.sep)
    status_code = os.system(cmd)
    check_error(status_code, 'While extracting frames from file {}, command was ##{}##'.format(in_file, cmd))

def make_metadata_file(fps, count, scale, source, out_dir):
    lines = [
        'fps={}\n'.format(fps),
        'count={}\n'.format(count),
        'scale={}\n'.format(scale),
        'source={}'.format(source)
    ]
    with open(join(out_dir, 'metadata'), 'w') as meta_file:
        meta_file.writelines(lines)

def compute_train_test_val_video_lists(videos_dir):
    train_ratio, test_ratio, val_ratio = TRAIN_TEST_VALIDATION
    all_videos = set(os.listdir(videos_dir))
    test_count = int(test_ratio * len(all_videos))
    val_count = int(val_ratio * len(all_videos))
    test_videos = random.sample(all_videos, test_count)
    remaining_videos = all_videos - set(test_videos)
    val_videos = random.sample(remaining_videos, val_count)
    train_videos = list(remaining_videos - set(val_videos))
    return {
        TRAIN_PATH: train_videos,
        TEST_PATH: test_videos,
        VALIDATION_PATH: val_videos
    }

# TODO Remove temp folder aftor not needed anymore of the first if branch
def frames_from_videos(videos_info, frames_dir, scale, counter = 0):
    for vid_info in videos_info:
        try:
            fps = vid_info.fps
            file_in = vid_info.path
            dir_temp = join(frames_dir, 'temp_{}'.format(counter))
            make_dir_if_missing(dir_temp)
            extract_frames(file_in, dir_temp, scale, fps)
            frames_count = len(os.listdir(dir_temp))
            if frames_count > FRAME_COUNT_TRESHOLD_FOR_VIDEO_SPLIT:
                counter = split_frame_folder(frames_count, dir_temp, frames_dir, counter, fps, scale, file_in)
                # TODO Handle metadata file in context of video split
            else:
                os.rename(dir_temp, join(frames_dir, str(counter)))
                make_metadata_file(fps, frames_count, scale, file_in, dir_temp)
                counter += 1
        except Exception as e:
            print('Exception at counter', counter, '; File', file_in, e)
        print('Step', counter)
    return counter

def compute_video_split_intervals(frames_count):
    # Video frames are numbered starting with 1
    split_numb = (frames_count // FRAME_COUNT_TRESHOLD_FOR_VIDEO_SPLIT) + 2
    bounderies = np.floor(np.linspace(1, frames_count, split_numb))
    intervals = [[int(bounderies[i] + 1), int(bounderies[i + 1])] for i in range(len(bounderies) - 1)]
    intervals[0][0] = 1
    return intervals

def split_frame_folder(frames_count, frames_dir, parent_dir, counter, fps, scale, file_in):
    intervals = compute_video_split_intervals(frames_count)
    for interval in intervals:
        out_dir = join(parent_dir, str(counter))
        make_dir_if_missing(out_dir)
        for i in range(interval[0], interval[-1] + 1):
            frame_file = join(frames_dir, str(i) + '.jpg')
            shutil.move(frame_file, out_dir)
        frame_count_for_split = interval[-1] - interval[0] + 1
        make_metadata_file(fps, frame_count_for_split, scale, file_in, out_dir)
        counter += 1
    return counter

def write_list_to_file(list, file):
    with open(file, 'a') as f:
        f.writelines('\n'.join(list) + '\n')

def read_videos_list_from_file(file_path):
    with open(file_path, 'r') as file:
        return file.readlines()

def read_train_test_val_splits_from_files():
    train_list = read_videos_list_from_file(TRAIN_LIST_PATH)
    test_list = read_videos_list_from_file(TEST_LIST_PATH)
    val_list = read_videos_list_from_file(VALIDATION_LIST_PATH)
    return {
        TRAIN_PATH: train_list,
        TEST_PATH: test_list,
        VALIDATION_PATH: val_list
    }

def path_and_fps(vid_dataset, video):
    return '{path};{fps}'.format(path = join(vid_dataset.dir, video), fps = vid_dataset.fps)

def create_train_test_val_file_splits(datasets):
    for vid_dataset in datasets:
        train_test_val = compute_train_test_val_video_lists(vid_dataset.dir)
        for frames_path, videos in train_test_val.items():
            videos_list_file = join(FRAMES_PATH, os.path.split(frames_path)[-1] + '_list')
            videos_info = [path_and_fps(vid_dataset, vid) for vid in videos]
            write_list_to_file(videos_info, videos_list_file)

def get_vide_info_from_string(text: str):
    split = text.split(';')
    return VideoInfo(path = split[0], fps = split[1])

def merge_datasets_and_extract_frames(train_test_val, scale):
    counter = 0
    for frames_path, videos_info_text in train_test_val.items():
        video_infos = [get_vide_info_from_string(vid_inf) for vid_inf in videos_info_text]
        counter = frames_from_videos(video_infos, frames_path, scale, counter)

if __name__ == '__main__':
    make_frames_dir()
    # datasets = [SOMETHING_SOMETHING, CHARDES, CHARDES_EGO]
    datasets = [CHARDES, CHARDES_EGO]
    # Uncomment if not having train/test/validation file splits
    # create_train_test_val_file_splits(datasets)
    train_test_val = read_train_test_val_splits_from_files()
    merge_datasets_and_extract_frames(train_test_val, SCALE_BIG)