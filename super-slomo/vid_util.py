import os
from os.path import join, exists
import random

class VidDataset():
    def __init__(self, dir, fps):
        self.dir = dir
        self.fps = fps

DATASET_ROOT = 'C:\\DOC\\Vid compress\\Dataset\\Experimental'
RAW_VIDEO_PATH = join(DATASET_ROOT, 'raw')
FRAMES_PATH = join(DATASET_ROOT, 'frames')
TRAIN_PATH = join(FRAMES_PATH, 'train')
TEST_PATH = join(FRAMES_PATH, 'test')
VALIDATION_PATH = join(FRAMES_PATH, 'validation')

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

SCALE_BIG = 200
SCALE_SMALL = 100

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
    cmd = 'ffmpeg -i "{in_file}" -vf scale=w={scale}:h={scale}:force_original_aspect_ratio=decrease -r {fps}/1 "{out_file}{separator}%03d.jpg"'\
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
    return train_videos, test_videos, val_videos

def frames_from_videos(videos_dir, videos_list, frames_dir, scale, fps, counter = 0):
    for video in videos_list:
        try:
            file_in = join(videos_dir, video)
            file_out = join(frames_dir, str(counter))
            make_dir_if_missing(file_out)
            extract_frames(file_in, file_out, scale, fps)
            frames_count = len(os.listdir(file_out))
            make_metadata_file(fps, frames_count, scale, file_in, file_out)
        except Exception as e:
            print('Exception at counter', counter, '; File', file_in, e)
        counter += 1
        print('Step', counter)
    return counter

if __name__ == '__main__':
    make_frames_dir()
    counter = 0
    for vid_dataset in [SOMETHING_SOMETHING, CHARDES, CHARDES_EGO]:
        train_videos, test_videos, val_videos = compute_train_test_val_video_lists(vid_dataset.dir)
        train_test_val = [(TRAIN_PATH, train_videos), (TEST_PATH, test_videos), (VALIDATION_PATH, val_videos)]
        for frames_path, videos in train_test_val:
            counter = frames_from_videos(vid_dataset.dir, videos, frames_path, SCALE_BIG, vid_dataset.fps, counter)

    # train_videos, test_videos, val_videos = compute_train_test_val_video_lists(videos_dir)
    # counter = frames_from_videos(PATH_SOMETHING_SOMETHING_RAW, FRAMES_PATH, scale = SCALE_BIG, fps = FPS_SOMETHING_SOEMTHING)
    # counter = frames_from_videos(PATH_CHARDES_RAW, FRAMES_PATH, scale = SCALE_BIG, fps = FPS_CHARDES, counter=counter)
    # counter = frames_from_videos(PATH_EGO_RAW, FRAMES_PATH, scale = SCALE_BIG, fps = FPS_EGO, counter=counter)

# TODO split train / test / validation or train / test ?
# TODO cum faci splitul per dataset chardes, chardesego, jester, something-something
# TODO vezi cum aduci si scale mic pt features intermediare