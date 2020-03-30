import os
from os.path import join, exists

DATASET_ROOT = 'C:\\DOC\\Vid compress\\Dataset\\Experimental'
RAW_VIDEO_PATH = join(DATASET_ROOT, 'raw')
FRAMES_PATH = join(DATASET_ROOT, 'frames')

SOMETHING_SOMETHING_DIR = 'Something_something'
PATH_SOMETHING_SOMETHING_RAW = join(RAW_VIDEO_PATH, SOMETHING_SOMETHING_DIR)
PATH_SOMETHING_SOMETHING_FRAMES = join(FRAMES_PATH, SOMETHING_SOMETHING_DIR)

CHARDES_DIR = 'Chardes'
PATH_CHARDES_RAW = join(RAW_VIDEO_PATH, CHARDES_DIR)
PATH_CHARDES_FRAMES = join(FRAMES_PATH, CHARDES_DIR)

EGO_DIR = 'ChardesEgo'
PATH_EGO_RAW = join(RAW_VIDEO_PATH, EGO_DIR)
PATH_EGO_FRAMES = join(FRAMES_PATH, EGO_DIR)

def check_error(status_code, error_msg):
    if status_code > 0:
        raise Exception(error_msg)

def make_dir_if_missing(dir):
    if not exists(dir):
        os.mkdir(dir)

def make_frames_dir():
    assert exists(PATH_SOMETHING_SOMETHING_RAW)
    make_dir_if_missing(FRAMES_PATH)

def extract_frames(in_file, out_file, fps):
    cmd = 'ffmpeg -i "{in_file}" -r {fps}/1 "{out_file}{separator}%03d.jpg"'\
        .format(in_file = in_file, out_file = out_file, fps = fps, separator = os.path.sep)
    status_code = os.system(cmd)
    check_error(status_code, 'While extracting frames from file {}, command was ##{}##'.format(in_file, cmd))

def frames_from_videos(videos_dir, frames_dir, counter = 0):
    for video_file in os.listdir(videos_dir):
        try:
            file_in = join(videos_dir, video_file)
            file_out = join(frames_dir, str(counter))
            make_dir_if_missing(file_out)
            extract_frames(file_in, file_out, fps = 12)
        except Exception as e:
            print('Exception at counter', counter, '; File', video_file, e)
        counter += 1
    return counter

if __name__ == '__main__':
    make_frames_dir()
    counter = frames_from_videos(PATH_SOMETHING_SOMETHING_RAW, FRAMES_PATH, counter = 148093)
    counter = frames_from_videos(PATH_CHARDES_RAW, FRAMES_PATH, counter=counter)
    counter = frames_from_videos(PATH_EGO_RAW, FRAMES_PATH, counter=counter)