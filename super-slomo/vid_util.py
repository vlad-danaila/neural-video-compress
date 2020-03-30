import os
from os.path import join, exists

DATASET_ROOT = 'C:\\DOC\\Vid compress\\Dataset\\Experimental'
RAW_VIDEO_PATH = join(DATASET_ROOT, 'raw')
FRAMES_PATH = join(DATASET_ROOT, 'frames')

SOMETHING_SOMETHING_DIR = 'Something_something'
PATH_SOMETHING_SOMETHING_RAW = join(RAW_VIDEO_PATH, SOMETHING_SOMETHING_DIR)
FPS_SOMETHING_SOEMTHING = 12

CHARDES_DIR = 'Chardes'
PATH_CHARDES_RAW = join(RAW_VIDEO_PATH, CHARDES_DIR)
FPS_CHARDES = 30

EGO_DIR = 'ChardesEgo'
PATH_EGO_RAW = join(RAW_VIDEO_PATH, EGO_DIR)
FPS_EGO = 30

SCALE_BIG = 200
SCALE_SMALL = 100

def check_error(status_code, error_msg):
    if status_code > 0:
        raise Exception(error_msg)

def make_dir_if_missing(dir):
    if not exists(dir):
        os.mkdir(dir)

def make_frames_dir():
    assert exists(PATH_SOMETHING_SOMETHING_RAW)
    make_dir_if_missing(FRAMES_PATH)

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

def frames_from_videos(videos_dir, frames_dir, scale, fps, counter = 0):
    for video_file in os.listdir(videos_dir):
        try:
            file_in = join(videos_dir, video_file)
            file_out = join(frames_dir, str(counter))
            make_dir_if_missing(file_out)
            extract_frames(file_in, file_out, scale, fps)
            frames_count = len(os.listdir(file_out))
            make_metadata_file(fps, frames_count, scale, file_in, file_out)
        except Exception as e:
            print('Exception at counter', counter, '; File', video_file, e)
        counter += 1
    return counter

if __name__ == '__main__':
    make_frames_dir()
    counter = frames_from_videos(PATH_SOMETHING_SOMETHING_RAW, FRAMES_PATH, scale = SCALE_BIG, fps = FPS_SOMETHING_SOEMTHING, counter = 148093)
    counter = frames_from_videos(PATH_CHARDES_RAW, FRAMES_PATH, scale = SCALE_BIG, fps = FPS_CHARDES, counter=counter)
    counter = frames_from_videos(PATH_EGO_RAW, FRAMES_PATH, scale = SCALE_BIG, fps = FPS_EGO, counter=counter)

# TODO split train / test / validation or train / test ?
# TODO cum faci splitul per dataset chardes, chardesego, jester, something-something
# TODO vezi cum aduci si scale mic pt features intermediare