"""The dataloader for the UBFC-PHYS dataset.

Details for the UBFC-PHYS Dataset see https://sites.google.com/view/ybenezeth/ubfc-phys.
If you use this dataset, please cite this paper:
R. Meziati Sabour, Y. Benezeth, P. De Oliveira, J. Chappé, F. Yang. 
"UBFC-Phys: A Multimodal Database For Psychophysiological Studies Of Social Stress", 
IEEE Transactions on Affective Computing, 2021.
"""
import glob
import os
import re
from multiprocessing import Pool, Process, Value, Array, Manager

import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm
import csv
import pandas as pd

class UBFCPHYSLoader(BaseLoader):
    """The data loader for the UBFC-PHYS dataset."""

    def __init__(self, name, data_path, config_data, device=None):
        """Initializes an UBFC-PHYS dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "RawData" for below dataset structure:
                -----------------
                     RawData/
                     |   |-- s1/
                     |       |-- vid_s1_T1.avi
                     |       |-- vid_s1_T2.avi
                     |       |-- vid_s1_T3.avi
                     |       |...
                     |       |-- bvp_s1_T1.csv
                     |       |-- bvp_s1_T2.csv
                     |       |-- bvp_s1_T3.csv
                     |   |-- s2/
                     |       |-- vid_s2_T1.avi
                     |       |-- vid_s2_T2.avi
                     |       |-- vid_s2_T3.avi
                     |       |...
                     |       |-- bvp_s2_T1.csv
                     |       |-- bvp_s2_T2.csv
                     |       |-- bvp_s2_T3.csv
                     |...
                     |   |-- sn/
                     |       |-- vid_sn_T1.avi
                     |       |-- vid_sn_T2.avi
                     |       |-- vid_sn_T3.avi
                     |       |...
                     |       |-- bvp_sn_T1.csv
                     |       |-- bvp_sn_T2.csv
                     |       |-- bvp_sn_T3.csv
                -----------------
                name(string): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        self.filtering = config_data.FILTERING
        super().__init__(name, data_path, config_data, device)

    def get_raw_data(self, data_path):
        """Returns data directories under the path(For UBFC-PHYS dataset)."""
        data_dirs = glob.glob(data_path + os.sep + "s*" + os.sep + "*.avi")
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = [{"index": re.search(
            'vid_(.*).avi', data_dir).group(1), "path": data_dir} for data_dir in data_dirs]
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values."""
        if begin == 0 and end == 1:  # return the full directory if begin == 0 and end == 1
            return data_dirs

        file_num = len(data_dirs)
        choose_range = range(int(begin * file_num), int(end * file_num))
        data_dirs_new = []

        for i in choose_range:
            data_dirs_new.append(data_dirs[i])

        return data_dirs_new

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """   invoked by preprocess_dataset for multi_process.   """
        filename = os.path.split(data_dirs[i]['path'])[-1]
        saved_filename = data_dirs[i]['index']

        video_path = os.path.join(data_dirs[i]['path'])
        bvp_path = os.path.join(os.path.dirname(data_dirs[i]['path']),"bvp_{0}.csv".format(saved_filename))

        if self._can_stream_resize_preprocess(config_preprocess):
            frames = self.read_video_crop_resize_streaming(video_path, config_preprocess)
        else:
            # Read Frames
            frames = self.read_video(video_path)

        # Read Labels
        if getattr(config_preprocess, "USE_PSUEDO_PPG_LABEL", False):
            bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
        else:
            bvps = self.read_wave(bvp_path)

        bvps = BaseLoader.resample_ppg(bvps, frames.shape[0])

        if self._can_stream_resize_preprocess(config_preprocess):
            frames_clips, bvps_clips = self.preprocess_resized_frames(frames, bvps, config_preprocess)
        else:
            frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
        file_list_dict[i] = input_name_list

    @staticmethod
    def _can_stream_resize_preprocess(config_preprocess):
        """Use a memory-safe UBFC-Phys path when it is output-equivalent.

        The original path first loads the full 1024x1024 video and then calls
        crop_face_resize(). For the KR-RPPG -> UBFC-Phys configs, face detection
        is fixed to the first frame and labels are device BVP, so we can stream
        frames through the same crop/resize operation and then run the unchanged
        normalization and chunking code.
        """
        if os.environ.get("RPPG_UBFCPHYS_STREAM_PREPROCESS", "1") == "0":
            return False
        crop = config_preprocess.CROP_FACE
        detection = crop.DETECTION
        return (
            not getattr(config_preprocess, "USE_PSUEDO_PPG_LABEL", False)
            and not detection.DO_DYNAMIC_DETECTION
            and not detection.USE_MEDIAN_FACE_BOX
        )

    def read_video_crop_resize_streaming(self, video_file, config_preprocess):
        """Read a UBFC-Phys video frame-by-frame and reproduce crop_face_resize().

        This avoids materializing the full-resolution video while preserving the
        downstream array that crop_face_resize() would have returned.
        """
        vid_obj = cv2.VideoCapture(video_file)
        vid_obj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = vid_obj.read()
        if not success:
            vid_obj.release()
            raise ValueError(f"Unable to read video file: {video_file}")

        first_frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
        crop = config_preprocess.CROP_FACE
        resize = config_preprocess.RESIZE

        if crop.DO_CROP_FACE:
            face_region = self.face_detection(
                first_frame,
                crop.BACKEND,
                crop.USE_LARGE_FACE_BOX,
                crop.LARGE_BOX_COEF)
            face_region = np.asarray([face_region], dtype='int')[0]
        else:
            face_region = np.asarray([0, 0, first_frame.shape[0], first_frame.shape[1]], dtype='int')

        frame_count = int(vid_obj.get(cv2.CAP_PROP_FRAME_COUNT))
        channels = first_frame.shape[2]
        if frame_count > 0:
            resized_frames = np.zeros((frame_count, resize.H, resize.W, channels))
        else:
            resized_frames = []

        def crop_resize_one(rgb_frame):
            if crop.DO_CROP_FACE:
                rgb_frame = rgb_frame[
                    max(face_region[1], 0):min(face_region[1] + face_region[3], rgb_frame.shape[0]),
                    max(face_region[0], 0):min(face_region[0] + face_region[2], rgb_frame.shape[1])]
            return cv2.resize(rgb_frame, (resize.W, resize.H), interpolation=cv2.INTER_AREA)

        index = 0
        while success:
            rgb_frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            resized_frame = crop_resize_one(rgb_frame)
            if frame_count > 0 and index < frame_count:
                resized_frames[index] = resized_frame
            else:
                resized_frames.append(resized_frame)
            index += 1
            success, frame = vid_obj.read()
        vid_obj.release()

        if frame_count > 0:
            return resized_frames[:index]
        return np.asarray(resized_frames, dtype=float)

    def preprocess_resized_frames(self, frames, bvps, config_preprocess):
        """Run the unchanged post-resize preprocessing path."""
        data = list()
        for data_type in config_preprocess.DATA_TYPE:
            f_c = frames.copy()
            if data_type == "Raw":
                data.append(f_c)
            elif data_type == "DiffNormalized":
                data.append(BaseLoader.diff_normalize_data(f_c))
            elif data_type == "Standardized":
                data.append(BaseLoader.standardized_data(f_c))
            else:
                raise ValueError("Unsupported data type!")
        data = np.concatenate(data, axis=-1)
        if config_preprocess.LABEL_TYPE == "Raw":
            pass
        elif config_preprocess.LABEL_TYPE == "DiffNormalized":
            bvps = BaseLoader.diff_normalize_label(bvps)
        elif config_preprocess.LABEL_TYPE == "Standardized":
            bvps = BaseLoader.standardized_label(bvps)
        else:
            raise ValueError("Unsupported label type!")

        if config_preprocess.DO_CHUNK:
            frames_clips, bvps_clips = self.chunk(data, bvps, config_preprocess.CHUNK_LENGTH)
        else:
            frames_clips = np.array([data])
            bvps_clips = np.array([bvps])

        return frames_clips, bvps_clips

    def load_preprocessed_data(self):
        """ Loads the preprocessed data listed in the file list.

        Args:
            None
        Returns:
            None
        """
        file_list_path = self.file_list_path  # get list of files in
        file_list_df = pd.read_csv(file_list_path)
        base_inputs = file_list_df['input_files'].tolist()
        filtered_inputs = []

        for input in base_inputs:
            input_name = input.split(os.sep)[-1].split('.')[0].rsplit('_', 1)[0]
            if self.filtering.USE_EXCLUSION_LIST and input_name in self.filtering.EXCLUSION_LIST :
                # Skip loading the input as it's in the exclusion list
                continue
            if self.filtering.SELECT_TASKS and not any(task in input_name for task in self.filtering.TASK_LIST):
                # Skip loading the input as it's not in the task list
                continue
            filtered_inputs.append(input)

        if not filtered_inputs:
            raise ValueError(self.dataset_name + ' dataset loading data error!')
        
        filtered_inputs = sorted(filtered_inputs)  # sort input file name list
        labels = [input_file.replace("input", "label") for input_file in filtered_inputs]
        self.inputs = filtered_inputs
        self.labels = labels
        self.preprocessed_data_len = len(filtered_inputs)

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T,H,W,3) """
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = list()
        while success:
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frames.append(frame)
            success, frame = VidObj.read()
        return np.asarray(frames)

    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        bvp = []
        with open(bvp_file, "r") as f:
            d = csv.reader(f)
            for row in d:
                bvp.append(float(row[0]))
        return np.asarray(bvp)
