"""The data loader for the KR-rPPG dataset (BFDT release layout).

KR-rPPG is a clinical rPPG dataset collected at three hospital sites in Korea. This loader reads
the de-identified release (CC BY-NC 4.0, obtained from the KR-rPPG dataset record):

    ${KR_RPPG_ROOT}/                     # the released "KR-rPPG/" folder
    |-- metadata/
    |   |-- fold_assignment.csv          # participant_key (pNNN), fold (0-4)
    |   `-- ...
    |-- p001/
    |   |-- T00/                         # T00 talking, T01 stationary, T02 yaw, T03 pitch
    |   |   |-- video_bfdt_ffv1.avi      # BFDT de-identified video (FFV1, 640x480, 30 fps)
    |   |   |-- BVP_synchronized.csv     # timestamp, bvp (frame-aligned contact BVP)
    |   |   |-- face_bboxes.csv          # x1, y1, x2, y2 (one row per frame)
    |   |   `-- ...
    |   `-- T01/ ...
    `-- p296/

Session index (used for cache and prediction names): "<pNNN>_<Txx>" (release ID and task).

Faces are cropped with the median of the released per-frame face boxes (enlarged by
LARGE_BOX_COEF) and the first 6 s of every session are skipped, as in the manuscript benchmark.

The environment variable KR_RPPG_VIDEO_FILE (optional) overrides the name of the video file in each
session folder (default video_bfdt_ffv1.avi).

K-fold mode (KFOLD_CSV non-empty and KFOLD_ID >= 0):
  * KFOLD_TEST_FOLD=True  -> only participants of fold KFOLD_ID (out-of-fold evaluation);
  * otherwise             -> the train pool (all other folds), sorted by release ID, sliced by
                             BEGIN/END (manuscript: train 0.0-0.9, valid 0.9-1.0).
  * KFOLD_SPLIT_CSV (optional, columns fold, participant_key, split with split in {train, valid})
    overrides the BEGIN/END slicing of the train pool: the "train" loader takes split == train and
    the "valid" loader takes split == valid for fold KFOLD_ID.

KR-rPPG adaptation of rPPG-Toolbox (Responsible AI Source Code License v1.1, see ../../LICENSE and
../../NOTICE.md), written for the release layout above.
"""
import glob
import os
import re
import traceback

import cv2
import numpy as np
import pandas as pd
from dataset.data_loader.BaseLoader import BaseLoader

RELEASE_VIDEO = os.environ.get("KR_RPPG_VIDEO_FILE", "video_bfdt_ffv1.avi")
BVP_FILE = "BVP_synchronized.csv"
BBOX_FILE = "face_bboxes.csv"
SKIP_SECONDS = 6
_PARTICIPANT_DIR = re.compile(r"^p\d{3}$")
_SESSION_DIR = re.compile(r"^T0[0-3]$")


class KoreanrPPGLoader(BaseLoader):
    """The data loader for the KR-rPPG dataset (release layout)."""

    def __init__(self, name, data_path, config_data, device=None):
        """Initializes a KR-rPPG dataloader.

        Args:
            name(str): name of the dataloader ("train", "valid", "test", ...).
            data_path(str): path to the released KR-rPPG folder (the one holding pNNN/ and metadata/).
            config_data(CfgNode): data settings (ref: config.py).
        """
        super().__init__(name, data_path, config_data, device)

    def get_raw_data(self, data_path):
        """Returns one entry per released session that has video, BVP and face boxes."""
        data_dirs = []
        for participant_path in sorted(glob.glob(os.path.join(data_path, "p*"))):
            participant_id = os.path.basename(participant_path)
            if not (os.path.isdir(participant_path) and _PARTICIPANT_DIR.match(participant_id)):
                continue
            for session_path in sorted(glob.glob(os.path.join(participant_path, "T0*"))):
                session_id = os.path.basename(session_path)
                if not _SESSION_DIR.match(session_id):
                    continue
                needed = (RELEASE_VIDEO, BVP_FILE, BBOX_FILE)
                if all(os.path.exists(os.path.join(session_path, f)) for f in needed):
                    data_dirs.append({
                        "index": f"{participant_id}_{session_id}",
                        "path": session_path,
                        "participant": participant_id,
                        "session": session_id,
                    })
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty! Expected <root>/pNNN/T0x/" + RELEASE_VIDEO)
        print(f"Found {len(data_dirs)} sessions in the KR-rPPG release")
        return data_dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns the sessions of the selected participants (no participant spans two splits)."""
        kfold_test_fold = bool(getattr(self.config_data, "KFOLD_TEST_FOLD", False))
        if begin == 0 and end == 1 and not kfold_test_fold:
            return data_dirs

        participants = {}
        for data_dir in data_dirs:
            participants.setdefault(data_dir["participant"], []).append(data_dir)

        kfold_csv = getattr(self.config_data, "KFOLD_CSV", "")
        kfold_id = int(getattr(self.config_data, "KFOLD_ID", -1))
        if kfold_csv and kfold_id >= 0:
            folds = pd.read_csv(kfold_csv, dtype={"participant_key": str})
            test_keys = set(folds.loc[folds["fold"] == kfold_id, "participant_key"])
            if kfold_test_fold:
                selected = sorted(k for k in participants if k in test_keys)
                print(f"[K-Fold TEST] fold {kfold_id}: {len(selected)} participants")
            else:
                pool = sorted(k for k in participants if k not in test_keys)
                selected = self._train_pool_subset(pool, kfold_id, begin, end)
                print(f"[K-Fold] fold {kfold_id} | {self.dataset_name} {begin:.2f}-{end:.2f}: "
                      f"{len(selected)}/{len(pool)} train-pool participants")
        else:
            keys = sorted(participants)
            selected = keys[int(begin * len(keys)):int(end * len(keys))]

        out = []
        for key in selected:
            out.extend(participants[key])
        return out

    def _train_pool_subset(self, pool, kfold_id, begin, end):
        split_csv = getattr(self.config_data, "KFOLD_SPLIT_CSV", "")
        if not split_csv:
            return pool[int(begin * len(pool)):int(end * len(pool))]
        if self.dataset_name not in ("train", "valid"):
            raise ValueError("KFOLD_SPLIT_CSV is only defined for the train and valid loaders")
        split = pd.read_csv(split_csv, dtype={"participant_key": str})
        rows = split[(split["fold"] == kfold_id) & (split["split"] == self.dataset_name)]
        wanted = set(rows["participant_key"])
        unknown = wanted - set(pool)
        if unknown:
            raise ValueError(f"KFOLD_SPLIT_CSV lists {len(unknown)} participants outside the fold-{kfold_id} train pool")
        return [k for k in pool if k in wanted]

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """Invoked by preprocess_dataset for multi_process."""
        saved_filename = data_dirs[i]["index"]
        session_path = data_dirs[i]["path"]
        try:
            if "None" not in config_preprocess.DATA_AUG:
                raise ValueError(f"Unsupported DATA_AUG for {self.dataset_name}: {config_preprocess.DATA_AUG}")
            frames = self.read_video(os.path.join(session_path, RELEASE_VIDEO))
            if config_preprocess.USE_PSUEDO_PPG_LABEL:
                bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
            else:
                bvps = self.read_wave(os.path.join(session_path, BVP_FILE))
            face_bboxes = self.read_face_bboxes(os.path.join(session_path, BBOX_FILE))

            min_len = min(len(frames), len(bvps), len(face_bboxes))
            frames, bvps, face_bboxes = frames[:min_len], bvps[:min_len], face_bboxes[:min_len]

            frames_clips, bvps_clips = self.preprocess_with_bboxes(frames, bvps, face_bboxes, config_preprocess)
            input_name_list, _ = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
            file_list_dict[i] = input_name_list
        except Exception as exc:  # keep the other sessions going; the session is reported and skipped
            print(f"Error processing {saved_filename}: {exc}")
            traceback.print_exc()
            file_list_dict[i] = []

    @staticmethod
    def read_video(video_file, skip_seconds=SKIP_SECONDS, fps=30):
        """Reads a video file after skipping the first `skip_seconds`; returns RGB frames (T, H, W, 3)."""
        vid = cv2.VideoCapture(video_file)
        for _ in range(int(skip_seconds * fps)):
            success, _ = vid.read()
            if not success:
                break
        frames = []
        success, frame = vid.read()
        while success:
            frames.append(cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB))
            success, frame = vid.read()
        vid.release()
        return np.asarray(frames)

    @staticmethod
    def read_wave(bvp_file, skip_seconds=SKIP_SECONDS, fps=30):
        """Reads the 'bvp' column of BVP_synchronized.csv after skipping the first `skip_seconds`."""
        df = pd.read_csv(bvp_file)
        return np.asarray(df["bvp"].values[int(skip_seconds * fps):])

    @staticmethod
    def read_face_bboxes(bbox_file, skip_seconds=SKIP_SECONDS, fps=30):
        """Reads face_bboxes.csv (x1, y1, x2, y2) after skipping the first `skip_seconds`."""
        df = pd.read_csv(bbox_file)
        return df[["x1", "y1", "x2", "y2"]].values[int(skip_seconds * fps):]

    def preprocess_with_bboxes(self, frames, bvps, face_bboxes, config_preprocess):
        """Crops with the released face boxes, then applies the usual toolbox transforms and chunking."""
        frames = self.crop_face_with_bboxes(
            frames, face_bboxes,
            config_preprocess.CROP_FACE.USE_LARGE_FACE_BOX,
            config_preprocess.CROP_FACE.LARGE_BOX_COEF,
            config_preprocess.RESIZE.W,
            config_preprocess.RESIZE.H)

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
        elif config_preprocess.LABEL_TYPE == "DiffOnly":
            bvps = BaseLoader.diff_only_label(bvps)
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

    @staticmethod
    def crop_face_with_bboxes(frames, bboxes, use_larger_box, larger_box_coef, width, height):
        """Crops every frame with the session-median face box (optionally enlarged) and resizes it."""
        total_frames = len(frames)
        resized_frames = np.zeros((total_frames, height, width, 3), dtype=np.float32)

        x1, y1, x2, y2 = np.median(bboxes, axis=0).astype(int)
        if use_larger_box:
            w, h = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            new_w, new_h = int(w * larger_box_coef), int(h * larger_box_coef)
            x1, y1 = cx - new_w // 2, cy - new_h // 2
            x2, y2 = cx + new_w // 2, cy + new_h // 2

        for i in range(total_frames):
            frame = frames[i]
            h_frame, w_frame = frame.shape[:2]
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(w_frame, x2), min(h_frame, y2)
            if x2c > x1c and y2c > y1c:
                face = frame[y1c:y2c, x1c:x2c]
                resized_frames[i] = cv2.resize(face, (width, height), interpolation=cv2.INTER_AREA)
            else:  # invalid box: fall back to the whole frame
                resized_frames[i] = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        return resized_frames
