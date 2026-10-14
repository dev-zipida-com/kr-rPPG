"""Fixed parameters of the KR-rPPG release (manuscript Figure 2 / Table 6)."""

import os

BLOCK_SIZE = 8
KEEP_RATIO = 0.02  # BFDT keep ratio k_r; with 8x8 blocks only the DC coefficient survives
# Supported BFDT block sizes: 8 (the paper setting, default) and 16 (optional). Any other size is refused
# (block 1 is the identity and would keep the raw faces).
RELEASE_BLOCK_SIZES = (8, 16)
FRAME_SIZE = (640, 480)  # (width, height) of released videos
FPS = 30.0

TASKS = ("T00", "T01", "T02", "T03")
SITES = ("incheon", "saintvincent", "seoul")
# The custodian source tree is <site>/<group>/pNNN/T0x/. Group folders are recruitment sub-folders named
# "NN-<label>"; any folder matching this pattern is indexed (their names are never released).
SOURCE_GROUP_PATTERN = r"^\d{2}-[A-Za-z][A-Za-z0-9_-]*$"

RELEASE_DIRNAME = "KR-rPPG"
METADATA_DIRNAME = "metadata"

SOURCE_VIDEO = "video_ZIP_RAW_BGRA.avi"
RELEASE_VIDEO = "video_bfdt_ffv1.avi"
FRAMES_TIMESTAMP = "frames_timestamp.csv"
BVP_SYNC = "BVP_synchronized.csv"
HR_SYNC = "HR_synchronized.csv"
SPO2_SYNC = "SpO2_synchronized.csv"
FACE_BBOXES = "face_bboxes.csv"
INFO_TXT = "info.txt"

SESSION_TABLES = (FRAMES_TIMESTAMP, BVP_SYNC, HR_SYNC, SPO2_SYNC, FACE_BBOXES)
# info.txt (camera model, codec, frame size) stays in the source and is not released.
RELEASE_SESSION_FILES = (RELEASE_VIDEO, BVP_SYNC, HR_SYNC, SPO2_SYNC, FRAMES_TIMESTAMP, FACE_BBOXES)
SOURCE_SESSION_FILES = (SOURCE_VIDEO,) + SESSION_TABLES + (INFO_TXT,)

TABLE_HEADERS = {
    FRAMES_TIMESTAMP: ("frame", "timestamp"),
    BVP_SYNC: ("timestamp", "bvp"),
    HR_SYNC: ("frame", "timestamp", "hr"),
    SPO2_SYNC: ("frame", "timestamp", "spo2"),
    FACE_BBOXES: ("x1", "y1", "x2", "y2"),
}
# Column index of the timestamp in every file that carries one.
TIMESTAMP_COLUMN = {FRAMES_TIMESTAMP: 1, BVP_SYNC: 0, HR_SYNC: 1, SPO2_SYNC: 1}
# The source sync files hold the frame timestamps after a float64 round trip,
# so they differ from frames_timestamp.csv by at most ~2.4e-7 s.
TIMESTAMP_TOLERANCE_S = "0.000001"

# Inter-frame gaps above this are reported as warnings (webcam stalls/reconnects).
FRAME_GAP_WARN_S = "1.0"

# Never written to, whatever --source or --protect say (in addition to the source tree, which is always refused).
# Set on the custodian machine, os.pathsep-separated, e.g. the whole dataset folder that holds <source_root> and
# any other folder that must stay untouched.
PROTECTED_ROOTS = tuple(p for p in os.environ.get("KRRPPG_PROTECTED_ROOTS", "").split(os.pathsep) if p)

# Builder version that last changed the bytes of any per-session output. Session records store it as
# settings["builder_version"]; a build reuses a session only when it matches, so a release that only
# changes metadata/ (1.2.0: 24-column participants.csv) must not bump it or every video is re-encoded.
SESSION_OUTPUT_VERSION = "1.1.0"

# verify warns about any category of a categorical participants.csv column (site, sex, the 0/1 flags) held by
# fewer participants than this. The numeric columns of the manuscript Table 6 coding (1.4.0: age, height, weight,
# bmi, the nine measurements, other_comorbidity_count) have no categories and are not checked against it; they can
# hold rare values, and combinations of columns are not k-anonymous. The residual risk is managed under the data
# use agreement (DUA).
MIN_CATEGORY_COUNT = 5

PARTICIPANT_KEY = "participant_key"
PARTICIPANTS_CSV = "participants.csv"
DICTIONARY_CSV = "participants_dictionary.csv"
AVAILABILITY_CSV = "task_session_availability.csv"
FOLDS_CSV = "fold_assignment.csv"
METADATA_FILES = (PARTICIPANTS_CSV, DICTIONARY_CSV, AVAILABILITY_CSV, FOLDS_CSV)
AVAILABILITY_COLUMNS = tuple(f"{task.lower()}_released" for task in TASKS)

TIMESTAMP_MODES = ("relative", "absolute")
ID_ORDERS = ("random", "sorted")

RELEASE_DIR_MODE = 0o750
RELEASE_FILE_MODE = 0o640
PRIVATE_DIR_MODE = 0o700
PRIVATE_FILE_MODE = 0o600
