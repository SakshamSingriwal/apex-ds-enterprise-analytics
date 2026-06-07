import io
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from scipy.io import wavfile

    SCIPY_WAV_AVAILABLE = True
except ImportError:
    SCIPY_WAV_AVAILABLE = False

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import torchvision

    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

try:
    from transformers import AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    SKLEARN_MI_AVAILABLE = True
except ImportError:
    SKLEARN_MI_AVAILABLE = False


def _diagnostic(success: bool, modality: str, details: Dict[str, Any]) -> Dict[str, Any]:
    return {"success": success, "modality": modality, "details": details}


def profile_modality(df: pd.DataFrame, filename: str = "", mime: str = "") -> Dict[str, Any]:
    if not isinstance(df, pd.DataFrame):
        return _diagnostic(
            False,
            "tabular",
            {
                "error": "Input is not a pandas DataFrame",
                "input_type": type(df).__name__,
            },
        )

    try:
        diag: Dict[str, Any] = {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "dtypes": df.dtypes.apply(lambda dt: str(dt)).to_dict(),
            "missing_values": int(df.isnull().sum().sum()),
            "missing_pct": float(df.isnull().sum().sum() / max(df.size, 1) * 100),
            "duplicated_rows": int(df.duplicated().sum()),
            "memory_bytes": int(df.memory_usage(deep=True).sum()),
        }

        numeric_cols = df.select_dtypes(include=np.number).columns
        cat_cols = df.select_dtypes(exclude=np.number).columns

        diag["numeric_columns"] = int(len(numeric_cols))
        diag["categorical_columns"] = int(len(cat_cols))
        diag["column_list"] = list(df.columns)

        if len(numeric_cols) > 0:
            diag["numeric_summary"] = df[numeric_cols].describe().to_dict()

        if len(cat_cols) > 0:
            diag["categorical_summary"] = {}
            for col in cat_cols:
                diag["categorical_summary"][col] = {
                    "nunique": int(df[col].nunique()),
                    "top_values": df[col].value_counts(dropna=False).head(5).to_dict(),
                }

        file_ext = Path(filename).suffix.lower() if filename else ""
        modality_type = "tabular"
        if mime:
            if "tabular" in mime or "csv" in mime or "excel" in mime:
                modality_type = "tabular"
            elif "image" in mime or file_ext in [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"]:
                modality_type = "image"
            elif "audio" in mime or file_ext in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]:
                modality_type = "audio"
            elif "video" in mime or file_ext in [".mp4", ".avi", ".mov", ".mkv"]:
                modality_type = "video"
        elif file_ext:
            if file_ext in [".csv", ".tsv", ".xlsx", ".parquet", ".json"]:
                modality_type = "tabular"
            elif file_ext in [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"]:
                modality_type = "image"
            elif file_ext in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]:
                modality_type = "audio"
            elif file_ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
                modality_type = "video"

        diag["modality"] = modality_type
        diag["filename"] = filename
        diag["mime"] = mime
        diag["quality_metrics"] = {
            "completeness_pct": float(
                100 - diag["missing_pct"]
            ),
            "duplication_pct": float(
                diag["duplicated_rows"] / max(diag["rows"], 1) * 100
            ),
            "dimensionality": int(diag["columns"]),
            "sample_size": int(diag["rows"]),
            "sparsity": float(
                diag["missing_values"] / max(diag["rows"] * diag["columns"], 1)
            ),
        }

        return _diagnostic(True, modality_type, diag)
    except Exception as exc:
        return _diagnostic(
            False,
            "tabular",
            {"error": f"Profiling failed: {type(exc).__name__}: {exc}"},
        )


def extract_tabular_features(
    df: pd.DataFrame,
    target: Optional[str] = None,
    n_features: int = 50,
    method: str = "mutual_info",
) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if df.empty:
        return df.copy()

    work_df = df.copy()

    if target and target in work_df.columns:
        y = work_df[target]
        work_df = work_df.drop(columns=[target])
    else:
        y = None

    for col in work_df.columns:
        if work_df[col].dtype == object or str(work_df[col].dtype) == "category":
            work_df[col] = work_df[col].astype(str).astype("category").cat.codes

    work_df = work_df.select_dtypes(include=[np.number])

    if work_df.empty:
        return pd.DataFrame()

    if work_df.isnull().values.any():
        work_df = work_df.fillna(work_df.median())

    if y is not None and SKLEARN_MI_AVAILABLE and method.lower() == "mutual_info":
        try:
            if pd.api.types.is_numeric_dtype(y):
                mi_scores = mutual_info_regression(work_df, y)
            else:
                y_enc = pd.Series(y).astype("category").cat.codes
                mi_scores = mutual_info_classif(work_df, y_enc)
            scores = pd.Series(mi_scores, index=work_df.columns)
            scores = scores.sort_values(ascending=False)
            top_cols = scores.head(min(n_features, len(scores))).index.tolist()
            result = work_df[top_cols].copy()
            if y is not None:
                result[target] = y.reset_index(drop=True)
            return result
        except Exception:
            pass

    variances = work_df.var().sort_values(ascending=False)
    top_cols = variances.head(min(n_features, len(variances))).index.tolist()
    result = work_df[top_cols].copy()
    if y is not None:
        result[target] = y.reset_index(drop=True)
    return result


def preprocess_for_transformer(
    df: pd.DataFrame,
    target: Optional[str] = None,
    task: str = "classification",
) -> Dict[str, Any]:
    if not isinstance(df, pd.DataFrame):
        return _diagnostic(
            False,
            "tabular",
            {"error": "df must be a pandas DataFrame for transformer preprocessing."},
        )

    result: Dict[str, Any] = {}

    try:
        work_df = df.copy()
        if target and target in work_df.columns:
            y = work_df[target]
            X = work_df.drop(columns=[target])
        else:
            y = None
            X = work_df

        cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()
        num_cols = X.select_dtypes(include=np.number).columns.tolist()

        if cat_cols:
            if TRANSFORMERS_AVAILABLE:
                try:
                    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                    texts = X[cat_cols].astype(str).agg(" ".join, axis=1).tolist()
                    encoded = tokenizer(
                        texts,
                        padding=True,
                        truncation=True,
                        max_length=128,
                        return_tensors="pt" if TORCH_AVAILABLE else "np",
                    )
                    if TORCH_AVAILABLE:
                        result["input_ids"] = encoded["input_ids"].numpy()
                        result["attention_mask"] = encoded["attention_mask"].numpy()
                    else:
                        result["input_ids"] = np.array(encoded["input_ids"])
                        result["attention_mask"] = np.array(
                            encoded["attention_mask"]
                        )
                except Exception as exc:
                    result["input_ids"] = pd.get_dummies(
                        X[cat_cols], drop_first=True
                    ).values.astype(np.float32)
                    result["attention_mask"] = np.ones(
                        (result["input_ids"].shape[0],), dtype=np.int64
                    )
            else:
                encoded_matrix = pd.get_dummies(
                    X[cat_cols], drop_first=True
                ).values.astype(np.float32)
                result["input_ids"] = encoded_matrix
                result["attention_mask"] = np.ones(
                    (encoded_matrix.shape[0],), dtype=np.int64
                )
        else:
            result["input_ids"] = np.empty((len(X), 0), dtype=np.float32)
            result["attention_mask"] = np.ones((len(X),), dtype=np.int64)

        if num_cols:
            num_data = X[num_cols].fillna(0).values.astype(np.float32)
            if TORCH_AVAILABLE:
                result["numeric_features"] = torch.from_numpy(num_data)
            else:
                result["numeric_features"] = num_data
        else:
            empty_num = np.empty((len(X), 0), dtype=np.float32)
            if TORCH_AVAILABLE:
                result["numeric_features"] = torch.from_numpy(empty_num)
            else:
                result["numeric_features"] = empty_num

        if y is not None:
            if task == "classification":
                if pd.api.types.is_numeric_dtype(y):
                    result["labels"] = y.values.astype(np.int64)
                else:
                    result["labels"] = (
                        pd.Series(y).astype("category").cat.codes.values.astype(np.int64)
                    )
            else:
                result["labels"] = y.values.astype(np.float32)
        else:
            result["labels"] = None

        result["task"] = task
        result["num_samples"] = int(len(X))
        result["num_features"] = int(
            result["input_ids"].shape[-1] + result["numeric_features"].shape[-1]
        )
        result["feature_names"] = list(X.columns)

        return _diagnostic(True, "tabular", result)
    except Exception as exc:
        return _diagnostic(
            False,
            "tabular",
            {"error": f"Transformer preprocessing failed: {type(exc).__name__}: {exc}"},
        )


def preprocess_image(file_bytes: Union[bytes, bytearray, memoryview]) -> Dict[str, Any]:
    if not PIL_AVAILABLE and not TORCHVISION_AVAILABLE:
        return _diagnostic(
            False,
            "image",
            {
                "error": "PIL (Pillow) and torchvision are both missing. Install one of: pillow, torchvision",
                "pil_available": PIL_AVAILABLE,
                "torchvision_available": TORCHVISION_AVAILABLE,
            },
        )

    try:
        if isinstance(file_bytes, (bytearray, memoryview)):
            file_bytes = bytes(file_bytes)

        if not file_bytes:
            return _diagnostic(False, "image", {"error": "Empty file_bytes provided."})

        img = Image.open(io.BytesIO(file_bytes))
        img.load()

        original_size = img.size
        img_mode = img.mode
        array = np.array(img)

        processed_img = img.convert("RGB")
        target_size = (224, 224)
        processed_img = processed_img.resize(target_size, Image.Resampling.LANCZOS)
        arr = np.array(processed_img).astype(np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)

        output = arr

        if TORCHVISION_AVAILABLE:
            try:
                from torchvision import transforms as tvt

                transform = tvt.Compose([
                    tvt.Resize(target_size),
                    tvt.ToTensor(),
                    tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                tensor = transform(pil_img)
                if TORCH_AVAILABLE:
                    output_tensor = tensor
                else:
                    output_tensor = tensor.numpy()
                result_data = output_tensor
            except Exception:
                result_data = output
        else:
            result_data = output

        details = {
            "original_shape": list(array.shape),
            "original_size": list(original_size),
            "mode": img_mode,
            "processed_shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "backend": "torchvision" if TORCHVISION_AVAILABLE else "PIL/numpy",
            "normalized": TORCHVISION_AVAILABLE,
        }

        return _diagnostic(True, "image", {"array": result_data, "details": details})
    except Exception as exc:
        return _diagnostic(
            False,
            "image",
            {"error": f"Image preprocessing failed: {type(exc).__name__}: {exc}"},
        )


def preprocess_audio(file_bytes: Union[bytes, bytearray, memoryview]) -> Dict[str, Any]:
    if not LIBROSA_AVAILABLE and not SCIPY_WAV_AVAILABLE:
        return _diagnostic(
            False,
            "audio",
            {
                "error": "librosa and scipy.io.wavfile are both missing. Install: librosa[librosa] or scipy",
                "librosa_available": LIBROSA_AVAILABLE,
                "scipy_wav_available": SCIPY_WAV_AVAILABLE,
            },
        )

    try:
        if isinstance(file_bytes, (bytearray, memoryview)):
            file_bytes = bytes(file_bytes)

        if not file_bytes:
            return _diagnostic(False, "audio", {"error": "Empty file_bytes provided."})

        sr = None
        y = None

        if LIBROSA_AVAILABLE:
            try:
                y, sr = librosa.load(io.BytesIO(file_bytes), sr=22050, mono=True)
                backend = "librosa"
            except Exception as exc:
                if SCIPY_WAV_AVAILABLE:
                    try:
                        sr, wav_data = wavfile.read(io.BytesIO(file_bytes))
                        if wav_data.ndim > 1:
                            wav_data = wav_data.mean(axis=1)
                        y = wav_data.astype(np.float32)
                        if y.max() > 1.0:
                            y = y / 32768.0
                        backend = "scipy_wavfile"
                    except Exception as scipy_exc:
                        return _diagnostic(
                            False,
                            "audio",
                            {
                                "error": f"Both librosa ({exc}) and scipy ({scipy_exc}) failed.",
                            },
                        )
                else:
                    return _diagnostic(
                        False,
                        "audio",
                        {"error": f"librosa failed and scipy missing: {exc}"},
                    )
        elif SCIPY_WAV_AVAILABLE:
            sr, wav_data = wavfile.read(io.BytesIO(file_bytes))
            if wav_data.ndim > 1:
                wav_data = wav_data.mean(axis=1)
            y = wav_data.astype(np.float32)
            if y.max() > 1.0:
                y = y / 32768.0
            backend = "scipy_wavfile"

        if y is None:
            return _diagnostic(False, "audio", {"error": "Could not decode audio data."})

        mel_spec = None
        if LIBROSA_AVAILABLE:
            try:
                mel_spec = librosa.feature.melspectrogram(
                    y=y, sr=sr if sr else 22050, n_mels=64
                )
                mel_spec = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)
            except Exception:
                mel_spec = None

        details = {
            "sample_rate": int(sr) if sr else 22050,
            "num_samples": int(len(y)),
            "duration_seconds": float(len(y) / (sr if sr else 22050)),
            "backend": backend,
            "has_mel_spectrogram": mel_spec is not None,
            "dtype": "float32",
        }

        return _diagnostic(
            True,
            "audio",
            {"waveform": y, "sample_rate": sr, "mel_spectrogram": mel_spec, "details": details},
        )
    except Exception as exc:
        return _diagnostic(
            False,
            "audio",
            {"error": f"Audio preprocessing failed: {type(exc).__name__}: {exc}"},
        )


def preprocess_video(
    file_bytes: Union[bytes, bytearray, memoryview],
) -> Dict[str, Any]:
    if not CV2_AVAILABLE:
        return _diagnostic(
            False,
            "video",
            {
                "error": "opencv-python (cv2) is missing. Install: opencv-python",
                "cv2_available": CV2_AVAILABLE,
                "moviepy_available": False,
            },
        )

    try:
        if isinstance(file_bytes, (bytearray, memoryview)):
            file_bytes = bytes(file_bytes)

        if not file_bytes:
            return _diagnostic(False, "video", {"error": "Empty file_bytes provided."})

        temp_path = None
        cap = None
        frames = []

        try:
            temp_path = os.path.join(
                os.environ.get("TEMP", "/tmp"), f"kilo_video_{id(file_bytes)}_{os.getpid()}.tmp"
            )
            with open(temp_path, "wb") as f:
                f.write(file_bytes)

            cap = cv2.VideoCapture(temp_path)
            if not cap.isOpened():
                return _diagnostic(
                    False,
                    "video",
                    {"error": "OpenCV could not open the video file."},
                )

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            target_size = (224, 224)
            max_frames = 16

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % max(1, frame_count // max_frames) == 0 or frame_idx < max_frames:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_frame).resize(
                        target_size, Image.Resampling.LANCZOS
                    )
                    arr = np.array(pil_img).astype(np.float32) / 255.0
                    arr = arr.transpose(2, 0, 1)
                    frames.append(arr)
                frame_idx += 1
                if len(frames) >= max_frames:
                    break

            while len(frames) < max_frames and frame_idx < frame_count:
                ret, frame = cap.read()
                if not ret:
                    break
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame).resize(
                    target_size, Image.Resampling.LANCZOS
                )
                arr = np.array(pil_img).astype(np.float32) / 255.0
                arr = arr.transpose(2, 0, 1)
                frames.append(arr)
                frame_idx += 1

            cap.release()
            cap = None
        finally:
            if cap is not None:
                cap.release()
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

        if not frames:
            return _diagnostic(
                False,
                "video",
                {"error": "No frames could be extracted from the video."},
            )

        video_array = np.stack(frames, axis=0).astype(np.float32)

        details = {
            "backend": "opencv",
            "fps": float(fps) if fps else 0.0,
            "frame_count_total": int(frame_count) if frame_count else 0,
            "frames_extracted": int(len(frames)),
            "resolution": [int(width), int(height)] if width and height else [],
            "target_shape_per_frame": list(video_array.shape[1:]),
            "video_shape": list(video_array.shape),
            "dtype": "float32",
        }

        return _diagnostic(
            True, "video", {"frames": frames, "video_array": video_array, "details": details}
        )
    except Exception as exc:
        return _diagnostic(
            False,
            "video",
            {"error": f"Video preprocessing failed: {type(exc).__name__}: {exc}"},
        )
