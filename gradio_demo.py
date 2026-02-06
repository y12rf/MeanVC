import os

# --- 强制重定向所有缓存到项目所在盘符 (防止填满 C 盘) ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(PROJECT_ROOT, "temp_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# 设置环境变量
os.environ["HF_HOME"] = os.path.join(CACHE_DIR, "hf")
os.environ["TORCH_HOME"] = os.path.join(CACHE_DIR, "torch")
os.environ["GRADIO_TEMP_DIR"] = os.path.join(CACHE_DIR, "gradio")
os.environ["PYTHONPYCACHEPREFIX"] = os.path.join(CACHE_DIR, "pycache")
os.environ["MODELSCOPE_CACHE"] = os.path.join(CACHE_DIR, "modelscope")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# 将相关目录加入 Python 路径，修复原项目的导入问题
import sys

sys.path.append(os.path.join(PROJECT_ROOT, "src", "infer"))
sys.path.append(os.path.join(PROJECT_ROOT, "audio-slicer"))

# 忽略警告
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import gradio as gr
import torch
import librosa
import numpy as np
import os
import json
import torchaudio
import torch.nn as nn
import torchaudio.compliance.kaldi as kaldi
from librosa.filters import mel as librosa_mel_fn
from src.runtime.speaker_verification.verification import init_model as init_sv_model
import subprocess
import shutil
from pathlib import Path
from tqdm import tqdm
import threading
import time
from queue import Queue, Empty
import signal
import traceback
import platform
try:
    from slicer import Slicer as AudioSlicer
except Exception:
    AudioSlicer = None

# UI polish
CUSTOM_CSS = """
#refresh-models-btn {
  width: fit-content;
}
#refresh-models-btn button {
  width: auto;
  padding: 6px 14px;
  border-radius: 999px;
  border: 1px solid rgba(0, 0, 0, 0.15);
  background: linear-gradient(135deg, #f8fafc, #e2e8f0);
  color: #0f172a;
  font-weight: 600;
  letter-spacing: 0.2px;
}
#refresh-models-btn button:hover {
  background: linear-gradient(135deg, #eef2f7, #dbe3ef);
}
#refresh-models-btn button:active {
  transform: translateY(1px);
}
"""

# Constants/Paths
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_CONFIG_PATH = "src/config/config_200ms.json"
CKPT_PATH = "src/ckpt/meanvc_200ms.pt"
ASR_CKPT_PATH = "src/ckpt/fastu2++.pt"
VOCODER_CKPT_PATH = "src/ckpt/vocos.pt"

# 修复模型路径问题 - 创建符号链接或复制文件
RUNTIME_CKPT_DIR = os.path.join(PROJECT_ROOT, "src", "runtime", "ckpt")
os.makedirs(RUNTIME_CKPT_DIR, exist_ok=True)
# 如果源文件存在但目标文件不存在，创建符号链接
src_pt = os.path.join(PROJECT_ROOT, "src", "ckpt", "fastu2++.pt")
dst_pt = os.path.join(RUNTIME_CKPT_DIR, "fastu2++.pt")
if os.path.exists(src_pt) and not os.path.exists(dst_pt):
    try:
        os.symlink(src_pt, dst_pt)
    except (OSError, NotImplementedError):
        # Windows 可能不支持符号链接，复制文件
        shutil.copy2(src_pt, dst_pt)
SV_CKPT_PATH = "src/runtime/speaker_verification/ckpt/wavlm_large_finetune.pth"

# --- Feature Extraction Utils (Copied from src/infer/infer_ref.py with robustness fixes) ---


def _amp_to_db(x, min_level_db):
    min_level = np.exp(min_level_db / 20 * np.log(10))
    min_level = torch.ones_like(x) * min_level
    return 20 * torch.log10(torch.maximum(min_level, x))


def _normalize(S, max_abs_value, min_db):
    return torch.clamp(
        (2 * max_abs_value) * ((S - min_db) / (-min_db)) - max_abs_value,
        -max_abs_value,
        max_abs_value,
    )


class MelSpectrogramFeatures(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        n_fft=1024,
        win_size=640,
        hop_length=160,
        n_mels=80,
        fmin=0,
        fmax=8000,
        center=True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.center = center
        self.mel_basis = {}
        self.hann_window = {}

    def forward(self, y):
        dtype_device = str(y.dtype) + "_" + str(y.device)
        fmax_dtype_device = str(self.fmax) + "_" + dtype_device
        wnsize_dtype_device = str(self.win_size) + "_" + dtype_device
        if fmax_dtype_device not in self.mel_basis:
            mel = librosa_mel_fn(
                sr=self.sample_rate,
                n_fft=self.n_fft,
                n_mels=self.n_mels,
                fmin=self.fmin,
                fmax=self.fmax,
            )
            self.mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(
                dtype=y.dtype, device=y.device
            )
        if wnsize_dtype_device not in self.hann_window:
            self.hann_window[wnsize_dtype_device] = torch.hann_window(self.win_size).to(
                dtype=y.dtype, device=y.device
            )

        spec = torch.stft(
            y,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_size,
            window=self.hann_window[wnsize_dtype_device],
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=False,
        )
        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
        spec = torch.matmul(self.mel_basis[fmax_dtype_device], spec)
        spec = _amp_to_db(spec, -115) - 20
        spec = _normalize(spec, 1, -115)
        return spec


def extract_fbanks(
    wav, sample_rate=16000, mel_bins=80, frame_length=25, frame_shift=12.5
):
    wav = wav * (1 << 15)
    if isinstance(wav, np.ndarray):
        wav = torch.from_numpy(wav)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    fbanks = kaldi.fbank(
        wav,
        frame_length=frame_length,
        frame_shift=frame_shift,
        snip_edges=True,
        num_mel_bins=mel_bins,
        energy_floor=0.0,
        dither=0.0,
        sample_frequency=sample_rate,
    )
    fbanks = fbanks.unsqueeze(0)
    return fbanks


def extract_features_from_audio(
    source_path, reference_path, asr_model, sv_model, mel_extractor, device
):
    source_wav, _ = librosa.load(source_path, sr=16000)
    ref_wav, _ = librosa.load(reference_path, sr=16000)
    spk_emb, prompt_mel = _extract_reference_features(
        ref_wav, sv_model, mel_extractor, device
    )
    bn = _extract_bn_from_wave(source_wav, asr_model, device)
    return bn, spk_emb, prompt_mel


def _extract_reference_features(ref_wav, sv_model, mel_extractor, device):
    ref_wav_tensor = torch.from_numpy(ref_wav).unsqueeze(0).to(device)
    with torch.no_grad():
        spk_emb = sv_model(ref_wav_tensor)
        prompt_mel = mel_extractor(ref_wav_tensor)
        prompt_mel = prompt_mel.transpose(1, 2)
    return spk_emb, prompt_mel


def _extract_bn_from_wave(source_wav, asr_model, device):
    source_fbanks = (
        extract_fbanks(source_wav, frame_length=25, frame_shift=10).float().to(device)
    )
    with torch.no_grad():
        offset = 0
        decoding_chunk_size = 5
        num_decoding_left_chunks = 2
        subsampling = 4
        context = 7
        stride = subsampling * decoding_chunk_size
        required_cache_size = decoding_chunk_size * num_decoding_left_chunks
        decoding_window = (decoding_chunk_size - 1) * subsampling + context
        att_cache = torch.zeros((0, 0, 0, 0), device=device)
        cnn_cache = torch.zeros((0, 0, 0, 0), device=device)

        bn_chunks = []
        for i in range(0, source_fbanks.shape[1], stride):
            fbank_chunk = source_fbanks[:, i : i + decoding_window, :]
            if fbank_chunk.shape[1] < required_cache_size:
                pad_size = required_cache_size - fbank_chunk.shape[1]
                fbank_chunk = torch.nn.functional.pad(
                    fbank_chunk, (0, 0, 0, pad_size), mode="constant", value=0.0
                )

            encoder_output, att_cache, cnn_cache = asr_model.forward_encoder_chunk(
                fbank_chunk, offset, required_cache_size, att_cache, cnn_cache
            )
            offset += encoder_output.size(1)
            bn_chunks.append(encoder_output)

        bn = torch.cat(bn_chunks, dim=1)
        bn = bn.transpose(1, 2)
        bn = torch.nn.functional.interpolate(
            bn, size=int(bn.shape[2] * 4), mode="linear", align_corners=False
        )
        bn = bn.transpose(1, 2)
    return bn


def _split_long_audio(audio, max_samples, overlap=0):
    if max_samples <= 0 or audio.shape[0] <= max_samples:
        return [audio]
    chunks = []
    start = 0
    while start < audio.shape[0]:
        end = min(start + max_samples, audio.shape[0])
        chunks.append(audio[start:end])
        if end >= audio.shape[0]:
            break
        start = max(0, end - overlap)
    return chunks


def _slice_audio_with_slicer(
    audio,
    sr,
    db_thresh=None,
    min_len_ms=None,
    win_l_ms=None,
    win_s_ms=None,
    max_sil_ms=None,
    max_chunk_sec=None,
    overlap_ms=0,
):
    if AudioSlicer is None:
        return [audio]
    db_thresh = float(
        os.getenv("MEANVC_SLICE_DB_THRESH", "-40") if db_thresh is None else db_thresh
    )
    min_len_ms = int(
        os.getenv("MEANVC_SLICE_MIN_LEN_MS", "3000")
        if min_len_ms is None
        else min_len_ms
    )
    win_l_ms = int(
        os.getenv("MEANVC_SLICE_WIN_L_MS", "300") if win_l_ms is None else win_l_ms
    )
    win_s_ms = int(
        os.getenv("MEANVC_SLICE_WIN_S_MS", "20") if win_s_ms is None else win_s_ms
    )
    max_sil_ms = int(
        os.getenv("MEANVC_SLICE_MAX_SIL_MS", "500") if max_sil_ms is None else max_sil_ms
    )
    slicer = AudioSlicer(
        sr=sr,
        db_threshold=db_thresh,
        min_length=min_len_ms,
        win_l=win_l_ms,
        win_s=win_s_ms,
        max_silence_kept=max_sil_ms,
    )
    chunks = slicer.slice(audio)
    if max_chunk_sec is None:
        max_chunk_sec = float(os.getenv("MEANVC_MAX_CHUNK_SEC", "15"))
    max_samples = int(sr * max_chunk_sec) if max_chunk_sec and max_chunk_sec > 0 else 0
    overlap_samples = int(sr * (overlap_ms / 1000.0)) if overlap_ms else 0
    final_chunks = []
    for c in chunks:
        final_chunks.extend(_split_long_audio(c, max_samples, overlap=overlap_samples))
    return [c for c in final_chunks if c is not None and len(c) > 0]


@torch.inference_mode()
def inference(model, vocos, bn, spk_emb, prompt_mel, chunk_size, steps, device):
    if steps == 1:
        timesteps = torch.tensor([1.0, 0.0], device=device)
    elif steps == 2:
        timesteps = torch.tensor([1.0, 0.8, 0.0], device=device)
    else:
        timesteps = torch.linspace(1.0, 0.0, steps + 1, device=device)

    seq_len = bn.shape[1]
    x_pred = []
    B = 1
    kv_cache = None
    use_kv_cache = True

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        bn_chunk = bn[:, start:end]
        chunk_len = bn_chunk.shape[1]
        x = torch.randn(B, chunk_len, 80, device=device, dtype=bn_chunk.dtype)

        for i in range(steps):
            t = timesteps[i].item()
            r = timesteps[i + 1].item()
            t_tensor = torch.full((B,), t, device=x.device)
            r_tensor = torch.full((B,), r, device=x.device)

            def _call_model(curr_kv_cache, offset_val):
                return model(
                    x,
                    t_tensor,
                    r_tensor,
                    bn_chunk,
                    spk_emb,
                    prompt_mel,
                    None,  # cache
                    offset_val,  # offset = current position
                    curr_kv_cache,
                )

            try:
                u, tmp_kv_cache = _call_model(kv_cache if use_kv_cache else None, start if use_kv_cache else 0)
            except RuntimeError as e:
                msg = str(e)
                if use_kv_cache and (
                    "apply_rotary_pos_emb" in msg
                    or "size of tensor a" in msg
                    or "rotary" in msg
                ):
                    print("[DiT] KV cache mismatch detected. Disabling KV cache and retrying.")
                    use_kv_cache = False
                    kv_cache = None
                    u, tmp_kv_cache = _call_model(None, 0)
                else:
                    raise
            x = x - (t - r) * u
            kv_cache = tmp_kv_cache if use_kv_cache else None

        x_pred.append(x)

        if use_kv_cache and start > 40 and kv_cache is not None:
            # 检查kv_cache结构是否完整
            try:
                if (
                    kv_cache[0] is not None
                    and kv_cache[0][0] is not None
                    and kv_cache[0][0].shape[2] > 100
                ):
                    for i in range(len(kv_cache)):
                        if kv_cache[i] is not None and kv_cache[i][0] is not None:
                            new_k = kv_cache[i][0][:, :, -100:, :]
                            new_v = kv_cache[i][1][:, :, -100:, :]
                            kv_cache[i] = (new_k, new_v)
            except (TypeError, IndexError):
                pass

    x_pred = torch.cat(x_pred, dim=1)
    mel = x_pred.transpose(1, 2)
    mel = (mel + 1) / 2
    mel = mel.float()
    y_g_hat = _vocos_decode(mel, device)

    return mel, y_g_hat


def _vocos_decode(mel, device):
    if _models.get("vocos_force_cpu") or os.getenv("MEANVC_VOCOS_CPU") == "1":
        return _vocos_decode_cpu(mel, device)
    try:
        return _models["vocos"].decode(mel)
    except RuntimeError as e:
        if "UNSUPPORTED DTYPE: complex" not in str(e):
            raise
        _models["vocos_force_cpu"] = True
        print("[Vocos] GPU decode failed (complex dtype). Falling back to CPU (sticky).")
        return _vocos_decode_cpu(mel, device)


def _vocos_decode_cpu(mel, device):
    vocos_cpu = _models.get("vocos_cpu")
    if vocos_cpu is None:
        voc_path = _models.get("vocoder_path", VOCODER_CKPT_PATH)
        vocos_cpu = torch.jit.load(voc_path, map_location="cpu")
        vocos_cpu.eval()
        _models["vocos_cpu"] = vocos_cpu
    mel_cpu = mel.float().cpu()
    wav = vocos_cpu.decode(mel_cpu)
    if device.startswith("cuda"):
        return wav.to(device)
    return wav


# --- Model Management ---

_models = None

def get_model_list():
    """Scan for available models in ckpt directories"""
    search_paths = [
        os.path.join(PROJECT_ROOT, "src", "ckpt"),
        os.path.join(PROJECT_ROOT, "src", "runtime", "ckpt"),
        os.path.join(PROJECT_ROOT, "src", "runtime", "speaker_verification", "ckpt"),
        os.path.join(PROJECT_ROOT, "ckpts"),
        os.path.join(PROJECT_ROOT, "src", "ckpts"),
        os.path.join(PROJECT_ROOT, "results"),
    ]
    model_files = []
    for p in search_paths:
        if not os.path.exists(p):
            continue
        for root, _, files in os.walk(p):
            for f in files:
                if f.endswith((".pt", ".pth")):
                    model_files.append(os.path.join(root, f))
    # Sort by modification time (newest first), then name
    model_files = list(dict.fromkeys(model_files))  # de-dupe while preserving order
    model_files.sort(key=lambda x: (os.path.getmtime(x), x), reverse=True)
    return model_files

def unload_models():
    global _models
    if _models:
        del _models
        _models = None
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return "模型已卸载，显存已释放"

def load_models(ckpt_path=None, asr_path=None, vocoder_path=None, sv_path=None, config_path=None):
    global _models
    
    # Use defaults if not provided
    ckpt_path = ckpt_path or CKPT_PATH
    asr_path = asr_path or ASR_CKPT_PATH
    vocoder_path = vocoder_path or VOCODER_CKPT_PATH
    sv_path = sv_path or SV_CKPT_PATH
    config_path = config_path or MODEL_CONFIG_PATH

    print(f"Loading models to {DEVICE}...")
    print(f"Main Model: {ckpt_path}")
    
    try:
        with open(config_path) as f:
            model_config = json.load(f)

        print(" - Loading DiT model...")
        dit_model = torch.jit.load(ckpt_path, map_location=DEVICE).to(DEVICE).float()

        print(" - Loading Vocos (Vocoder)...")
        vocos = torch.jit.load(vocoder_path, map_location=DEVICE).to(DEVICE).float()

        print(" - Loading ASR model (Content extraction)...")
        asr_model = torch.jit.load(asr_path, map_location=DEVICE).to(DEVICE)

        print(" - Loading Speaker Verification model (WavLM)...")
        if not os.path.exists(sv_path):
             raise FileNotFoundError(f"Speaker verification model not found at {sv_path}")
             
        sv_model = init_sv_model("wavlm_large", sv_path).to(DEVICE)
        sv_model.eval()

        print(" - Initializing Mel extractor...")
        mel_extractor = MelSpectrogramFeatures(
            sample_rate=16000,
            n_fft=1024,
            win_size=640,
            hop_length=160,
            n_mels=80,
            fmin=0,
            fmax=8000,
            center=True,
        ).to(DEVICE)

        _models = {
            "dit": dit_model,
            "vocos": vocos,
            "asr": asr_model,
            "sv": sv_model,
            "mel": mel_extractor,
            "vocoder_path": vocoder_path,
        }
        return f"模型加载成功！\n主模型: {os.path.basename(ckpt_path)}"
    except Exception as e:
        print(f"Error loading models: {e}")
        traceback.print_exc()
        return f"加载失败: {str(e)}"

def load_models_ui(ckpt_dropdown, asr_dropdown, vocoder_dropdown, sv_dropdown):
    return load_models(ckpt_dropdown, asr_dropdown, vocoder_dropdown, sv_dropdown)


def voice_conversion(
    source_audio_path,
    reference_audio_path,
    steps,
    chunk_size,
    use_slicer,
    auto_slice,
    auto_slice_min_sec,
    max_chunk_sec,
    slice_db_thresh,
    slice_min_len_ms,
    slice_win_l_ms,
    slice_win_s_ms,
    slice_max_sil_ms,
    slice_overlap_ms,
    vocos_cpu,
):
    if not _models:
        return None, "错误：请先在页面顶部加载模型！"
        
    if source_audio_path is None or reference_audio_path is None:
        return None, "Please provide both source and reference audio."

    try:
        source_wav, _ = librosa.load(source_audio_path, sr=16000)
        ref_wav, _ = librosa.load(reference_audio_path, sr=16000)
        _models["vocos_force_cpu"] = bool(vocos_cpu)
        spk_emb, prompt_mel = _extract_reference_features(
            ref_wav, _models["sv"], _models["mel"], DEVICE
        )

        should_slice = bool(use_slicer)
        if auto_slice:
            try:
                auto_slice_min_sec = float(auto_slice_min_sec)
            except Exception:
                auto_slice_min_sec = 0
            if auto_slice_min_sec and (len(source_wav) / 16000.0) < auto_slice_min_sec:
                should_slice = False
            else:
                should_slice = True

        if should_slice:
            chunks = _slice_audio_with_slicer(
                source_wav,
                16000,
                db_thresh=slice_db_thresh,
                min_len_ms=slice_min_len_ms,
                win_l_ms=slice_win_l_ms,
                win_s_ms=slice_win_s_ms,
                max_sil_ms=slice_max_sil_ms,
                max_chunk_sec=max_chunk_sec,
                overlap_ms=slice_overlap_ms,
            )
        else:
            max_samples = int(16000 * max_chunk_sec) if max_chunk_sec and max_chunk_sec > 0 else 0
            overlap_samples = int(16000 * (slice_overlap_ms / 1000.0)) if slice_overlap_ms else 0
            chunks = _split_long_audio(source_wav, max_samples, overlap=overlap_samples)
        if not chunks:
            return None, "错误：切片结果为空"

        converted_chunks = []
        for i, chunk in enumerate(chunks):
            if chunk is None or len(chunk) == 0:
                continue
            bn = _extract_bn_from_wave(chunk, _models["asr"], DEVICE)
            _, wav = inference(
                _models["dit"],
                _models["vocos"],
                bn,
                spk_emb,
                prompt_mel,
                chunk_size,
                steps,
                DEVICE,
            )
            wav_np = wav.squeeze().cpu().numpy()
            converted_chunks.append(wav_np)

        if not converted_chunks:
            return None, "错误：未生成任何有效音频片段"

        wav_out = np.concatenate(converted_chunks, axis=0)
        return (16000, wav_out), f"Success (chunks={len(converted_chunks)})"
    except Exception as e:
        traceback.print_exc()
        return None, str(e)


def preprocess_dataset(input_dir, output_dir, progress=gr.Progress()):
    """
    预处理数据集：提取Mel、BN、xvector特征
    """
    try:
        input_path = Path(input_dir)
        output_path = Path(output_dir)

        if not input_path.exists():
            return "错误：输入目录不存在"

        # 创建输出目录
        mel_dir = output_path / "mels"
        bn_dir = output_path / "bns"
        xvector_dir = output_path / "xvectors"

        mel_dir.mkdir(parents=True, exist_ok=True)
        bn_dir.mkdir(parents=True, exist_ok=True)
        xvector_dir.mkdir(parents=True, exist_ok=True)

        log_messages = []
        log_messages.append(f"开始预处理数据集...")
        log_messages.append(f"输入目录: {input_dir}")
        log_messages.append(f"输出目录: {output_dir}")

        # 获取所有音频文件
        audio_files = list(input_path.glob("*.wav")) + list(input_path.glob("*.mp3"))
        if not audio_files:
            return "错误：输入目录中没有找到音频文件 (.wav 或 .mp3)"

        log_messages.append(f"找到 {len(audio_files)} 个音频文件")

        # 步骤1：提取Mel频谱
        progress(0.1, desc="提取Mel频谱...")
        log_messages.append("\n步骤1/3: 提取Mel频谱")

        for i, audio_file in enumerate(audio_files):
            try:
                # 使用已定义的MelSpectrogramFeatures类
                mel_extractor = MelSpectrogramFeatures()
                wav, sr = librosa.load(str(audio_file), sr=16000)
                wav_tensor = torch.from_numpy(wav).unsqueeze(0)

                with torch.no_grad():
                    mel = mel_extractor(wav_tensor)
                    mel_np = mel.squeeze().cpu().numpy()

                output_file = mel_dir / f"{audio_file.stem}.npy"
                np.save(str(output_file), mel_np)

            except Exception as e:
                log_messages.append(f"  跳过 {audio_file.name}: {str(e)}")

            if i % 10 == 0:
                progress(
                    0.1 + 0.2 * (i / len(audio_files)),
                    desc=f"Mel提取 {i}/{len(audio_files)}",
                )

        log_messages.append(f"Mel频谱提取完成，保存到 {mel_dir}")

        # 步骤2：提取BN特征（需要ASR模型）
        progress(0.4, desc="提取BN特征...")
        log_messages.append("\n步骤2/3: 提取BN特征")

        mel_files = list(mel_dir.glob("*.npy"))
        log_messages.append(f"使用预训练ASR模型提取BN特征...")

        # 演示模式：只处理第一个文件
        if audio_files:
            try:
                audio_file = audio_files[0]
                log_messages.append(f"  正在提取BN特征...")
                # 调用预处理脚本
                cmd = [
                    sys.executable,
                    "src/preprocess/extract_bn_160ms.py",
                    "--input_dir",
                    str(input_path),
                    "--output_dir",
                    str(bn_dir),
                ]
                result = subprocess.run(
                    cmd, capture_output=True, text=True, cwd=PROJECT_ROOT
                )
                if result.returncode == 0:
                    log_messages.append(f"  BN特征已提取")
                else:
                    log_messages.append(f"  BN提取错误: {result.stderr}")
                    log_messages.append(f"  输出: {result.stdout}")
            except Exception as e:
                log_messages.append(f"  BN提取错误: {str(e)}")

        log_messages.append(f"BN特征提取完成，保存到 {bn_dir}")

        # 步骤3：提取声纹特征
        progress(0.7, desc="提取声纹特征...")
        log_messages.append("\n步骤3/3: 提取声纹特征")

        try:
            cmd = [
                sys.executable,
                "src/preprocess/extract_spk_emb_wavlm.py",
                "--input_dir",
                str(input_path),
                "--output_dir",
                str(xvector_dir),
                "--device",
                "cuda:0" if torch.cuda.is_available() else "cpu",
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=PROJECT_ROOT
            )
            if result.returncode == 0:
                log_messages.append(f"  声纹特征提取完成")
            else:
                log_messages.append(f"  错误: {result.stderr}")
                log_messages.append(f"  输出: {result.stdout}")
        except Exception as e:
            log_messages.append(f"  声纹提取错误: {str(e)}")

        progress(1.0, desc="预处理完成")

        # 生成数据列表
        log_messages.append("\n生成训练数据列表...")
        train_list = output_path / "train.list"
        with open(train_list, "w") as f:
            for audio_file in audio_files[:10]:  # 演示：只使用前10个
                utt_id = audio_file.stem
                bn_path = bn_dir / f"{utt_id}.npy"
                mel_path = mel_dir / f"{utt_id}.npy"
                xvector_path = xvector_dir / f"{utt_id}.npy"
                prompt_mel_path = mel_path  # 使用自己的mel作为prompt

                if bn_path.exists() and mel_path.exists() and xvector_path.exists():
                    f.write(
                        f"{utt_id}|{bn_path}|{mel_path}|{xvector_path}|{prompt_mel_path}\n"
                    )

        log_messages.append(f"数据列表已保存到: {train_list}")
        log_messages.append(f"\n预处理完成！")
        log_messages.append(f"请检查输出目录: {output_path}")

        return "\n".join(log_messages)

    except Exception as e:
        return f"预处理错误: {str(e)}"


# --- Training Logic ---

training_process = None
training_stop_event = threading.Event()

def find_free_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def stop_training():
    global training_process
    if training_process:
        print("Stopping training process...")
        training_stop_event.set()
        # Windows specific kill
        if os.name == 'nt':
            subprocess.call(['taskkill', '/F', '/T', '/PID', str(training_process.pid)])
        else:
            training_process.terminate()
        training_process = None
        return "训练已停止"
    return "没有正在运行的训练进程"

def _enqueue_training_output(pipe, output_queue):
    """Read bytes from pipe and push parsed lines/progress to queue."""
    buffer = []
    while True:
        ch = pipe.read(1)
        if not ch:
            break
        if ch == b"\r":
            line = "".join(buffer)
            output_queue.put(("progress", line))
            buffer = []
        elif ch == b"\n":
            line = "".join(buffer)
            output_queue.put(("line", line))
            buffer = []
        else:
            try:
                buffer.append(ch.decode("utf-8", errors="ignore"))
            except Exception:
                continue
    if buffer:
        output_queue.put(("line", "".join(buffer)))
    output_queue.put(("eof", ""))

def train_model(
    dataset_path,
    exp_name,
    batch_size,
    learning_rate,
    epochs,
    save_every,
    gpu_ids,
    num_workers,
    max_len,
    flow_ratio,
    cfg_ratio,
    cfg_scale,
    p,
    steps,
    cfg_strength,
    chunk_size,
    result_dir,
    reset_lr,
    resumable_with_seed,
    grad_accumulation_steps,
    grad_ckpt,
    feature_list,
    additional_feature_list,
    feature_pad_values,
):
    global training_process
    
    if training_process is not None:
        yield "训练已经运行中，请先停止当前训练。"
        return

    training_stop_event.clear()
    
    dataset_path = Path(dataset_path)
    if not (dataset_path / "train.list").exists():
        yield f"错误：在 {dataset_path} 下未找到 train.list。请先运行数据预处理。"
        return

    # Windows safe default: 0 workers to avoid spawn/pickle issues
    if os.name == 'nt' and num_workers > 0:
        print("Warning: On Windows, num_workers > 0 might cause issues. If it fails, try setting it to 0.")

    port = find_free_port()
    gpu_ids_list = [x.strip() for x in gpu_ids.split(',') if x.strip()]
    num_gpus = len(gpu_ids_list)
    save_dir = str(Path(PROJECT_ROOT) / "ckpts" / exp_name)
    
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{env.get('PYTHONPATH', '')}{os.pathsep}{PROJECT_ROOT}"
    # Ensure training logs (including tqdm) flush promptly when piped
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    # Construct command
    if num_gpus <= 1:
        # Single-GPU or CPU on Windows: Run directly with python to avoid accelerate's libuv/distrib issues
        cmd = [sys.executable, "-u", "src/train/train.py"]
        if num_gpus == 1:
            env["CUDA_VISIBLE_DEVICES"] = gpu_ids_list[0]
    else:
        # Multi-GPU: Still use accelerate launch
        cmd = [
            "accelerate", "launch",
            "--config-file", "default_config.yaml",
            "--main_process_port", str(port),
            "--num_processes", str(num_gpus),
            "--gpu_ids", gpu_ids
        ]
        cmd.append("src/train/train.py")
    
    # Add shared arguments
    cmd.extend([
        "--model-config", "src/config/config_160ms.json",
        "--batch-size", str(int(batch_size)),
        "--max-len", str(int(max_len)),
        "--flow-ratio", str(float(flow_ratio)),
        "--cfg-ratio", str(float(cfg_ratio)),
        "--cfg-scale", str(float(cfg_scale)),
        "--p", str(float(p)),
        "--num-workers", str(int(num_workers)),
        "--feature-list", str(feature_list).strip(),
        "--additional-feature-list", str(additional_feature_list).strip(),
        "--feature-pad-values", str(feature_pad_values).strip(),
        "--steps", str(int(steps)),
        "--cfg-strength", str(float(cfg_strength)),
        "--chunk-size", str(int(chunk_size)),
        "--result-dir", str(result_dir).strip(),
        "--save-per-updates", str(int(save_every)),
        "--reset-lr", str(int(reset_lr)),
        "--epochs", str(int(epochs)),
        "--learning-rate", str(learning_rate),
        "--resumable-with-seed", str(int(resumable_with_seed)),
        "--grad-accumulation-steps", str(int(grad_accumulation_steps)),
        "--grad-ckpt", str(int(grad_ckpt)),
        "--exp-name", exp_name,
        "--dataset-path", str(dataset_path)
    ])

    print(f"Executing command: {' '.join(cmd)}")
    
    # On Windows, we need to handle path separators carefully or let shell=False do it
    try:
        training_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=PROJECT_ROOT,
            env=env,
            bufsize=0,  # Unbuffered binary output
        )
    except Exception as e:
        yield f"启动训练失败: {str(e)}\n请确保已安装 accelerate: pip install accelerate"
        return

    log_buffer = [
        f"启动训练...\n命令: {' '.join(cmd)}\nPID: {training_process.pid}\n保存路径: {save_dir}\n\n"
    ]
    yield "".join(log_buffer)

    # Read output in a background thread and stream to UI
    import collections
    deque_buffer = collections.deque(maxlen=50)  # Keep last 50 lines to avoid lag
    output_queue = Queue()
    reader_thread = threading.Thread(
        target=_enqueue_training_output,
        args=(training_process.stdout, output_queue),
        daemon=True,
    )
    reader_thread.start()

    while True:
        if training_stop_event.is_set():
            deque_buffer.append("[用户请求停止训练]")
            yield "\n".join(deque_buffer)
            break

        try:
            kind, payload = output_queue.get(timeout=0.1)
        except Empty:
            if training_process.poll() is not None and output_queue.empty():
                break
            continue

        if kind == "progress":
            if payload.strip():
                if deque_buffer and (
                    "%" in deque_buffer[-1]
                    or "it/s" in deque_buffer[-1]
                    or "Epoch" in deque_buffer[-1]
                ):
                    deque_buffer[-1] = payload
                else:
                    deque_buffer.append(payload)
            yield "\n".join(deque_buffer)
        elif kind == "line":
            deque_buffer.append(payload)
            yield "\n".join(deque_buffer)
        elif kind == "eof":
            break
    rc = training_process.poll()
    if rc is None:
        # Ensure we have the final exit code after stdout closes
        rc = training_process.wait()
    training_process = None
    if rc == 0:
        deque_buffer.append(f"\n训练完成！\n保存路径: {save_dir}")
    elif not training_stop_event.is_set():
        deque_buffer.append(f"\n训练异常退出，退出代码: {rc}")
        
    yield "\n".join(deque_buffer)


# --- Debug Diagnostics ---

def run_diagnostics():
    lines = []
    try:
        lines.append("== 基础信息 ==")
        lines.append(f"Python: {sys.version.split()[0]}")
        lines.append(f"Executable: {sys.executable}")
        lines.append(f"OS: {platform.platform()}")
        lines.append("")

        lines.append("== Torch/CUDA ==")
        lines.append(f"torch: {torch.__version__}")
        lines.append(f"cuda available: {torch.cuda.is_available()}")
        lines.append(f"cuda version: {torch.version.cuda}")
        lines.append(f"cudnn: {torch.backends.cudnn.version()}")
        lines.append(f"device count: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            try:
                idx = torch.cuda.current_device()
                lines.append(f"current device: cuda:{idx}")
                lines.append(f"device name: {torch.cuda.get_device_name(idx)}")
            except Exception as e:
                lines.append(f"device info error: {e}")
        lines.append(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
        lines.append("")

        lines.append("== 显存占用 ==")
        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
            mem_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            lines.append(f"allocated: {mem_alloc:.2f} MB")
            lines.append(f"reserved: {mem_reserved:.2f} MB")
        else:
            lines.append("CUDA 不可用，跳过显存统计")
        lines.append("")

        lines.append("== 模型设备 ==")
        if _models:
            for k, m in _models.items():
                try:
                    param = next(m.parameters())
                    lines.append(f"{k}: device={param.device}, dtype={param.dtype}")
                except StopIteration:
                    lines.append(f"{k}: no parameters")
                except Exception as e:
                    lines.append(f"{k}: error {e}")
        else:
            lines.append("未加载模型")
        lines.append("")

        lines.append("== CUDA 小测试 ==")
        if torch.cuda.is_available():
            try:
                before = torch.cuda.memory_allocated()
                a = torch.randn(512, 512, device="cuda")
                b = torch.randn(512, 512, device="cuda")
                c = a @ b
                torch.cuda.synchronize()
                after = torch.cuda.memory_allocated()
                delta = (after - before) / (1024 ** 2)
                lines.append(f"matmul ok, delta allocated: {delta:.2f} MB")
                del a, b, c
                torch.cuda.empty_cache()
            except Exception as e:
                lines.append(f"cuda test failed: {e}")
        else:
            lines.append("CUDA 不可用，跳过测试")
        lines.append("")
    except Exception as e:
        lines.append(f"诊断异常: {e}")

    return "\n".join(lines)


# --- Gradio UI ---

with gr.Blocks(title="MeanVC Demo & Training", css=CUSTOM_CSS) as demo:
    gr.Markdown("# MeanVC: Lightweight and Streaming Zero-Shot Voice Conversion")
    gr.Markdown("语音转换演示与训练工具")

    with gr.Tabs():
        # Tab 1: 语音转换
        with gr.TabItem("语音转换"):
            gr.Markdown("### 实时语音转换控制台")

            # --- Model Management UI ---
            with gr.Accordion("模型加载管理 (Model Management)", open=True):
                with gr.Row():
                    refresh_btn = gr.Button(
                        "刷新模型列表",
                        size="sm",
                        variant="secondary",
                        elem_id="refresh-models-btn",
                        scale=0,
                        min_width=140,
                    )
                    model_status = gr.Textbox(
                        label="加载状态",
                        value="未加载",
                        interactive=False,
                        scale=1,
                    )
                
                with gr.Row():
                    ckpt_dropdown = gr.Dropdown(
                        label="主模型 (DiT)", 
                        choices=get_model_list(), 
                        value=CKPT_PATH if os.path.exists(CKPT_PATH) else None,
                        interactive=True,
                        allow_custom_value=True
                    )
                    asr_dropdown = gr.Dropdown(
                        label="ASR模型 (Content)", 
                        choices=get_model_list(), 
                        value=ASR_CKPT_PATH if os.path.exists(ASR_CKPT_PATH) else None,
                        interactive=True,
                        allow_custom_value=True
                    )
                
                with gr.Row():
                    vocoder_dropdown = gr.Dropdown(
                        label="声码器 (Vocoder)", 
                        choices=get_model_list(), 
                        value=VOCODER_CKPT_PATH if os.path.exists(VOCODER_CKPT_PATH) else None,
                        interactive=True,
                        allow_custom_value=True
                    )
                    sv_dropdown = gr.Dropdown(
                        label="声纹模型 (Speaker)", 
                        choices=get_model_list(), 
                        value=SV_CKPT_PATH if os.path.exists(SV_CKPT_PATH) else None,
                        interactive=True,
                        allow_custom_value=True
                    )

                with gr.Row():
                    load_btn = gr.Button("加载模型", variant="primary")
                    unload_btn = gr.Button("卸载模型", variant="stop")

                def refresh_choices():
                    full_list = get_model_list()
                    return (
                        gr.Dropdown(choices=full_list), 
                        gr.Dropdown(choices=full_list), 
                        gr.Dropdown(choices=full_list), 
                        gr.Dropdown(choices=full_list)
                    )

                refresh_btn.click(
                    fn=refresh_choices,
                    inputs=[],
                    outputs=[ckpt_dropdown, asr_dropdown, vocoder_dropdown, sv_dropdown]
                )
                
                load_btn.click(
                    fn=load_models_ui,
                    inputs=[ckpt_dropdown, asr_dropdown, vocoder_dropdown, sv_dropdown],
                    outputs=[model_status]
                )
                
                unload_btn.click(
                    fn=unload_models,
                    inputs=[],
                    outputs=[model_status]
                )

            gr.Markdown("### 推理参数设置")
            with gr.Row():
                with gr.Column():
                    source_audio = gr.Audio(
                        type="filepath", label="源音频（要转换的声音）"
                    )
                    ref_audio = gr.Audio(type="filepath", label="参考音频（目标音色）")

                    with gr.Accordion("高级参数", open=False):
                        steps_slider = gr.Slider(
                            minimum=1, maximum=10, value=2, step=1, label="降噪步数"
                        )
                        chunk_size_slider = gr.Slider(
                            minimum=1, maximum=30, value=20, step=1, label="块大小"
                        )
                        use_slicer_checkbox = gr.Checkbox(
                            label="启用自动切片 (Audio Slicer)", value=True
                        )
                        auto_slice_checkbox = gr.Checkbox(
                            label="根据音频时长自动决定是否切片", value=True
                        )
                        auto_slice_min_sec = gr.Number(
                            label="自动切片阈值 (秒)", value=12, precision=0
                        )
                        max_chunk_sec = gr.Number(
                            label="最大切片时长 (秒)", value=15, precision=0
                        )
                        slice_db_thresh = gr.Number(
                            label="切片静音阈值 (dB)", value=-40
                        )
                        slice_min_len_ms = gr.Number(
                            label="切片最小时长 (ms)", value=3000, precision=0
                        )
                        slice_win_l_ms = gr.Number(
                            label="切片长窗 (ms)", value=300, precision=0
                        )
                        slice_win_s_ms = gr.Number(
                            label="切片短窗 (ms)", value=20, precision=0
                        )
                        slice_max_sil_ms = gr.Number(
                            label="切片最大静音保留 (ms)", value=500, precision=0
                        )
                        slice_overlap_ms = gr.Number(
                            label="切片重叠 (ms)", value=0, precision=0
                        )
                        vocos_cpu_checkbox = gr.Checkbox(
                            label="Vocos 强制 CPU 解码", value=False
                        )

                    submit_btn = gr.Button("开始转换", variant="primary")

                with gr.Column():
                    output_audio = gr.Audio(label="转换后的音频")
                    status_msg = gr.Textbox(label="转换日志", interactive=False)

            submit_btn.click(
                fn=voice_conversion,
                inputs=[
                    source_audio,
                    ref_audio,
                    steps_slider,
                    chunk_size_slider,
                    use_slicer_checkbox,
                    auto_slice_checkbox,
                    auto_slice_min_sec,
                    max_chunk_sec,
                    slice_db_thresh,
                    slice_min_len_ms,
                    slice_win_l_ms,
                    slice_win_s_ms,
                    slice_max_sil_ms,
                    slice_overlap_ms,
                    vocos_cpu_checkbox,
                ],
                outputs=[output_audio, status_msg],
            )

            gr.Examples(
                examples=[
                    [
                        "src/runtime/example/test.wav",
                        "src/runtime/example/test.wav",
                        2,
                        20,
                        True,
                        True,
                        12,
                        15,
                        -40,
                        3000,
                        300,
                        20,
                        500,
                        0,
                        False,
                    ],
                ],
                inputs=[
                    source_audio,
                    ref_audio,
                    steps_slider,
                    chunk_size_slider,
                    use_slicer_checkbox,
                    auto_slice_checkbox,
                    auto_slice_min_sec,
                    max_chunk_sec,
                    slice_db_thresh,
                    slice_min_len_ms,
                    slice_win_l_ms,
                    slice_win_s_ms,
                    slice_max_sil_ms,
                    slice_overlap_ms,
                    vocos_cpu_checkbox,
                ],
            )

        # Tab 2: 数据预处理
        with gr.TabItem("数据预处理"):
            gr.Markdown("### 准备训练数据集")
            gr.Markdown("""
            此功能将自动：
            1. 提取Mel频谱（10ms帧移）
            2. 提取内容特征BN（160ms窗口）
            3. 提取声纹特征（xvector）
            4. 生成训练数据列表
            """)

            with gr.Row():
                with gr.Column():
                    input_dir = gr.Textbox(
                        label="输入目录",
                        placeholder="包含.wav音频文件的目录路径",
                        value="path/to/your/audio/files",
                    )
                    output_dir = gr.Textbox(
                        label="输出目录",
                        placeholder="预处理后数据保存路径",
                        value="path/to/output/features",
                    )

                    preprocess_btn = gr.Button("开始预处理", variant="primary")

                with gr.Column():
                    preprocess_output = gr.Textbox(
                        label="处理日志", lines=20, interactive=False
                    )

            preprocess_btn.click(
                fn=preprocess_dataset,
                inputs=[input_dir, output_dir],
                outputs=preprocess_output,
            )

        # Tab 3: 模型训练
        with gr.TabItem("模型训练"):
            gr.Markdown("### 训练 MeanVC 模型")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### 训练设置")
                    train_dataset_path = gr.Textbox(
                        label="数据目录 (含 train.list)",
                        placeholder="对应预处理的输出目录",
                        value="path/to/output/features"
                    )
                    train_exp_name = gr.Textbox(
                        label="实验名称",
                        value="my_meanvc_train",
                        info="模型检查点将保存在 项目目录/ckpts/<实验名称> 下"
                    )
                    
                    with gr.Row():
                        train_batch_size = gr.Number(label="Batch Size", value=16, precision=0)
                        train_lr = gr.Number(label="Learning Rate", value=1e-4)
                    
                    with gr.Row():
                        train_epochs = gr.Number(label="Epochs", value=1000, precision=0)
                        train_save_every = gr.Number(label="Save Every (Steps)", value=10000, precision=0)
                    
                    with gr.Row():
                        train_gpu = gr.Textbox(label="GPU IDs (例如: 0 或 0,1)", value="0")
                        train_workers = gr.Number(label="Num Workers (Windows建议0)", value=0, precision=0)

                    with gr.Accordion("高级训练参数", open=False):
                        train_max_len = gr.Number(label="Max Len", value=1000, precision=0)
                        train_flow_ratio = gr.Number(label="Flow Ratio", value=0.50)
                        train_cfg_ratio = gr.Number(label="CFG Ratio", value=0.1)
                        train_cfg_scale = gr.Number(label="CFG Scale", value=2.0)
                        train_p = gr.Number(label="p", value=0.5)
                        train_steps = gr.Number(label="Steps", value=1, precision=0)
                        train_cfg_strength = gr.Number(label="CFG Strength", value=2.0)
                        train_chunk_size = gr.Number(label="Chunk Size", value=16, precision=0)
                        train_result_dir = gr.Textbox(label="Result Dir", value="results")
                        train_reset_lr = gr.Number(label="Reset LR (0/1)", value=0, precision=0)
                        train_resumable_seed = gr.Number(label="Resumable Seed", value=666, precision=0)
                        train_grad_accum = gr.Number(label="Grad Accum Steps", value=1, precision=0)
                        train_grad_ckpt = gr.Number(label="Grad CKPT (0/1)", value=0, precision=0)
                        train_feature_list = gr.Textbox(
                            label="Feature List",
                            value="bn mel xvector",
                        )
                        train_additional_feature_list = gr.Textbox(
                            label="Additional Feature List",
                            value="inputs_length prompt",
                        )
                        train_feature_pad_values = gr.Textbox(
                            label="Feature Pad Values",
                            value="0. -1.0 0.",
                        )

                    with gr.Row():
                        start_train_btn = gr.Button("开始训练", variant="primary")
                        stop_train_btn = gr.Button("停止训练", variant="stop")
                
                with gr.Column(scale=2):
                    gr.Markdown("#### 训练日志")
                    train_log_output = gr.Textbox(
                        label="实时日志", 
                        lines=30, 
                        interactive=False,
                        max_lines=30,
                        autoscroll=True
                    )
            
            start_train_btn.click(
                fn=train_model,
                inputs=[
                    train_dataset_path, 
                    train_exp_name, 
                    train_batch_size, 
                    train_lr, 
                    train_epochs, 
                    train_save_every,
                    train_gpu,
                    train_workers,
                    train_max_len,
                    train_flow_ratio,
                    train_cfg_ratio,
                    train_cfg_scale,
                    train_p,
                    train_steps,
                    train_cfg_strength,
                    train_chunk_size,
                    train_result_dir,
                    train_reset_lr,
                    train_resumable_seed,
                    train_grad_accum,
                    train_grad_ckpt,
                    train_feature_list,
                    train_additional_feature_list,
                    train_feature_pad_values,
                ],
                outputs=train_log_output
            )
            
            stop_train_btn.click(
                fn=stop_training,
                inputs=[],
                outputs=train_log_output
            )

        # Tab 4: Debug
        with gr.TabItem("Debug"):
            gr.Markdown("### 诊断信息")
            debug_btn = gr.Button("运行诊断", variant="primary")
            debug_output = gr.Textbox(label="诊断输出", lines=25, interactive=False)
            debug_btn.click(
                fn=run_diagnostics,
                inputs=[],
                outputs=debug_output
            )

if __name__ == "__main__":
    print("=" * 50)
    print("Web界面: http://127.0.0.1:7860")
    print("API文档: http://127.0.0.1:7860/?view=api")
    print("=" * 50)
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
