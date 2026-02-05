import os

# --- 强制重定向所有缓存到项目所在盘符 (防止填满 C 盘) ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(PROJECT_ROOT, "temp_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# 设置环境变量，必须在导入其他库之前
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
from queue import Queue
import signal
import traceback

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

    ref_wav, _ = librosa.load(reference_path, sr=16000)
    ref_wav_tensor = torch.from_numpy(ref_wav).unsqueeze(0).to(device)

    with torch.no_grad():
        spk_emb = sv_model(ref_wav_tensor)
        prompt_mel = mel_extractor(ref_wav_tensor)
        prompt_mel = prompt_mel.transpose(1, 2)

    return bn, spk_emb, prompt_mel


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

            u, tmp_kv_cache = model(
                x,
                t_tensor,
                r_tensor,
                bn_chunk,
                spk_emb,
                prompt_mel,
                None,  # cache
                start,  # offset = current position
                kv_cache,
            )
            x = x - (t - r) * u
            kv_cache = tmp_kv_cache

        x_pred.append(x)

        if start > 40 and kv_cache is not None:
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
    y_g_hat = vocos.decode(mel)

    return mel, y_g_hat


# --- Gradio App Logic ---

_models = {}


def load_models():
    global _models
    if _models:
        return _models

    if not os.path.exists(SV_CKPT_PATH):
        raise FileNotFoundError(
            f"Speaker verification model not found at {SV_CKPT_PATH}. Please download it manually from the link in README.md."
        )

    print(f"Loading models to {DEVICE}...")

    with open(MODEL_CONFIG_PATH) as f:
        model_config = json.load(f)

    print(" - Loading DiT model...")
    dit_model = torch.jit.load(CKPT_PATH, map_location=DEVICE).to(DEVICE)

    print(" - Loading Vocos (Vocoder)...")
    vocos = torch.jit.load(VOCODER_CKPT_PATH, map_location=DEVICE).to(DEVICE)

    print(" - Loading ASR model (Content extraction)...")
    asr_model = torch.jit.load(ASR_CKPT_PATH, map_location=DEVICE).to(DEVICE)

    print(" - Loading Speaker Verification model (WavLM)...")
    sv_model = init_sv_model("wavlm_large", SV_CKPT_PATH).to(DEVICE)
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
    }
    return _models


def voice_conversion(source_audio_path, reference_audio_path, steps, chunk_size):
    if source_audio_path is None or reference_audio_path is None:
        return None, "Please provide both source and reference audio."

    try:
        models = load_models()
        bn, spk_emb, prompt_mel = extract_features_from_audio(
            source_audio_path,
            reference_audio_path,
            models["asr"],
            models["sv"],
            models["mel"],
            DEVICE,
        )
        _, wav = inference(
            models["dit"],
            models["vocos"],
            bn,
            spk_emb,
            prompt_mel,
            chunk_size,
            steps,
            DEVICE,
        )
        wav_np = wav.squeeze().cpu().numpy()
        return (16000, wav_np), "Success"
    except Exception as e:
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


# --- Gradio UI ---

with gr.Blocks(title="MeanVC Demo & Training") as demo:
    gr.Markdown("# MeanVC: Lightweight and Streaming Zero-Shot Voice Conversion")
    gr.Markdown("语音转换演示与训练工具")

    with gr.Tabs():
        # Tab 1: 语音转换
        with gr.TabItem("语音转换"):
            gr.Markdown("### 将源音频的声音转换为参考音频的音色")

            with gr.Row():
                with gr.Column():
                    source_audio = gr.Audio(
                        type="filepath", label="源音频（要转换的声音）"
                    )
                    ref_audio = gr.Audio(type="filepath", label="参考音频（目标音色）")

                    with gr.Accordion("高级设置", open=False):
                        steps_slider = gr.Slider(
                            minimum=1, maximum=10, value=2, step=1, label="降噪步数"
                        )
                        chunk_size_slider = gr.Slider(
                            minimum=1, maximum=30, value=20, step=1, label="块大小"
                        )

                    submit_btn = gr.Button("开始转换", variant="primary")

                with gr.Column():
                    output_audio = gr.Audio(label="转换后的音频")
                    status_msg = gr.Textbox(label="状态", interactive=False)

            submit_btn.click(
                fn=voice_conversion,
                inputs=[source_audio, ref_audio, steps_slider, chunk_size_slider],
                outputs=[output_audio, status_msg],
            )

            gr.Examples(
                examples=[
                    [
                        "src/runtime/example/test.wav",
                        "src/runtime/example/test.wav",
                        2,
                        20,
                    ],
                ],
                inputs=[source_audio, ref_audio, steps_slider, chunk_size_slider],
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

if __name__ == "__main__":
    print("Pre-loading models before launching UI...")
    load_models()
    print("Success: All models loaded. Launching UI...")
    print("=" * 50)
    print("Web界面: http://127.0.0.1:7860")
    print("API文档: http://127.0.0.1:7860/?view=api")
    print("=" * 50)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
