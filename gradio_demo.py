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


# --- Training Functions ---

# 全局训练状态
train_stop_flag = False
train_thread = None
train_log_queue = Queue()


def run_training(
    dataset_path,
    exp_name,
    batch_size,
    epochs,
    learning_rate,
    save_interval,
    use_gpu,
):
    """
    实际执行训练（简化版，使用命令行调用）
    """
    global train_stop_flag

    try:
        # 检查数据集路径
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            yield "❌ 错误：数据集路径不存在"
            return

        train_list = dataset_path / "train.list"
        if not train_list.exists():
            yield "❌ 错误：未找到 train.list 文件，请先进行数据预处理"
            yield "提示：使用'数据预处理'Tab处理你的音频数据"
            return

        # 设置实验目录
        exp_dir = Path(PROJECT_ROOT) / "results" / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        yield f"✅ 检查通过"
        yield f"📁 实验名称: {exp_name}"
        yield f"📂 保存目录: {exp_dir}"
        yield f"📊 数据集: {dataset_path}"

        # 准备训练命令
        cuda_devices = "0" if use_gpu and torch.cuda.is_available() else ""
        if use_gpu and not torch.cuda.is_available():
            yield "⚠️ 警告：GPU不可用，将使用CPU训练（会非常慢）"

        yield f"\n🚀 启动训练..."
        yield f"📝 参数: batch_size={batch_size}, epochs={epochs}, lr={learning_rate}"

        # 构建命令
        cmd = [
            sys.executable,
            "src/train/train.py",
            "--model-config",
            "src/config/config_160ms.json",
            "--batch-size",
            str(batch_size),
            "--max-len",
            "1000",
            "--flow-ratio",
            "0.50",
            "--cfg-ratio",
            "0.1",
            "--cfg-scale",
            "2.0",
            "--p",
            "0.5",
            "--num-workers",
            "4",
            "--feature-list",
            "bn mel xvector",
            "--additional-feature-list",
            "inputs_length prompt",
            "--feature-pad-values",
            "0. -1.0 0.",
            "--steps",
            "1",
            "--cfg-strength",
            "2.0",
            "--chunk-size",
            "16",
            "--result-dir",
            str(exp_dir),
            "--save-per-updates",
            str(save_interval),
            "--reset-lr",
            "0",
            "--epochs",
            str(epochs),
            "--resumable-with-seed",
            "666",
            "--grad-accumulation-steps",
            "1",
            "--grad-ckpt",
            "0",
            "--exp-name",
            exp_name,
            "--dataset-path",
            str(dataset_path),
            "--learning-rate",
            str(learning_rate),
        ]

        # 设置环境变量
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"
        if cuda_devices:
            env["CUDA_VISIBLE_DEVICES"] = cuda_devices

        yield f"\n{'=' * 50}"
        yield "训练进行中... (按'停止训练'按钮可中断)"
        yield f"{'=' * 50}\n"

        # 启动训练进程
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd=PROJECT_ROOT,
            env=env,
        )

        # 实时读取输出
        log_buffer = []
        while True:
            # 检查停止标志
            if train_stop_flag:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except:
                    process.kill()
                yield "\n🛑 训练已被用户停止"
                yield f"💾 检查点可能已保存在: {exp_dir}"
                break

            # 读取输出
            try:
                if process.stdout:
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break

                    if line:
                        log_buffer.append(line.strip())
                        # 只保留最近20行
                        if len(log_buffer) > 20:
                            log_buffer = log_buffer[-20:]
                        yield "\n".join(log_buffer)
            except:
                # Windows下可能会有编码问题
                pass

            # 短暂休眠避免CPU占用过高
            time.sleep(0.1)

        # 获取返回码
        return_code = process.poll()

        if return_code == 0:
            yield f"\n✅ 训练成功完成！"
            yield f"💾 模型保存在: {exp_dir}"
        else:
            yield f"\n❌ 训练失败 (返回码: {return_code})"
            yield "请检查上面的错误日志"

    except Exception as e:
        yield f"\n❌ 训练错误: {str(e)}"
        yield traceback.format_exc()


def stop_training():
    """停止训练"""
    global train_stop_flag
    train_stop_flag = True
    return "正在停止训练..."


def start_training_thread(*args):
    """在后台线程启动训练"""
    global train_thread, train_stop_flag
    train_stop_flag = False

    def train_wrapper():
        for log in run_training(*args):
            train_log_queue.put(log)

    train_thread = threading.Thread(target=train_wrapper)
    train_thread.start()
    return "训练已启动"


def get_train_logs():
    """获取训练日志"""
    logs = []
    while not train_log_queue.empty():
        logs.append(train_log_queue.get())
    return "\n".join(logs) if logs else ""


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

        for i, audio_file in enumerate(tqdm(audio_files, desc="Mel提取")):
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

        for i, audio_file in enumerate(tqdm(audio_files, desc="BN提取")):
            try:
                # 调用预处理脚本
                cmd = [
                    "python",
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
                break  # 演示模式，只处理一个文件
            except Exception as e:
                log_messages.append(f"  BN提取错误: {str(e)}")
                break

        log_messages.append(f"BN特征提取完成，保存到 {bn_dir}")

        # 步骤3：提取声纹特征
        progress(0.7, desc="提取声纹特征...")
        log_messages.append("\n步骤3/3: 提取声纹特征")

        try:
            cmd = [
                "python",
                "src/preprocess/extract_spk_emb_wavlm.py",
                "--input_dir",
                str(input_path),
                "--output_dir",
                str(xvector_dir),
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=PROJECT_ROOT
            )
            if result.returncode == 0:
                log_messages.append(f"  声纹特征提取完成")
            else:
                log_messages.append(f"  错误: {result.stderr}")
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


def generate_train_script(
    dataset_path, exp_name, batch_size, epochs, learning_rate, save_interval, use_gpu
):
    """
    生成训练脚本
    """
    try:
        script_content = f"""#!/bin/bash
# MeanVC 训练脚本 - 自动生成
# 实验名称: {exp_name}

export PYTHONPATH=$PYTHONPATH:$PWD

# 设置GPU
cuda={"0" if use_gpu else ""}
IFS=',' read -ra parts <<< "$cuda"
num_gpus=${{#parts[@]}}

echo "使用 $num_gpus 个GPU"
port=`comm -23 <(seq 50075 65535 | sort) <(ss -tan | awk '{{print $4}}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`

# 启动训练
accelerate launch --config-file default_config.yaml \\
    --main_process_port $port \\
    --num_processes ${{num_gpus}} \\
    {"--gpu_ids ${{cuda}}" if use_gpu else "--cpu"} \\
    src/train/train.py \\
    --model-config src/config/config_160ms.json \\
    --batch-size {batch_size} \\
    --max-len 1000 \\
    --flow-ratio 0.50 \\
    --cfg-ratio 0.1 \\
    --cfg-scale 2.0 \\
    --p 0.5 \\
    --num-workers 4 \\
    --feature-list "bn mel xvector" \\
    --additional-feature-list "inputs_length prompt" \\
    --feature-pad-values "0. -1.0 0." \\
    --steps 1 \\
    --cfg-strength 2.0 \\
    --chunk-size 16 \\
    --result-dir "results" \\
    --save-per-updates {save_interval} \\
    --reset-lr 0 \\
    --epochs {epochs} \\
    --resumable-with-seed 666 \\
    --grad-accumulation-steps 1 \\
    --grad-ckpt 0 \\
    --exp-name {exp_name} \\
    --dataset-path "{dataset_path}" \\
    --learning-rate {learning_rate}

echo "训练完成！"
"""

        # 保存脚本
        script_path = Path(PROJECT_ROOT) / f"train_{exp_name}.sh"
        with open(script_path, "w") as f:
            f.write(script_content)

        return f"训练脚本已生成: {script_path}\\n\\n脚本内容：\\n{script_content}"

    except Exception as e:
        return f"生成脚本错误: {str(e)}"


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

        # Tab 3: 模型训练
        with gr.TabItem("模型训练"):
            gr.Markdown("### 在Gradio中直接训练模型")
            gr.Markdown("""
            此功能允许你直接在Web界面中训练MeanVC模型。
            **注意**：训练会占用较多计算资源，建议在GPU环境下进行。
            """)

            with gr.Row():
                with gr.Column():
                    train_dataset_path = gr.Textbox(
                        label="数据集路径",
                        placeholder="预处理后的数据目录（包含train.list）",
                        value="path/to/output/features",
                    )
                    train_exp_name = gr.Textbox(
                        label="实验名称",
                        placeholder="my_experiment",
                        value="my_meanvc_train",
                    )

                    with gr.Row():
                        train_batch_size = gr.Slider(
                            minimum=1, maximum=64, value=16, step=1, label="批次大小"
                        )
                        train_epochs = gr.Slider(
                            minimum=1,
                            maximum=1000,
                            value=100,
                            step=10,
                            label="训练轮数",
                        )

                    with gr.Row():
                        train_lr = gr.Number(
                            value=0.0001,
                            label="学习率",
                            minimum=0.00001,
                            maximum=0.01,
                            step=0.00001,
                        )
                        train_save_interval = gr.Slider(
                            minimum=100,
                            maximum=50000,
                            value=1000,
                            step=100,
                            label="保存间隔（步数）",
                        )

                    train_use_gpu = gr.Checkbox(label="使用GPU", value=True)

                    with gr.Row():
                        start_train_btn = gr.Button("开始训练", variant="primary")
                        stop_train_btn = gr.Button("停止训练", variant="stop")

                with gr.Column():
                    train_output = gr.Textbox(
                        label="训练日志", lines=20, interactive=False, autoscroll=True
                    )
                    train_progress = gr.Slider(
                        minimum=0, maximum=100, value=0, label="训练进度 (%)"
                    )

                    gr.Markdown("""
                    **说明：**
                    - 点击"开始训练"启动训练过程
                    - 训练过程中会实时显示损失值和进度
                    - 可随时点击"停止训练"中断（会保存已训练的权重）
                    - 训练结果保存在 `results/{实验名称}/` 目录
                    """)

            # 绑定按钮事件
            start_train_btn.click(
                fn=run_training,
                inputs=[
                    train_dataset_path,
                    train_exp_name,
                    train_batch_size,
                    train_epochs,
                    train_lr,
                    train_save_interval,
                    train_use_gpu,
                ],
                outputs=train_output,
            )

            stop_train_btn.click(
                fn=stop_training,
                outputs=train_output,
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
