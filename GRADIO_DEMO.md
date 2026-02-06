# Gradio Demo 使用说明

本文档说明如何使用 `gradio_demo.py` 的 Web UI 进行模型加载、语音转换、数据预处理、训练与调试。

## 1. 启动

在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe gradio_demo.py
```

启动后访问：

- Web UI: `http://127.0.0.1:7860`
- API 文档: `http://127.0.0.1:7860/?view=api`

## 2. 模型加载

在「语音转换」页顶部的「模型加载管理」中：

1. 点击「刷新模型列表」
2. 在下拉框中选择主模型 / ASR / Vocoder / Speaker
3. 点击「加载模型」

模型来源会自动扫描以下目录：

- `src/ckpt/`
- `src/runtime/ckpt/`
- `src/runtime/speaker_verification/ckpt/`
- `ckpts/`
- `src/ckpts/`
- `results/`

## 3. 语音转换

在「语音转换」页：

1. 选择「源音频」
2. 选择「参考音频」
3. 点击「开始转换」

### 推理高级参数

「高级参数」里可调：

- `steps`：降噪步数
- `chunk_size`：推理块大小
- 切片相关（Audio Slicer）：
  - 启用自动切片
  - 自动切片阈值（秒）
  - 最大切片时长（秒）
  - 静音阈值 / 切片窗口 / 最小片段 / 最大静音 / 重叠
- `Vocos 强制 CPU 解码`（若 GPU 端 complex 报错）

## 4. 数据预处理

在「数据预处理」页：

1. 输入音频目录（含 `.wav` / `.mp3`）
2. 输出目录（将生成 `mels/ bns/ xvectors/ train.list`）
3. 点击「开始预处理」

## 5. 训练

在「模型训练」页：

- 必填：数据目录、实验名
- 可选：学习率、batch size、epochs 等
- 高级参数中可以调节训练脚本全部常用参数

训练产物默认保存到：

```
ckpts/<实验名>/model_last.pt
```

## 6. Debug

在「Debug」页点击「运行诊断」可查看：

- Python / Torch 版本
- CUDA 可用性与设备信息
- 显存占用
- 已加载模型的 device/dtype
- CUDA 小测试结果

## 7. 常见问题

**Q: 模型加载了但显存不动？**  
A: Debug 页确认模型是否已加载到 `cuda`。若未加载请检查启动用的 python 环境和依赖版本。

**Q: 转换时报 `UNSUPPORTED DTYPE: complex`？**  
A: 勾选「Vocos 强制 CPU 解码」或等待自动回退。

**Q: 长音频报错或效果差？**  
A: 使用自动切片，或适当增大「最大切片时长」，减少切片次数。

## 8. 致谢

自动切片功能引用了以下项目：

```
https://github.com/flutydeer/audio-slicer
```
