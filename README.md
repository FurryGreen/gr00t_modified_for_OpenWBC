# Isaac-GR00T for OpenWBC 使用说明

本项目基于 NVIDIA 的 GR00T-N1.5-3B 模型，添加了WBC的data config ([`OpenWBCDataConfig`](./gr00t/experiment/data_config.py))，以兼容G1移动操作。另外对[`eval_policy`](./scripts/eval_policy.py)进行了部分可视化结果优化。

原文档说明请见 [`./README_raw_gr00t.md`](./README_raw_gr00t.md)

---

## 🔧 模型安装

可基于公开镜像：ngc-cuda124-g1:2.0。主要加装了tmux，卸载了opencv-python, 方便训练。
```bash
cd Isaac-GR00T

# 注意修改 pyproject.toml，取消 opencv-python 的依赖。本仓库应该已经做过了

# 安装依赖
pip install -e .
```
**Note** 运行程序如遇以下报错：
```bash
AttributeError: module 'cv2' has no attribute 'CV_8U'
```
请卸载并重新安装无头版 opencv：
```bash
pip uninstall opencv-python-headless
pip install opencv-python-headless
```

---

## 📁 数据准备

### 1. 下载模型

下载模型到 Isaac-GR00T 同目录下
```bash
huggingface-cli download --resume-download nvidia/GR00T-N1.5-3B --local-dir ../models/GR00T-N1.5-3B/
```

### 2. (可跳过) 下载 Huggingface 测试数据集（可选：用于可乐测试）

```bash
# 登录 Huggingface
huggingface-cli login

huggingface-cli download --repo-type dataset --resume-download JimmyPeng02/pick_cola_gr00t4 \
  --cache-dir ../datasets/test/ --local-dir-use-symlinks False
```

### 3. 本地数据集准备（抓娃娃数据）

已经有处理好的数据，可通过以下方式建立软连接到 Isaac-GR00T 同目录：
```bash
ln -s /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/wuyuxuan-240108110054/EAhw/datasets/multiobj_pick ../datasets/
```

---

## 🧠 模型微调

### 测试指令

这一部分以及各种参数含义详见[`./README_raw_gr00t.md`](./README_raw_gr00t.md)
```bash
python scripts/gr00t_finetune.py --help

python scripts/gr00t_finetune.py \
  --base-model-path ../models/GR00T-N1.5-3B \
  --dataset-path ./demo_data/robot_sim.PickNPlace \
  --num-gpus 1 \
  --lora_rank 64 \
  --lora_alpha 128 \
  --batch-size 32
```

### 单任务全模型微调（结果保存到 `./save/`，可能需要提前创建save文件夹X）

```bash
python scripts/gr00t_finetune.py \
  --output-dir ./save/pick_mickey_mouse \
  --base-model-path ../models/GR00T-N1.5-3B \
  --dataset-path ../datasets/multiobj_pick/pick_mickey_mouse \
  --num-gpus 1 \
  --batch-size 64 \
  --report-to tensorboard \
  --max_steps 20000 \
  --data-config openwbc_g1

python scripts/gr00t_finetune.py \
  --output-dir ./save/pick_pink_fox \
  --base-model-path ../models/GR00T-N1.5-3B \
  --dataset-path ../datasets/multiobj_pick/pick_pink_fox \
  --num-gpus 1 \
  --batch-size 64 \
  --report-to tensorboard \
  --max_steps 20000 \
  --data-config openwbc_g1

python scripts/gr00t_finetune.py \
  --output-dir ./save/pick_toy_cat \
  --base-model-path ../models/GR00T-N1.5-3B \
  --dataset-path ../datasets/multiobj_pick/pick_toy_cat \
  --num-gpus 1 \
  --batch-size 64 \
  --report-to tensorboard \
  --max_steps 20000 \
  --data-config openwbc_g1

python scripts/gr00t_finetune.py \
  --output-dir ./save/pick_toy_sloth \
  --base-model-path ../models/GR00T-N1.5-3B \
  --dataset-path ../datasets/multiobj_pick/pick_toy_sloth \
  --num-gpus 1 \
  --batch-size 64 \
  --report-to tensorboard \
  --max_steps 20000 \
  --data-config openwbc_g1
```

### 多任务微调

直接运行以下命令：
```bash
dataset_list=(
  "../datasets/multiobj_pick/pick_mickey_mouse"
  "../datasets/multiobj_pick/pick_pink_fox"
  "../datasets/multiobj_pick/pick_toy_cat"
  "../datasets/multiobj_pick/pick_toy_sloth"
)

python scripts/gr00t_finetune.py \
  --base-model-path ../models/GR00T-N1.5-3B \
  --dataset-path ${dataset_list[@]} \
  --num-gpus 1 \
  --output-dir ./save/multiobj_pick_WBC/ \
  --data-config openwbc_g1 \
  --embodiment-tag new_embodiment \
  --batch-size 128 \
  --max_steps 20000 \
  --report-to tensorboard
```

---

## 📊 数据集评估

```bash
python scripts/eval_policy.py \
  --plot \
  --model_path ./save/test/checkpoint-1000 \
  --dataset-path ../datasets/multiobj_pick/pick_mickey_mouse \
  --embodiment-tag new_embodiment \
  --data-config openwbc_g1 \
  --modality-keys base_motion left_hand right_hand
```

评估后生成图像 `test_fig.png`（位于 Isaac-GR00T 根目录），可用于观察模型预测结果与 GT 曲线的对比。

---

## 🤖 实机推理（测试中，还不知道咋用）

```bash
# 启动服务端
python scripts/inference_service.py --model_path ../models/GR00T-N1.5-3B --server

# 启动客户端
python scripts/inference_service.py --client
```

