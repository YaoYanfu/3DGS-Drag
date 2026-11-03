# 抬臂实验：文件与成果导览

核对日期：2026-09-26。本文记录服务器上已完成的 `person_1_raisearm` 首次拖动实验（`person_raise_arm_server_02`）。路径如无特别说明，均相对于项目目录：

```text
/media/ubuntu/data/legiony_3dgsdrag/code/3DGS-Drag/
```

## 现在有几个高斯点云？

**已保存的完整 3DGS 模型有两个：初始模型和拖动优化后的最终模型。**

| 阶段 | 文件 | 高斯数量 | 作用 |
|---|---|---:|---|
| 拖动前 | `output/person_base/point_cloud/iteration_7000/point_cloud.ply` | 231,171 | 用 7,000 次基础训练得到的初始 3DGS，也是本次拖动的 `--ply` 输入 |
| 拖动后 | `output/person_raise_arm_server_02/person_1_raisearm/20260926-154932/point_cloud/iteration_100/point_cloud.ply` | 235,233 | 完成一次拖动、一次扩散编辑及 100 次高斯优化后的结果 |

两者相差 **4,062 个高斯**，与日志中的 `4062 points are selected for dragging` 一致：这次拖动复制了所选高斯，旧位置的高斯仍在模型中，但不透明度被降低。粗形变后、扩散编辑前的中间高斯状态只在运行内存中，**没有另存一个 PLY**。

各输出目录下还有约 10,879 个顶点、约 294 KB 的 `input.ply`。它是从数据集复制的 **COLMAP 稀疏点云**，用于场景初始化和记录，不能当作上表中的训练好 3DGS。失败的 `server_01` 运行没有生成最终高斯 PLY。

## 服务器工作目录

| 路径 | 内容与用途 |
|---|---|
| `/media/ubuntu/data/legiony_3dgsdrag/code/3DGS-Drag/` | 项目代码和本次实验的输出；以下各表的相对路径均从这里开始 |
| `/media/ubuntu/data/legiony_3dgsdrag/data/official_example/person_big/` | 官方人物场景数据；`images/` 有 350 张原始视角图，`sparse/0/` 是 COLMAP 相机和稀疏重建数据；`input/`、`distorted/`、`stereo/` 等是数据准备留下的素材或中间目录 |
| `/media/ubuntu/data/legiony_3dgsdrag/envs/draggs-py310/` | 本实验的 Python 虚拟环境、PyTorch 和已编译的 CUDA 扩展；不是实验结果 |
| `/media/ubuntu/data/legiony_3dgsdrag/huggingface-cache/` | Stable Diffusion v1.4、BLIP 等模型的本地缓存；LoRA 只保存增量权重，推理时仍需基础模型 |
| `/media/ubuntu/data/legiony_3dgsdrag/torch-cache/` | PyTorch/torchvision 下载权重所用的缓存 |
| `/media/ubuntu/data/legiony_3dgsdrag/linux-wheelhouse/` | 为离线安装上传的 Linux Python 安装包及版本清单；不是训练产物 |
| `/media/ubuntu/data/legiony_3dgsdrag/3dgsdrag_source_data.tar.gz` | 最初从本机传来的代码与数据压缩包；不是训练产物 |

项目代码中的 `cfgs/` 保存拖动配置；本次用 `cfgs/person_1_raisearm.json`，其中有 **6 组控制点及位移**。`train_drag.py` 是实验入口，`scene/`、`gaussian_renderer/`、`utils/` 和 `submodules/` 分别负责场景/高斯、渲染、LoRA 与辅助功能、CUDA 扩展。更完整的源码导览见同目录的《代码导览与首次拖动编辑.md》；其中 2026-09-21 的环境状态记录已过时，当前实验成果以本文为准。

## 本次输出：`output/`

```text
output/
├─ person_base/
│  ├─ cameras.json
│  ├─ cfg_args
│  ├─ input.ply
│  └─ point_cloud/iteration_7000/point_cloud.ply       # 初始 3DGS
├─ person_raise_arm_server_01/                         # 失败的首次服务器尝试
│  ├─ cameras.json、cfg_args、input.ply
│  └─ person_1_raisearm/20260926-135053/origin/       # 350 张初始渲染图
└─ person_raise_arm_server_02/                         # 已完成的实验
   ├─ cameras.json、cfg_args、input.ply
   └─ person_1_raisearm/20260926-154932/
      ├─ origin/                                       # 350 张拖动前渲染图
      ├─ render_real/iter_100/                         # 350 张最终模型渲染图
      └─ point_cloud/iteration_100/point_cloud.ply     # 最终 3DGS
```

| 文件夹或文件 | 具体含义 |
|---|---|
| `person_base/` | 初始 3DGS 的训练输出。本次没有重新训练基础模型，而是读取其中 `iteration_7000/point_cloud.ply` |
| `person_raise_arm_server_01/` | 第一次服务器运行在 LoRA 阶段因离线 tokenizer 加载失败而停止；仅保留基础场景记录和 350 张初始渲染图，**没有拖动后的 PLY** |
| `person_raise_arm_server_02/` | 修正 tokenizer 路径后成功完成的运行；`20260926-154932` 是该次运行的时间戳 |
| `origin/0.png` 至 `349.png` | 从拖动前的初始 3DGS 按 350 个相机视角渲染的图像；是 LoRA 的实例训练图，**不是数据集原始照片** |
| `render_real/iter_100/0.png` 至 `349.png` | 从最终优化后的 3DGS 用同一批相机渲染的图像；可与 `origin/` 同编号图片对照看抬臂与背景变化 |
| `point_cloud/iteration_100/point_cloud.ply` | 最终可供 3DGS 查看器加载的编辑结果；本次总共只做了 100 次高斯优化 |
| `cameras.json` | 场景相机参数记录；每次运行的输出根目录各有一份 |
| `cfg_args` | 该输出目录对应的命令行参数记录 |
| `input.ply` | 复制的 COLMAP 稀疏点云，约 10,879 点；**不是**训练好的高斯模型 |

本次参数是 `--drag_steps 1 --anneal_times 1 --camera_num 8 --train_iterations_per_edit 100`。扩散模型修复了选中的 **8 个视角**，随后这些图像被用作高斯优化的监督；代码没有把这 8 张扩散目标图单独写入 `output/`。本次也没有 `.pth` 训练检查点：默认检查点迭代数是 7,000 和 30,000，而本次只运行了 100 次；最终 PLY 仍由末次迭代保存。

## LoRA、日志和临时副本

| 路径 | 内容与用途 |
|---|---|
| `dreambooth/a man standing in front of a wall/class_images/` | 100 张类别图。首次失败运行已生成，成功运行复用了它们 |
| `dreambooth/a man standing in front of a wall/unet/` | UNet 的 LoRA 配置和 `adapter_model.safetensors`；它是增量权重，不是完整 Stable Diffusion 模型 |
| `dreambooth/a man standing in front of a wall/text_encoder/` | 文本编码器的 LoRA 配置和 `adapter_model.safetensors` |
| `tmp/a man standing in front of a wall/` | 代码从 `dreambooth/` 复制出的工作副本，供后续图像编辑加载；权威 LoRA 产物在 `dreambooth/` |
| `logs/person_raise_arm_server_01.log` | 首次失败运行的日志，记录 tokenizer 离线加载错误 |
| `logs/person_raise_arm_server_02.log` | 成功运行的日志：LoRA `200/200`、扩散编辑 `8/8`、高斯优化 `100/100`、`Training complete.` |

`dreambooth/` 和 `tmp/` 里另有 `a man standing in an office with a blue shirt/`，它是此前已有的另一组描述词目录，**不是本次成功运行的抬臂结果**。当前代码以描述词作为 LoRA 目录名；若以后再次得到同一描述词，发现 `unet/` 已存在时会复用其权重。

## 看结果时先打开什么

1. 用 3DGS 查看器分别打开上表中的 **初始** 和 **最终** `point_cloud.ply`，从相同角度比较人物手臂；不要把 `input.ply` 当作编辑结果。
2. 在服务器上对照 `origin/0.png` 与 `render_real/iter_100/0.png`，再抽查其他同编号视角，确认抬臂是否跨视角一致、背景有没有明显损坏。
3. 如需核对运行是否完整，查看 `logs/person_raise_arm_server_02.log` 末尾的 `Training complete.`。

日志证明流程执行完成；画面质量仍需通过 PLY 和多视角渲染图判断。
