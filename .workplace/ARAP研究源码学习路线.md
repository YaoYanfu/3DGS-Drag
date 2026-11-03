# 面向 ARAP 的 3DGS-Drag 源码学习路线

本文依据本地源码整理。目标是能解释主要文件和函数，沿一次拖拽追踪数据，并确定 ARAP 的接入边界。这里的 ARAP 部分是设计建议，尚未实现或验证画质收益。本轮没有运行训练或改动算法。

阅读位置以 `code/3DGS-Drag/` 为根目录；行号对应整理时版本，之后修改代码可能使其移动。已有《代码导览与首次拖动编辑.md》侧重安装与运行，本文侧重理解和改法。

## 1. 先认识三个对象

| 对象 | 保存什么 | 去哪里找 |
|---|---|---|
| `GaussianModel` | 全场景高斯的几何、外观参数，以及优化器、拖拽标签 | `scene/gaussian_model_drag.py:25` |
| `Scene` | 数据集、相机集合、场景尺度、输出位置，并引用高斯模型 | `scene/__init__.py:25` |
| `Drag` | 控制点、总位移、当前拖拽步数、控制点分组 | `train_drag.py:30` |

“场景”“高斯模型”“拖拽控制器”是三个不同职责。`TrainingState`（`train_drag.py:160`）把这些对象和图像、mask、扩散模型打包传递；`DragSchedule`（`:175`）保存循环次数和扩散强度。

注意有两个同名的 `GaussianModel`：

- `train.py` 用 `scene/gaussian_model.py`，负责基础 3DGS 训练。
- `train_drag.py:17` 显式导入 `scene/gaussian_model_drag.py`，负责编辑。

**只改基础训练版本的类，不会改到当前拖拽形变入口。**

## 2. 文件地图：先读主链，外围按需查

| 文件或目录 | 作用 | 本轮阅读重点 |
|---|---|---|
| `train_drag.py` | 编辑总入口：初始化、几何移动、扩散编辑、高斯优化、保存 | 优先读 `training` 和 `run_drag_training_loop`，再顺着调用走 |
| `cfgs/person_1_raisearm.json` | 抬臂的三维控制点与位移 | 读懂 `start`、`direction`、`voxel_size`、`mode` |
| `scene/gaussian_model_drag.py` | 参数存储、选点、复制形变、优化器同步 | `get_xyz`、`training_setup`、`set_drag_idx`、`deform_points` |
| `scene/deform.py` | 从控制点变化计算高斯中心位移与朝向变化 | `deform_Gaussians`、`get_transform`、四元数辅助函数 |
| `gaussian_renderer/__init__.py` | 将高斯与相机送入可微光栅化器 | `render` 的输入、输出和梯度路径 |
| `scene/__init__.py` | 加载数据、相机、初始点云，保存高斯 | `Scene.__init__`、`save`、`getTrainCameras` |
| `scene/cameras.py` | 保存图像、相机参数、视图和投影矩阵 | `Camera.__init__`，注意图像字段之后会被覆盖 |
| `arguments/__init__.py` | 定义数据、渲染和优化参数 | 查参数默认值；不能以“参数存在”推断主链使用了它 |
| `utils/train_dreambooth.py` | 描述场景、启动 LoRA 训练、执行 img2img | `train_dreambooth`、`img2img`，先掌握接口即可 |
| `utils/loss_utils.py` | L1、SSIM 等图像比较函数 | `l1_loss`、`ssim` 的 mask 含义 |
| `utils/perceptual.py` | VGG/LPIPS 感知距离 | `PerceptualLoss.forward` |
| `train.py` | 从数据集训练初始高斯模型 | 拖拽研究初期只需知道它提供 `.ply` |
| `convert.py`、`scene/dataset_readers.py` | 数据预处理与读取 | 数据或坐标有问题时再深入 |
| `submodules/` | CUDA 光栅化与初始化 KNN 扩展 | 初次理解 ARAP 接口不必进入 CUDA 实现 |
| `utils/knn.py` | CPU KDTree 查询邻居的小工具 | 主链未调用；不是已经接好的 ARAP 邻接图 |
| `utils/sde_drag.py`、`utils/cycle_sde.py` | 其他拖拽实验工具 | 当前主入口没有调用，暂时跳过 |

## 3. 自顶向下读一次编辑

按以下顺序打开代码，第一次只看调用，不钻进每一个函数内部：

```text
main()                                      train_drag.py:658
  └─ training()                             :558
       ├─ init_scene_and_gaussians()         :204
       │    ├─ Scene(...) + load_ply(...)
       │    ├─ training_setup(...)
       │    ├─ Drag(...)
       │    ├─ render_sets_train(...)        :151
       │    └─ model.train_dreambooth(...)
       └─ run_drag_training_loop()           :508
            对每个拖拽步：
            ├─ apply_one_drag_step()         :312
            │    └─ drag_copy_paste_step_deform() :139
            │         └─ gaussians.deform_points(...)
            │              └─ deform_Gaussians(...)
            └─ 对每个退火强度：
                 ├─ run_one_edit_pass()      :359
                 │    └─ model.img2img(...)
                 └─ optimize_gaussians_for_edits() :392
```

可以用下面这段伪代码复述，注意 LoRA 不在拖拽循环内部：

```python
state, schedule = init_scene_and_gaussians(...)  # 此处训练或复用 LoRA
for drag_step in range(schedule.drag_steps):
    apply_one_drag_step(...)                    # 直接改变几何
    for strength in compute_step_strengths(schedule):
        run_one_edit_pass(...)                  # 生成二维监督
        optimize_gaussians_for_edits(...)       # 用监督优化三维高斯
```

`drag_steps × anneal_times × train_iterations_per_edit` 才是这里高斯拟合的总步数（`:517`）。例如 `2 × 4 × 500 = 4000`，代表 2 次几何移动、8 轮扩散编辑、4000 次高斯参数更新，不包括 LoRA 训练。

### 初始化具体做了什么

`init_scene_and_gaussians` 的主要边界：

1. `:207` 创建编辑版高斯模型和 `Scene`；`Scene` 最初可能从稀疏点云初始化。
2. `:214` 若传入 `--ply`，再加载训练好的高斯，覆盖前面的初始化。
3. `:221` 建立 Adam；`:226` 读取拖拽配置并标记选区。
4. `:240` 渲染各相机下的选区 mask；`:258` 取训练相机子集。
5. `:267` 渲染编辑前全视角图，保存至 `origin/`；`:276` 用其训练或复用 LoRA。
6. `:279` 另外保存训练相机的原数据集图像，用于后续背景保持。

这里有两种“编辑前图像”：供 LoRA 的是初始高斯的渲染图；供背景保持的 `original_images` 是相机原数据图像。不要混为一组。

## 4. 高斯模型中的数据长什么样

设全场景有 `N` 个高斯，球谐阶数为 `d`，`K=(d+1)^2`。

| 属性 | 形状 | 含义／注意事项 |
|---|---|---|
| `_xyz` | `[N, 3]` | 高斯中心坐标；`get_xyz` 直接返回它 |
| `_rotation` | `[N, 4]` | 朝向四元数，实部在前；渲染访问 `get_rotation` 时归一化 |
| `_scaling` | `[N, 3]` | 存储对数尺度；`get_scaling` 做 `exp` 才是实际尺度 |
| `_opacity` | `[N, 1]` | 存储透明度的 logit；`get_opacity` 做 `sigmoid` |
| `_features_dc` | `[N, 1, 3]` | 球谐颜色的常数部分 |
| `_features_rest` | `[N, K-1, 3]` | 其余球谐外观系数 |
| `_drag_idx` | `[N]` | 0 为不属于当前拖拽组，1、2……为控制点区域标签 |

`training_setup`（`gaussian_model_drag.py:167`）将前六组参数交给 Adam。因此后面的图像拟合不只更新位置，还能更新形状、透明度和外观。

`get_xyz` 是 `@property`，用法是 `gaussians.get_xyz`，不是 `gaussians.get_xyz()`。本项目没有在这个属性里运行形变网络。

## 5. JSON 控制点如何变成一簇高斯

先读 `Drag.__init__`（`train_drag.py:31`）和 `setup_drag_idx`（`:117`）：

- `start[i]`：第 i 个三维控制点坐标，整体为 `[H,3]`。
- `direction[i]`：总位移，不是目标坐标，也不必是单位向量。
- `drag_steps`：将总位移等分几次。
- `voxel_size`：此实现用作欧氏距离选点半径，见 `get_point_mask`（`gaussian_model_drag.py:283`）；没有做体素化。
- `mode`：字符串 `"move"` 或 `"extend"`。
- `cluster`：可选的控制点分组；缺省时全部控制点同组，共同影响该组的高斯。

`set_drag_idx`（`:384`）按球形区域选点；如果球重叠，先处理的控制点优先取得高斯。随后 `setup_drag_idx` 将控制点位置改成它实际选到的高斯的质心。

抬臂配置有 6 个控制点，`voxel_size=0.3`，`mode="move"`，没有 `cluster`，所以 6 个控制点共同参与同一组形变。`local_mask` 虽然写在配置里，当前主链没有读取它。

以第一个控制点位移 `[-0.17,-0.1,-0.06]` 为例，若 `drag_steps=2`，每步移动量就是 `[-0.085,-0.05,-0.03]`。代码用“重定位后的控制点 + 每步位移”作为本步目标。

注意 `Drag.targets` 在质心重定位前就已创建，但真正执行移动的 `drag_copy_paste_step_deform` 不用它，而是重新计算本步目标。不要只盯着 `self.targets` 推断主链。

## 6. 粗形变：ARAP 最直接的接入位置

先读 `Drag.drag_copy_paste_step_deform`（`train_drag.py:139`）：

```python
step_vec = self.direction / float(self.steps)
start_points = self.startpoints
target_points = self.startpoints + step_vec
self.gaussians.deform_points(
    start_points, target_points, self.idxs_list, self.mode
)
self.startpoints = target_points.detach()
```

该函数管理步数和控制点目标，不求解几何。接着进入 `GaussianModel.deform_points`（`gaussian_model_drag.py:311`）：

1. 按 `idxs_list` 逐组处理。
2. 根据 `_drag_idx` 找出本组高斯；设数量为 `M`。
3. 取它们的中心 `[M,3]`、四元数 `[M,4]`，以及本组控制点 `[H,3]`。
4. `:324` 调用 `deform_Gaussians`，得到新中心和朝向。
5. 复制颜色、尺度、不透明度等属性，把新高斯追加到模型。
6. 新高斯继承拖拽标签；旧高斯标签清零。
7. 旧高斯在 `move` 模式下透明度设成约 0.01，`extend` 下约 0.5。

因此 copy-paste 是“保留旧点并降低其透明度，同时添加变形后的副本”。它没有直接删除旧点。下一次拖拽根据正标签选择上一轮副本，不是重新从全部点中按球选一次。

`densification_postfix`（`:435`）和 `cat_tensors_to_optimizer`（`:413`）把追加后的参数重新接进 Adam，并扩展动量状态。这里虽然名字带 densification，当前用途是追加复制点；不能据此说编辑循环正在运行基础 3DGS 的自适应增密策略。

**改算法时不要绕过这套追加和优化器维护，只简单给 `_xyz` 塞一个新的 `nn.Parameter`。否则模型参数与优化器持有的参数可能不再对应。**

### 当前 `deform_Gaussians` 实际计算什么

位置：`scene/deform.py:133`。这是普通张量计算函数，没有网络和优化器。

- 只有一个控制点：所有选中中心加上相同位移，四元数保持不变。
- 多个控制点：`get_transform`（`:103`）计算每个控制点的位移及一个估计朝向；再按高斯到控制点的距离混合这些量。
- `:163` 的位置公式是 `new_xyz = xyz + transform`。没有将中心绕某个支点乘以估计的旋转矩阵。
- `:164` 用四元数乘法合成新朝向，影响椭球方向。

也就是说，“中心如何走”与“椭球如何转”在这里是两条计算路径。看到四元数旋转，不能直接推断中心也做了刚体旋转。

阅读时值得单独核验、但本轮没有改动的两处：

1. `:146-153` 保留最近两个控制点后，把负距离除以负距离和。若距离是 1 和 3，权重会是 0.25 和 0.75，而不是近点权重大。不要把它口头解释成逆距离插值。
2. `quaternion_between_vectors`（`:87`）用 `[dot,cross]` 构造四元数。代入互相垂直的单位 x、y 向量，会得到 `[0,0,0,1]`；按本文件实部在前的约定，它对应绕 z 轴半周，并不将 x 转到 y。这值得做独立的旋转正确性检查。`get_transform` 中的权重也需按源码逐项核验。

保留这些观察，后续如需修正，应把“原实现修正”和“引入 ARAP”分开比较，避免混淆收益来源。

## 7. 扩散监督与高斯优化：把两条梯度路径分清

`apply_one_drag_step`（`train_drag.py:312`）在 `torch.no_grad()` 中执行粗形变，然后合并拖拽前后位置的选区 mask。此时没有 `loss.backward()`。

`run_one_edit_pass`（`:359`）渲染当前高斯，调用 `DreamBooth.img2img`，再将生成图写入相机的 `original_image`。这个字段最初是原数据集图，**此时已改为编辑监督图**。

`DreamBooth.train_dreambooth`（`utils/train_dreambooth.py:340`）接收 PNG 目录，不接收高斯对象；它训练或复用扩散模型 LoRA。`img2img`（`:400`）无梯度运行，返回二维目标图。扩散模型没有通过这次调用直接反传到三维高斯。

`optimize_gaussians_for_edits`（`train_drag.py:392`）才做高斯梯度更新：

```text
Gaussian 参数 → render → 当前渲染图 → 与目标图的损失
      ↑                                  │
      └──────── Adam ← backward ──────────┘
```

`render`（`gaussian_renderer/__init__.py:42`）取中心、尺度、朝向、透明度、球谐系数，调用 CUDA 光栅化器；返回 `render`、屏幕空间点、可见性和投影半径。理解 ARAP 接口时，掌握输入输出即可。

实际损失（`train_drag.py:443-466`）为：

\[
L=L_{perceptual}(I,I_{edit})+L_1(I,I_{edit})
+100\{(1-\lambda)L_1(I(1-M),I_{orig}(1-M))
+\lambda[1-SSIM_{outside}(I,I_{orig})]\}.
\]

其中 `M` 是累计编辑区域，`I_orig` 来自之前缓存的数据集图片。`p_ssim` 变量实际调用 VGG/LPIPS 感知损失，不是 SSIM；背景项才另外调用 SSIM。`utils/loss_utils.py:59` 将 mask 里的 SSIM 值置 1，使这些像素对 `1-SSIM` 不贡献误差。

图像损失没有按 `_drag_idx` 冻结非编辑高斯。六组参数都在优化器中，背景保护依赖损失约束，而不是点级冻结。

还有几处“写了但主链没用”的内容：

- `Drag.drag_loss`、`track_point`、`track_point_idx` 没在当前循环调用。
- `run_one_edit_pass` 没传 mask 给 img2img，不能说它在运行局部 inpainting。
- 返回的 `view.weight` 被保存，但当前损失没有使用。
- 相机按 mask 面积排序后又将整个排序随机打乱，再取前 `camera_num`；并非严格选面积最大的视角。
- 编辑循环没有调用 `update_learning_rate` 或 `densify_and_prune`；不要把基础训练循环的行为套过来。

## 8. ARAP 应该改哪里：先明确研究变量

以下是接入设计建议，不能视作当前代码已有功能。

| 路线 | 修改位置 | 回答的问题 |
|---|---|---|
| A：ARAP 粗形变 | 在 `GaussianModel.deform_points` 中替换形变计算调用 | 更好的几何初值能否减少拉伸、改善最终编辑 |
| B：ARAP 正则 | 在 `optimize_gaussians_for_edits` 的 `loss.backward()` 前加入几何项 | 图像拟合时保持邻域结构能否减少几何漂移 |

你的目标是 deformation 阶段，建议先做 A，再把 B 作为独立消融。粗形变区域在 `no_grad` 内，不能仅在那里加一个 loss 就指望现有高斯训练自动优化它。独立的 local/global ARAP 求解可在无自动求导的上下文运行；若采用 Adam 求解，则需显式在内部开启梯度，并使用独立优化变量。

### ARAP 需要的数据

经典 ARAP 衡量邻边形状能否由局部旋转解释：

\[
E_{ARAP}(X,R)=\sum_i\sum_{j\in\mathcal N(i)}w_{ij}
\|(x_i-x_j)-R_i(p_i-p_j)\|^2.
\]

`p` 是形变前的参考坐标，`x` 是待求坐标，`R_i` 是局部旋转，邻接关系与权重在参考形状上定义。需要添加拖动约束和固定区域约束，才能得到期望编辑。经典算法交替求局部旋转和全局位置。参考：[Sorkine 与 Alexa 的原论文](https://igl.ethz.ch/projects/ARAP/arap_web.pdf)。

这篇论文针对三角网格；将其用于 Gaussian 中心需要自己定义图。以下 KNN、边界等处理是面向本项目的适配建议：

1. **定义求解区域。** 初步可在选中的活动高斯上建图；若希望与身体连接处保持稳定，应纳入连接处的固定邻居或边界约束。仅对孤立选区求解，无法自动约束它与未参与图的身体之间的接缝。
2. **固定参考图。** 在本次粗形变开始前保存 `p`，建立邻接边并在求解期间固定。点云可从对称化 KNN 图加距离阈值开始，排除自身；需检查手臂与躯干距离很近时的错误跨部位连边。不能凭空使用网格余切权重。
3. **建立控制约束。** 当前每个 handle 对应一簇点。可以约束簇质心，或让指定的 handle 点保持原相对位置并各自加上位移；不要把簇内所有高斯都约束到同一个目标坐标，那会让它们塌缩。若全部选中点都是硬 handle，内部形变自由度也会受限，需明确自由点与控制点如何区分。
4. **处理固定区域与零空间。** 为每个连通分量提供足够的约束；单个位置约束仍可能留下旋转自由度。关节附近哪些点固定，是几何设计，不由 `mode` 自动决定。
5. **求 `new_xyz` 与局部 `R`。** 用 local/global 迭代，检查线性系统、反射修正 `det(R)=+1` 和收敛。
6. **同步椭球方向。** 将局部 `R` 转成四元数，以 `q_new = q_delta * q_old` 合成并归一化；第一版可保留原尺度、外观等复制逻辑。这不等于完整的外观旋转建模。
7. **接回复制逻辑。** 输出的中心和朝向必须与选中点顺序一致。若求解域包括固定边界，只把需要移动的活动点输出给现有复制步骤。

### 预期改动范围

- 新增 `scene/arap.py`：邻接图、约束、求解和局部旋转计算；不要混入扩散或保存逻辑。
- 修改 `scene/gaussian_model_drag.py:324` 附近：把求解所需的选区、控制点对应和边界信息传进去，取回新位置与朝向，保留后续复制和 Adam 状态维护。
- 修改 `train_drag.py` 参数入口并传递形变方法：能在原方法与 ARAP 间切换，便于对照。只有 A 路线时不必改图像 loss。

接口草图只表达所需职责，尚不是可粘贴运行的补丁：

```python
# graph / handles / anchors 需要由本步参考形状及标签显式构造。
new_xyz, local_R = solve_arap(
    rest_xyz, graph, handles, handle_targets, anchors
)
new_rotation = compose_with_original_rotation(local_R, old_rotation)
# 然后接回已有的属性复制、标签迁移与优化器扩展。
```

不能把 ARAP 简化成 `loss = mean((new_xyz-old_xyz)**2)`；那会惩罚位移本身，也没有表达局部旋转。

### 复制与多步拖拽的对应问题

每次复制会追加新行，旧点不消失，而且参数对象会被重新创建。若 B 路线要跨拟合迭代保存图，必须记录参考点到当前活动副本的索引映射，不能一直引用旧 Tensor 或旧副本的索引。

A 路线可以先在每个粗形变步开始时，为当前活动点重建参考与图；这约束的是“每一步的局部形变”。若始终以编辑前形状作为参考，则约束的是累计变化，必须维护跨步对应关系。两者不是同一实验。

当前控制点每步按计划位移推进，拟合后没有调用跟踪函数。若 ARAP 用软约束而实际 handle 没到目标，下步继续沿计划位置走可能产生偏差，应检查实际约束误差。

## 9. 怎样验证改动确实改到了想研究的阶段

先做独立几何检查，再花时间跑完整扩散流程：

| 检查 | 期望行为 |
|---|---|
| 零位移 | 中心基本不动，局部旋转接近单位阵 |
| 与锚点相容的整体平移 | 所有中心同向移动，邻边保持，朝向基本不变 |
| 与锚点相容的整体刚体旋转 | 中心和椭球方向都正确转动，ARAP 能量接近零 |
| 指定 handle 与固定边界 | 达到设定约束精度；固定边界无明显漂移 |
| 退化或不合格输入 | 空选区、孤立分量、重复点等在求解前被明确发现 |
| 复制后参数一致性 | 所有高斯属性行数一致，标签数一致，Adam 持有新参数 |

接着固定初始 PLY、JSON、相机集合、随机种子、扩散强度和训练预算，比较原方法与 ARAP。这里的初始 PLY 应是训练好的 Gaussian，而不是场景的 COLMAP `input.ply`。

至少记录三个阶段：粗形变后、扩散目标图、高斯拟合后。当前代码没有自动单独保存前两者，正式实验应在 `apply_one_drag_step` 后和 `run_one_edit_pass` 后补充保存，才能判断效果来自哪个阶段。

观察 handle 误差、邻域结构变化、关节接缝、非编辑区变化及多视角结果。两种方法应在共同的参考图上评估几何指标，避免各自换图后指标失去可比性。ARAP 不自动保证无自交、填洞或最终画质提升，收益需要实验确认。

如需进一步拆分因果，可以再用固定的二维监督比较拟合阶段；完整端到端比较则保留“初值变化导致扩散目标变化”这个实际效果。

## 10. 阅读练习：能回答这些，才算进入可改代码的状态

建议分三次，每次约 40–50 分钟：

1. **入口与数据。** 读第 1–5 节对应代码。脱稿画调用链，并解释 `Scene`、`GaussianModel`、`Drag` 各保存什么。
2. **形变。** 读第 6 节对应代码。手算单控制点平移，解释 `[M,H,3]` 的广播，以及副本和旧点的标签／透明度变化。
3. **梯度与改法。** 读第 7–9 节对应代码。指出 loss 从哪里反传，解释为什么 A 与 B 不是同一修改，并画出 ARAP 输入输出。

自测问题：

1. 如果只想改拖拽距离，改 JSON 的哪个字段？如果只想改几何传播方式，应进哪个函数？
2. `voxel_size=0.3` 选择的是体素还是球形邻域？一个 handle 是一个高斯吗？
3. 形变后旧高斯为什么仍在？下一步会拖动旧点还是新副本？
4. `new_xyz = xyz + transform` 中的 `transform` 是变换矩阵还是 `[M,3]` 位移？
5. 扩散模型更新了哪些模型参数？高斯参数的 `backward` 在哪里？
6. `viewpoint_cam.original_image` 在拟合阶段还是原照片吗？
7. 为什么把 ARAP loss 写在 `apply_one_drag_step` 里不会自动生效？
8. 如何让一簇 handle 点移动，同时避免把它们压到同一个点？

参考答案：1. `direction`；`deform_Gaussians` 及其在 `deform_points` 的调用处。2. 球形邻域；通常对应一簇高斯。3. copy-paste 降低旧点透明度而不删除；正标签迁移到新副本。4. 位移。5. LoRA 阶段更新扩散适配参数；高斯在 `optimize_gaussians_for_edits` 的 `loss.backward()` 更新。6. 已被替换为扩散编辑监督，原图另存在 `state.original_images`。7. 此处无自动求导且现有逻辑不优化该能量，需要求解器或明确的优化步骤。8. 约束簇质心或保持点内相对偏移，先明确控制／自由点划分。
