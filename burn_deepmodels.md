# Burn框架中《深度学习小书》第四、五章渐进式分析

## 引言

《深度学习小书》（Little Book of Deep Learning）第四章"模型组件"和第五章"架构"涵盖了深度学习的核心模型构建块和架构设计。本分析文档旨在提供一个渐进式的学习路径，将第四、五章的理论知识点与 Rust 深度学习框架 [Burn](https://github.com/tracel-ai/burn) 的代码实现进行对应。

**文档组织方式**：按知识点组织，每个知识点内部分三个深度层次循序渐进：

- **入门**：通过 Burn 高级 API 快速上手，适合初学者
- **进阶**：数学公式与源代码的直接对应，适合中级学习者
- **专家**：数值稳定性、底层实现细节，适合框架开发者

---

## 4. 模型组件（Model Components）

### 4.1 层的概念（The Concept of Layers）

深度学习模型由标准化的张量运算组合而成，这些运算经过设计和经验验证是通用且高效的。在 Burn 中，层通过 `#[derive(Module)]` 宏实现模块化设计。

#### 入门：Module trait 与层组合

**Burn 实现**：`burn::module::Module`

```rust
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

// 通过 #[derive(Module)] 定义自定义层
#[derive(Module, Debug)]
struct CustomLayer<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
}

impl<B: Backend> CustomLayer<B> {
    fn new(d_input: usize, d_hidden: usize, d_output: usize, device: &B::Device) -> Self {
        Self {
            linear1: LinearConfig::new(d_input, d_hidden).init(device),
            linear2: LinearConfig::new(d_hidden, d_output).init(device),
        }
    }
    
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(x);
        let x = burn::tensor::activation::relu(x);
        self.linear2.forward(x)
    }
}
```

#### 进阶：Module trait 架构

**Burn 源代码**：`crates/burn-core/src/module/base.rs`

```rust
pub trait Module<B: Backend>: Clone + Send + Debug {
    type Record: Record<B>;
    
    fn collect_devices(&self, devices: Devices<B>) -> Devices<B>;
    fn devices(&self) -> Devices<B>;
    fn fork(self, device: &B::Device) -> Self;
    fn to_device(self, device: &B::Device) -> Self;
    fn no_grad(self) -> Self;
    fn num_params(&self) -> usize;
    fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V); // 只读遍历参数树
    fn map<M: ModuleMapper<B>>(self, mapper: &mut M) -> Self; // 消费并重建参数树
    fn into_record(self) -> Self::Record;
    fn load_record(self, record: Self::Record) -> Self;
}
```

`visit()` 与 `map()` 构成了 Module 系统的“读写分离”设计：前者负责遍历和收集（如参数统计、梯度提取），后者负责按统一规则变换参数并返回更新后的新模块实例。

#### 专家：训练时的梯度收集与应用链路

> Visitor/Mapper 模式的基础定义（`ModuleVisitor`/`ModuleMapper` trait、`Param<T>` 包装结构、`#[derive(Module)]` 的宏展开细节）见 [`burn_foundations.md` §5](burn_foundations.md#5-深度的价值the-value-of-depth)。这里聚焦训练时的关键路径：`visit()` 如何收集梯度，`map()` 如何应用梯度。

**梯度收集链路（`visit`）**：

训练循环的关键三步 —— `backward()` → `GradientsParams::from_grads()` → `optim.step()`：

```rust
let grads = loss.backward();
let grads = GradientsParams::from_grads(grads, &model);  // 内部调用 module.visit()
model = optim.step(lr, model, grads);                     // 内部调用 module.map()
```

`GradientsParams::from_grads(...)` (crates/burn-optim/src/optim/grads.rs) 通过 `visit()` 遍历每个 `Param`，调用 `param.val().grad_remove(grads)` 按 `ParamId` 取出梯度 → 写入 `GradientsParams`。

**梯度应用链路（`map`）**：

`optim.step()` 创建 `SimpleOptimizerMapper`，调用 `module.map(&mut mapper)`，对每个参数执行：

- `param.consume()` 取出 `(id, tensor, mapper)`
- `grads.remove(id)` 按 `ParamId` 取梯度
- `optimizer.step(lr, tensor, grad, state)` 计算新参数
- `Param::from_mapped_value(...)` 重建参数节点，返回新模型实例

**流程总览**：`backward()` → `from_grads()` + `visit()` 收集 → `optim.step()` + `map()` 应用 → 返回参数更新后的新模型。

`map()` 是”遍历并重建”整个模块树（非原地改值），这保证优化器只需关心单参数更新，参数树遍历由 Module 系统统一处理。 


---

### 4.2 线性层（Linear Layers）

线性层（更准确说是仿射层）实现
$$
Y = XW + b
$$
是 MLP、CNN 分类头、Transformer 前馈网络中最常见的基础模块。

#### 入门：API 用法

**Burn 实现**：`burn::nn::Linear`

```rust
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

let device = Default::default();
let linear = LinearConfig::new(128, 64)  // 输入维度 128，输出维度 64
    .with_bias(true)                     // 包含偏置项
    .init(&device);

let input = Tensor::<Backend, 2>::random([32, 128], Distribution::Default, &device);
let output = linear.forward(input);      // 形状 [32, 64]
```

#### 进阶：数学公式与源码对应

项目当前锁定版本（`Cargo.lock` 对应 commit `8cc356e`）下，线性层关键实现分成两层：

1) `crates/burn-nn/src/modules/linear.rs`：模块封装与参数配置

```rust
impl<B: Backend> Linear<B> {
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        linear(
            input,
            self.weight.val(),
            self.bias.as_ref().map(|b| b.val()),
        )
    }
}
```

2) `crates/burn-tensor/src/tensor/module.rs`：底层张量算子 `linear`

```rust
pub fn linear<B: Backend, const D: usize>(
    input: Tensor<B, D>,
    weight: Tensor<B, 2>,
    bias: Option<Tensor<B, 1>>,
) -> Tensor<B, D> {
    if D == 1 {
        let input = input.unsqueeze::<2>();
        let output = linear(input, weight, bias);
        return output.squeeze_dim(0);
    }

    let weight = weight.unsqueeze::<D>();
    let bias = bias.map(|bias| bias.unsqueeze::<D>());

    let output = input.matmul(weight);
    match bias {
        Some(bias) => output.add(bias),
        None => output,
    }
}
```

这里有三个非常关键的实现细节：

- **1D 输入特殊分支**：`D == 1` 时先升维成 2D 做 matmul，再 `squeeze_dim(0)` 返回，保证 `Linear` 对向量输入也可用。
- **广播前置**：`weight.unsqueeze::<D>()` 与 `bias.unsqueeze::<D>()` 在 matmul/add 前执行，统一处理高维前缀批次维度（`[..., d_in] -> [..., d_out]`）。
- **与 PyTorch 语义差异**：Burn 的 `linear` **不自动转置权重**；文档注释明确指出这点（PyTorch `F.linear` 是 `x @ weight^T + b`）。

#### 专家：初始化、布局映射与训练语义

**1) `LinearConfig` 默认初始化不是“拍脑袋常量”，而是可推导的 Kaiming Uniform**

`LinearConfig` 默认值：
```rust
initializer: Initializer::KaimingUniform {
    gain: 1.0 / sqrt(3.0),
    fan_out_only: false,
}
```

而 `Initializer` 源码中：
- Kaiming Uniform 采样区间：
  $$[-a, a],\quad a = \sqrt{3} \cdot gain \cdot \frac{1}{\sqrt{fan}}$$
- 当 `gain = 1/\sqrt{3}` 且 `fan = fan_in`（`fan_out_only = false`）时：
  $$a = \frac{1}{\sqrt{fan_{in}}}$$

这正对应 `Linear` 注释中的经典区间 `U(-k, k), k=\sqrt{1/d_{input}}`。

同一文件还实现了 Xavier/Kaiming 的 normal/uniform 版本：
- Xavier: $$\sigma = \sqrt{\frac{2}{fan_{in}+fan_{out}}}$$
- Kaiming: $$\sigma = \frac{1}{\sqrt{fan}}$$（再乘对应 gain）

**gain 与 fan 详解**：

**fan（扇入/扇出）**：fan_in 是输入连接数（`d_input`），fan_out 是输出连接数（`d_output`）。对于线性层 $Y = XW + b$（$[\ldots, d_{in}] \times [d_{in}, d_{out}] \to [\ldots, d_{out}]$），每个输出神经元接收 $d_{in}$ 个输入。

**`fan_out_only` 旗标 — 前向方差 vs 反向梯度稳定**：

`fan_out_only` 的选择背后是两条不同的"方差稳定"目标：

- **`false`（fan_in 模式）**：初始化方差由 $1/fan\_in$ 决定，目标是在 **前向传播** 中保持每层激活值的方差不变 —— 即 $Var(y_l) \approx Var(y_{l-1})$。
- **`true`（fan_out 模式）**：初始化方差由 $1/fan\_out$ 决定，目标是在 **反向传播** 中保持每层梯度的方差不变 —— 即 $Var(\partial L/\partial x_l) \approx Var(\partial L/\partial x_{l-1})$。

**为什么不能同时满足**：

对于线性层 $y = xW$（$W \in \mathbb{R}^{fan\_in \times fan\_out}$）：

| 方向 | 方差关系 | 需要 $Var(W)$ |
|------|---------|--------------|
| **前向** | $Var(y) = fan\_in \cdot Var(W) \cdot Var(x)$ | $\frac{1}{fan\_in}$ |
| **反向** | $Var(\partial L/\partial x) = fan\_out \cdot Var(W) \cdot Var(\partial L/\partial y)$ | $\frac{1}{fan\_out}$ |

当 $fan\_in \neq fan\_out$ 时，两个条件无法同时满足。必须选择优先保证哪个方向。

**各模式的适用场景**：

| 模式 | 优先保证 | 适合场景 |
|------|---------|---------|
| **fan_in**（默认）| 前向激活稳定 | 极深网络（ResNet-152）、窄→宽层 |
| **fan_out** | 反向梯度稳定 | 宽→窄层（分类头）、GAN 生成器末端 |
| **Xavier** | 折中（调和平均） | 对称网络、tanh/sigmoid 激活 |

**fan_in 为什么是更好的默认**：

1. **激活爆炸比梯度爆炸更容易扩散**：激活值失控会在前向传播中逐层放大，直接影响所有后续层；梯度失控仅在反向传播中影响前半部分层。前向方差失控的破坏范围更大。
2. **与归一化层的协同**：BatchNorm / LayerNorm 在前向传播中直接校正激活分布，它们的归一化统计量（均值、方差）依赖稳定的前向激活。前向激活方差过大或过小都会影响归一化效果。
3. **反向梯度本身有额外的补偿机制**：梯度裁剪（§4.2）、学习率预热（§3.4）、跳跃连接（§4.7）的恒等梯度路径都可以在反向传播中缓解梯度异常。前向激活的异常缺乏同级别的"安全网"。
4. **He 论文的实证结论**：He et al. (2015) 的实验表明，对于 ReLU 网络，fan_in 模式在 30 层时的收敛效果与 fan_out 模式相当，但在更深的网络中稳定性更好。

**一个具体例子**：

考虑一个 $fan\_in=256, fan\_out=64$ 的线性层（如 CNN 末端的降维层）：

- **fan_in 模式**：$Var(W) = 1/256$ → 前向激活方差稳定，反向梯度方差会被放大 $fan\_out/fan\_in = 0.25$ 倍（实际是缩小，安全）
- **fan_out 模式**：$Var(W) = 1/64$ → 反向梯度方差稳定，前向激活方差被放大 $fan\_in/fan\_out = 4$ 倍（激活值逐层膨胀）

fan_in 模式在这个例子中更安全——梯度缩小不会导致训练中断，但激活膨胀可能导致数值溢出。



**gain（增益系数）**：由激活函数决定，用于纠正非线性变换对信号方差的影响：

| 激活函数 | gain（normal） | 原理 |
|---------|:-----------:|------|
| Linear / Identity | 1 | 无非线性变换 |
| ReLU | $\sqrt{2} \approx 1.414$ | ReLU 将一半输出置零，方差减半 |
| LeakyReLU($a$) | $\sqrt{2/(1+a^2)}$ | 负半轴保留部分信号 |
| Tanh | $5/3$ | Tanh 在零点的压缩效应 |

**为什么 ReLU 需要 gain = $\sqrt{2}$**：

考虑前向传播 $y = \text{ReLU}(Wx)$，$W$ 的每个元素 i.i.d. 均值为 0：

$$\text{Var}(y_l) = n_l \cdot \text{Var}(w_l) \cdot \text{Var}(y_{l-1}) \cdot \frac{1}{2}$$

其中 $1/2$ 来自 ReLU 将一半输入置零（对称分布经过 ReLU 后方差恰好减半）。要使 $\text{Var}(y_l) = \text{Var}(y_{l-1})$，需要：

$$\text{Var}(w) = \frac{2}{fan\_in}$$

正态分布 $\mathcal{N}(0, \sigma^2)$ 下：$\sigma = \sqrt{2 / fan\_in}$，即 gain = $\sqrt{2}$。
均匀分布 $U(-a, a)$ 下：$a^2/3 = 2/fan\_in$，即 $a = \sqrt{6 / fan\_in}$。

**Uniform 与 Normal 的方差等价转换**：

- 均匀分布 $U(-a, a)$ 方差 = $a^2/3$
- 正态分布 $\mathcal{N}(0, \sigma^2)$ 方差 = $\sigma^2$
- 等价转换：$a = \sigma \cdot \sqrt{3}$

这就是 Burn 代码中 `a = sqrt(3) * gain * kaiming_std` 里 `sqrt(3)` 的来源——将正态分布的标准差换算为均匀分布的边界。

**Kaiming 与 Xavier 的对比**：

| | Kaiming/He | Xavier/Glorot |
|---|---|---|
| **标准差** | $gain \cdot 1/\sqrt{fan\_in}$ | $gain \cdot \sqrt{2/(fan\_in + fan\_out)}$ |
| **激活假设** | ReLU 及变体 | 线性 / tanh / sigmoid |
| **设计思路** | 仅关注前向方差（fan_in） | 折中前后向方差（调和平均） |

Xavier 假设激活函数在零点附近近似线性，因此取 fan_in 和 fan_out 的调和平均来平衡前向和反向。Kaiming 专为 ReLU 设计——ReLU 在负半轴斜率为 0，线性假设不成立，需要 gain 补偿方差损失。

**Burn 默认 gain = $1/\sqrt{3}$ 的来源**：

这个值与 PyTorch `nn.Linear` 的默认行为一致：

$$\text{PyTorch: } kaiming\_uniform\_(weight, a=\sqrt{5}) \;\Longrightarrow\; gain = \sqrt{\frac{2}{1+a^2}} = \sqrt{\frac{2}{6}} = \frac{1}{\sqrt{3}}$$

代入 Burn 公式：$a_{bound} = \sqrt{3} \cdot \frac{1}{\sqrt{3}} \cdot \frac{1}{\sqrt{fan\_in}} = \frac{1}{\sqrt{fan\_in}}$

这个值的效果介于 gain=1（线性）和 gain=$\sqrt{2}$（ReLU）之间，是一个在实践中验证过的保守默认。如果模型全部使用 ReLU 激活，应将 gain 设为 $\sqrt{2}$ 以获得理论最优的方差保持。

**2) `LinearLayout::Col` 的重点在”参数存储布局”，不是改变前向公式**

`LinearLayout` 有 `Row | Col` 两种布局。

- `Row`：按 `[d_input, d_output]` 初始化权重。
- `Col`：先按 `[d_output, d_input]` 初始化，并通过 `save_mapper / load_mapper / init_mapper` 在保存、加载、初始化路径做转置映射，确保模块内部前向仍可按 `linear(input, weight, bias)` 一致工作。

`linear.rs` 测试 `layout` 与 `col_row_same_result` 专门验证了：
- Col 布局在序列化/反序列化后形状映射正确；
- Col 与 Row 在同权重语义下前向结果一致。

**3) 自动微分层面：Linear 的梯度来自基础算子组合规则**

`Linear::forward` 本身只是调用 `matmul + add` 组合；在 `Autodiff` 后端下，梯度由这些基础算子的反向规则自动组成。对应经典结果：
- $$\partial L/\partial X = (\partial L/\partial Y)W^T$$
- $$\partial L/\partial W = X^T(\partial L/\partial Y)$$
- $$\partial L/\partial b = \sum_{batch}(\partial L/\partial Y)$$

**4) 与外部框架互操作时最容易踩坑的一点**

若从 PyTorch `nn.Linear` 直接迁移权重，需先确认权重形状语义：
- PyTorch 常见存储：`[d_out, d_in]`
- Burn `linear` 计算期望：`[d_in, d_out]`

因此通常需要显式转置后再导入，避免“能跑但结果错”的隐蔽问题。

---

### 4.3 卷积层（Convolutional Layers）

卷积层是处理图像、音频等结构化信号的核心组件，具有平移等变性和参数共享特性。

#### 入门：API 用法

**Burn 实现**：`burn::nn::conv::{Conv1d, Conv2d, Conv3d}`

```rust
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::prelude::*;

let device = Default::default();
let conv = Conv2dConfig::new([3, 64], [3, 3])  // 输入通道 3，输出通道 64，核大小 3x3
    .with_stride([1, 1])                       // 步幅 1
    .with_padding([1, 1])                      // 填充 1（保持空间尺寸）
    .with_dilation([1, 1])                     // 膨胀率 1
    .with_groups(1)                            // 分组数 1
    .with_bias(true)                           // 包含偏置
    .init(&device);

let input = Tensor::<Backend, 4>::random([32, 3, 32, 32], Distribution::Default, &device);
let output = conv.forward(input);              // 形状 [32, 64, 32, 32]
```

#### 进阶：数学公式与源码对应

**2D 卷积公式**：
对于输入 $X \in \mathbb{R}^{B \times C_{\text{in}} \times H \times W}$，权重 $W \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times K \times L}$，偏置 $b \in \mathbb{R}^{C_{\text{out}}}$：
$$
Y_{b,c',h,w} = \sum_{c=1}^{C_{\text{in}}} \sum_{k=1}^{K} \sum_{l=1}^{L} X_{b,c,h+k,w+l} \cdot W_{c',c,k,l} + b_{c'}
$$

**Burn 源代码**：`crates/burn-nn/src/modules/conv/conv2d.rs`

```rust
impl<B: Backend> Conv2d<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        conv2d(
            input,
            self.weight.val(),
            self.bias.as_ref().map(|b| b.val()),
            PaddedConvOptions::new(self.stride, [self.padding, self.padding], self.dilation, self.groups),
        )
    }
}
```

卷积操作按维度分为 `conv1d`、`conv2d`、`conv3d` 三个独立函数，位于 `crates/burn-tensor/src/tensor/module.rs`。

#### 专家：卷积算法、梯度计算与性能优化

**1. 卷积算法的实现策略**

卷积操作的计算复杂度高（$O(B \cdot C_{\text{in}} \cdot C_{\text{out}} \cdot H \cdot W \cdot K \cdot L)$），不同后端采用各自的优化策略：

- **NdArray 后端**（CPU）：直接 MAD（Multiply-Add）嵌套循环实现，`crates/burn-ndarray/src/ops/conv.rs`
- **CubeCL 后端**（GPU）：提供多种策略可选
  - `direct`：直接卷积计算
  - `im2col`：将卷积展开为矩阵乘法，利用 GPU GEMM 优化（`crates/burn-cubecl/src/kernel/conv/im2col.rs`）
  - `implicit_gemm`：隐式 GEMM 算法，避免显式 im2col 的内存开销（`crates/burn-cubecl/src/kernel/conv/forward/implicit_gemm/`）

注意：Burn 目前未实现 Winograd 或 FFT 卷积算法。

**2. 自动微分与梯度计算**

卷积操作的反向传播需要计算三个梯度：

- **输入梯度**：$\frac{\partial L}{\partial X} = \text{conv2d\_transpose}(\frac{\partial L}{\partial Y}, W, \text{stride}, \text{padding}, \text{dilation})$
- **权重梯度**：$\frac{\partial L}{\partial W} = \text{conv2d}(X, \frac{\partial L}{\partial Y}, \text{stride}, \text{padding}, \text{dilation})$
- **偏置梯度**：$\frac{\partial L}{\partial b} = \sum_{b,h,w} \frac{\partial L}{\partial Y_{b,c,h,w}}$

Burn 的 `Autodiff` 后端为 `convolution` 操作注册自定义梯度函数，实现上述计算。

**3. 卷积变体与扩展**

- **转置卷积（反卷积）**：`ConvTransposeNd` 实现上采样，常用于生成模型和分割网络
  - 数学上等价于在输入间插入零的常规卷积
- **空洞卷积**：`dilation > 1` 增大感受野而不增加参数，用于多尺度特征提取
- **分组卷积**：`groups > 1` 将通道分组独立处理，减少参数和计算量
  - 极端情况 `groups = C_in = C_out` 为深度可分离卷积
- **可变形卷积**：通过学习偏移量使采样位置自适应内容

**4. Burn 中的性能优化**

- **`burn-fusion` 算子融合**：`Autodiff<B, burn::backend::Fusion>` 自动合并连续逐元素操作（如 `relu(bn(conv(x)))` 中的 add/mul/relu），减少中间张量分配和内存传输
- **后端策略选择**：CubeCL GPU 后端支持 `direct`、`im2col`、`implicit_gemm` 多种卷积策略，可通过 autotune 自动选择最优方案（`crates/burn-cubecl/src/kernel/conv/forward/`）
- **分组卷积**：`Conv2dConfig::with_groups(n)` 直接映射后端优化路径，无需特殊模块
- **混合精度**：`burn::backend::Autodiff<MixedPrecisionBackend<Wgpu>>` 自动管理 fp16/bf16 前向传播和 fp32 主权重副本

**5. 数值稳定性与精度**

- **整数卷积**：对量化模型使用整数卷积，需注意累加溢出问题
- **混合精度训练**：使用 fp16 权重但 fp32 累加器防止精度损失
- **梯度裁剪**：深层卷积网络易出现梯度爆炸，需在优化器设置梯度裁剪

**6. 感受野计算与特征图尺寸**

对于 D 维卷积：
$$
T' = \frac{T + 2P - K - (K-1)(D-1)}{S} + 1
$$
其中 $D$ 为膨胀率（dilation）。感受野大小 $R$ 与层数 $L$、核大小 $K$、膨胀率 $D$、步幅 $S$ 的关系为：
$$
R_L = 1 + \sum_{l=1}^{L} (K_l - 1) \times D_l \times \prod_{i=1}^{l-1} S_i
$$

**7. 与 §4.6 归一化层的联系**

卷积层常后接批归一化层（§4.6），训练时可分别更新，推理时可将批归一化参数（$\gamma, \beta$）融合到卷积权重中：
$$
W_{\text{fused}} = \gamma \cdot W, \quad b_{\text{fused}} = \gamma \cdot b + \beta
$$
这种融合减少推理时的计算量和内存访问。

---

### 4.4 池化层（Pooling Layers）

池化层通过下采样减少空间维度，提取局部特征并增加平移不变性。

#### 入门：API 用法

**Burn 实现**：`burn::nn::pool::{MaxPool1d, MaxPool2d, AvgPool1d, AvgPool2d}`

```rust
use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
use burn::prelude::*;

let device = Default::default();
let pool = MaxPool2dConfig::new([2, 2])  // 核大小 2x2
    .with_stride([2, 2])                 // 步幅 2
    .with_padding([0, 0])                // 无填充
    .init(&device);

let input = Tensor::<Backend, 4>::random([32, 64, 32, 32], Distribution::Default, &device);
let output = pool.forward(input);        // 形状 [32, 64, 16, 16]
```

#### 进阶：数学公式与源码对应

**最大池化公式**：
$$
Y_{b,c,h,w} = \max_{\substack{0 \leq k < K \\ 0 \leq l < L}} X_{b,c, S_h h + k, S_w w + l}
$$

**平均池化公式**：
$$
Y_{b,c,h,w} = \frac{1}{KL} \sum_{k=0}^{K-1} \sum_{l=0}^{L-1} X_{b,c, S_h h + k, S_w w + l}
$$

**Burn 源代码**：`crates/burn-nn/src/modules/pool/max_pool2d.rs`，底层实现在 `crates/burn-tensor/src/tensor/module.rs`

```rust
// MaxPool2d::forward 调用 burn::tensor::module::max_pool2d
impl MaxPool2d {
    pub fn forward<B: Backend>(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_batch_size, _channels_in, height_in, width_in] = input.dims();
        let ((top, bottom), (left, right)) =
            self.padding.calculate_padding_2d_pairs(height_in, width_in, &self.kernel_size, &self.stride);
        // 非对称 padding 时先显式 pad，再调用零 padding 的池化
        if top != bottom || left != right {
            let padded = input.pad((left, right, top, bottom), PadMode::Constant(f32::NEG_INFINITY));
            max_pool2d(padded, self.kernel_size, self.stride, [0, 0], self.dilation, self.ceil_mode)
        } else {
            max_pool2d(input, self.kernel_size, self.stride, [top, left], self.dilation, self.ceil_mode)
        }
    }
}
```

`max_pool2d` 由各后端分别实现。以 ndarray 后端（`crates/burn-ndarray/src/ops/maxpool.rs`）为例，核心逻辑是直接嵌套循环遍历输出位置和池化窗口：

```rust
for oh in 0..out_height {
    for ow in 0..out_width {
        let mut max_val = -f32::INFINITY;
        for kh in 0..kernel_height {
            for kw in 0..kernel_width {
                let val = x[[b, c, ih, iw]];
                if val > max_val { max_val = val; }
            }
        }
        output[[b, c, oh, ow]] = max_val;
    }
}
```

池化前先用 `-inf` 填充边界（`apply_padding_4d`），确保填充区域不影响最大值结果。

#### 专家：池化策略与梯度传播

**最大池化的梯度传播**（ndarray 后端 `max_pool2d_backward`，`crates/burn-ndarray/src/ops/maxpool.rs:205`）：

最大池化使用索引追踪（`max_pool2d_with_indices`）记录每个输出位置对应输入中最大值的位置，反向传播时梯度仅流向该索引处：

```rust
// backward: 仅最大值位置接收梯度
for h in 0..height {
    for w in 0..width {
        let index = indices[[b, c, h, w]];  // 前向时记录的最大值坐标
        output[[b, c, index_h, index_w]] += output_grad[[b, c, h, w]];
    }
}
```

平均池化则均匀分配：$\frac{\partial L}{\partial x_{ij}} = \frac{1}{K \times L} \cdot \frac{\partial L}{\partial y}$。

**自适应池化**：`AdaptiveAvgPool2d`（`crates/burn-nn/src/modules/pool/adaptive_avg_pool2d.rs`）按目标输出尺寸反推核大小与步幅，再调用标准 `avg_pool2d`，无需训练参数。

**池化的配置语义**：Burn 的 `PaddingConfig2d` 支持 `Valid`（无填充）、`Same`（保持尺寸，偶数核自动用非对称填充）和 `Explicit(top, left, bottom, right)` 三种模式。`ceil_mode` 控制输出尺寸上取整还是下取整。

---

### 4.5 Dropout 层

Dropout 是一种正则化技术，通过随机丢弃激活防止过拟合和促进独立特征学习。

#### 入门：API 用法

**Burn 实现**：`burn::nn::Dropout`

```rust
use burn::nn::{Dropout, DropoutConfig};
use burn::prelude::*;

let dropout = DropoutConfig::new(0.5).init();  // 丢弃概率 50%
let device = Default::default();

// 训练模式：自动微分后端（Autodiff）下自动生效
let input = Tensor::<Backend, 2>::random([32, 128], Distribution::Default, &device);
let output_train = dropout.forward(input.clone());

// 评估模式：非 Autodiff 后端下直接返回输入
let output_eval = dropout.forward(input);  // 等同恒等映射
```

#### 进阶：数学公式与源码对应

**Dropout 公式**：
$$
Y_i = \begin{cases}
0 & \text{以概率 } p \\
\frac{X_i}{1-p} & \text{以概率 } 1-p
\end{cases}
$$

**Burn 源代码**：`crates/burn-nn/src/modules/dropout.rs`

```rust
impl Dropout {
    pub fn forward<B: Backend, const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        // 非 Autodiff 后端或 prob=0 时直接返回输入（零开销）
        if !B::ad_enabled(&input.device()) || self.prob == 0.0 {
            return input;
        }
        // 训练模式：生成 Bernoulli 掩码并缩放
        let prob_keep = 1.0 - self.prob;
        let random = input.random_like(Distribution::Bernoulli(prob_keep));
        input * random * (1.0 / prob_keep)
    }
}
```

Dropout 的模式切换不依赖显式 `train()/eval()` 方法，而是通过 `B::ad_enabled()` 自动检测后端类型——`Autodiff` 后端自动启用 dropout，纯推理后端则直接透传。

#### 专家：Dropout 变体与空间 Dropout

**Dropout 变体**：
- **标准 Dropout**：独立丢弃每个激活。Burn 的 `Dropout` 是维度无关的——`forward<B, const D: usize>` 接受任意维度的张量
- **空间 Dropout 模式**：丢弃整个特征图（通道），适用于卷积层。Burn 没有独立的 `Dropout2d` 模块，但可通过在通道维度生成掩码手动实现：先创建形状 `[B, C, 1, 1]` 的 Bernoulli 掩码再广播
- **DropPath（随机深度）**：以概率 $p$ 丢弃整个残差块，需手动实现

**训练/评估模式切换**：Dropout 通过 `B::ad_enabled()` 自动检测当前后端类型：
- `Autodiff` 后端 → `ad_enabled() = true` → 执行 dropout
- 纯推理后端 → `ad_enabled() = false` → 透传输入

这比显式 `train()/eval()` 状态切换更安全——不会因忘记切换模式而导致推理时执行 dropout。

---

### 4.6 归一化层（Normalization Layers）

归一化层通过标准化激活的分布来稳定训练过程。

#### 入门：API 用法

**Batch Normalization**：
```rust
use burn::nn::norm::{BatchNorm, BatchNormConfig};
use burn::prelude::*;

let device = Default::default();
let bn = BatchNormConfig::new(128)  // 特征数 128
    .with_epsilon(1e-5)              // 数值稳定性常数
    .with_momentum(0.1)              // 移动平均动量
    .init(&device);

let input = Tensor::<Backend, 2>::random([32, 128], Distribution::Default, &device);
let output = bn.forward(input);      // forward<const D: usize> 自动适配任意维度
```

Burn 的 `BatchNorm<B>` 是维度泛型的——单个结构体通过 `forward<const D: usize>()` 处理 1D/2D/3D 输入，无 `BatchNorm1d/2d/3d` 的区分。

**Layer Normalization**：
```rust
use burn::nn::norm::{LayerNorm, LayerNormConfig};

let ln = LayerNormConfig::new(128)  // 特征维度 128
    .with_epsilon(1e-5)
    .init(&device);

let output = ln.forward(input);
```

#### 进阶：数学公式与源码对应

**批归一化公式**：
$$
\hat{x} = \frac{x - \mathbb{E}[x]}{\sqrt{\text{Var}[x] + \epsilon}} \quad;\quad y = \gamma \hat{x} + \beta
$$

**层归一化公式**：
$$
\hat{x}_i = \frac{x_i - \mathbb{E}[x]}{\sqrt{\text{Var}[x] + \epsilon}} \quad;\quad y_i = \gamma \hat{x}_i + \beta
$$

**Burn 源代码**：`crates/burn-nn/src/modules/norm/batch.rs`

```rust
// BatchNorm 前向传播（简化自源码）
impl<B: Backend> BatchNorm<B> {
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        // 通过 B::ad_enabled() 自动判断训练/推理模式
        match B::ad_enabled(&input.device()) {
            true => self.forward_train(input),
            false => self.forward_inference(input),
        }
    }
    
    fn forward_train<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        // 计算当前批次的均值与方差
        let mean = /* 沿 batch 和 spatial 维度求均值 */;
        let var = /* 沿 batch 和 spatial 维度求方差 */;
        // 更新移动统计量: running = (1-momentum)*old + momentum*new
        let running_mean = running_mean.mul_scalar(1.0 - self.momentum)
            .add(mean.clone().detach().mul_scalar(self.momentum));
        // 归一化 + 缩放
        let normalized = (input - mean) / (var + self.epsilon).sqrt();
        normalized.mul(self.gamma.val()).add(self.beta.val())
    }
}
```

注意：`running_mean` 更新遵循 `(1−momentum)×old + momentum×new`，而非常见的 `momentum×old + (1−momentum)×new`。`gamma`/`beta` 初始化为全 1/全 0。

#### 专家：归一化变体与应用场景

**归一化层对比**：
| 类型 | 归一化维度 | 适用场景 | Burn 实现 |
|------|-----------|----------|-----------|
| **BatchNorm** | 批次×空间维度 | 卷积网络、大批次训练 | `BatchNorm<B>` — `forward<const D: usize>` 泛型适配 |
| **LayerNorm** | 特征维度 | Transformer、RNN、小批次 | `LayerNorm<B>` |
| **InstanceNorm** | 空间维度 | 风格迁移、生成模型 | `InstanceNorm<B>` |
| **GroupNorm** | 分组特征 | 小批次训练 | `GroupNorm<B>` |
| **RmsNorm** | 特征维度 | LLaMA 类模型 | `RmsNorm<B>` |

**数值稳定性**：`eps` 参数防止除零错误，通常设置为 $10^{-5}$。

---

### 4.7 跳跃连接（Skip Connections）

跳跃连接通过将早期层的激活直接传递到深层，缓解梯度消失问题并促进特征重用。

#### 入门：API 用法

**残差连接实现**：
```rust
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

#[derive(Module, Debug)]
struct ResidualBlock<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
}

impl<B: Backend> ResidualBlock<B> {
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let residual = x.clone();
        let x = self.linear1.forward(x);
        let x = burn::tensor::activation::relu(x);
        let x = self.linear2.forward(x);
        let x = burn::tensor::activation::relu(x);
        x + residual  // 残差连接
    }
}
```

#### 进阶：数学公式与源码对应

**残差块公式**：
$$
y = \mathcal{F}(x; W) + x
$$

其中 $\mathcal{F}$ 表示残差函数（如两个卷积层的组合）。

**Burn 实现模式**：
```rust
// 1. 恒等映射（当维度匹配时）
output = block(input) + input;

// 2. 投影映射（当维度不匹配时）
output = block(input) + projection(input);

// 3. 瓶颈设计（ResNet 变体）
output = conv1x1(relu(bn(conv3x3(relu(bn(conv1x1(input))))))) + input;
```

#### 专家：残差连接变体与初始化

**Pre-activation 残差块**（ResNet v2）——归一化和激活在卷积之前，梯度路径更短：

```rust
// Pre-activation: BN → ReLU → Conv → BN → ReLU → Conv, 最后+ shortcut
fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
    let residual = x.clone();
    let x = self.bn1.forward(x);
    let x = burn::tensor::activation::relu(x);
    let x = self.conv1.forward(x);
    let x = self.bn2.forward(x);
    let x = burn::tensor::activation::relu(x);
    let x = self.conv2.forward(x);
    x + residual
}
```

**Bottleneck 残差块**——1×1 卷积压缩/恢复维度，减少参数：

```rust
fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
    let residual = x.clone();
    let x = self.conv1.forward(x);  // 1x1: C_in → C_mid (降维)
    let x = self.bn1.forward(x);
    let x = burn::tensor::activation::relu(x);
    let x = self.conv2.forward(x);  // 3x3: C_mid → C_mid
    let x = self.bn2.forward(x);
    let x = burn::tensor::activation::relu(x);
    let x = self.conv3.forward(x);  // 1x1: C_mid → C_out (升维)
    let x = self.bn3.forward(x);
    // shortcut 处理维度/通道不匹配
    let residual = match &self.shortcut {
        Some(proj) => proj.forward(residual),
        None => residual,
    };
    burn::tensor::activation::relu(x + residual)
}
```

**初始化策略**：最后一层 BN 的 $\gamma$ 初始化为 0（Burn 中可通过 `Initializer::Zeros` 配置），使初始残差块接近恒等映射，有利于极深网络早期训练。

**梯度流动**：
$$
\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \left(1 + \frac{\partial \mathcal{F}}{\partial x}\right)
$$
即使 $\frac{\partial \mathcal{F}}{\partial x} \approx 0$，梯度仍可通过恒等项 $1$ 直接回传。这也是为什么 ResNet 中最后一层 BN 零初始化是安全的——初始时 $\mathcal{F}(x) \approx 0$，整个块退化为恒等映射，梯度通过 $1$ 项无损传播。

---

### 4.8 注意力层（Attention Layers）

注意力层允许模型动态关注输入的不同部分，是 Transformer 架构的核心组件。

#### 入门：API 用法

**多头注意力**：
```rust
use burn::nn::attention::{MultiHeadAttention, MultiHeadAttentionConfig};
use burn::prelude::*;

let device = Default::default();
let mha = MultiHeadAttentionConfig::new(512, 8)  // 嵌入维度 512，8 个头
    .init(&device);

// 自注意力（Q, K, V 相同）
let query = Tensor::<Backend, 3>::random([32, 10, 512], Distribution::Default, &device);
let output = mha.forward(query.clone(), query.clone(), query);

// 交叉注意力（不同的 K, V）
let key_value = Tensor::<Backend, 3>::random([32, 20, 512], Distribution::Default, &device);
let output = mha.forward(query, key_value.clone(), key_value);
```

#### 进阶：数学公式与源码对应

**注意力公式**：
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

**多头注意力公式**：
$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O
$$
$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

**Burn 源代码**：`crates/burn-nn/src/modules/attention/multi_head_attention.rs`

```rust
impl<B: Backend> MultiHeadAttention<B> {
    pub fn forward(
        &self,
        query: Tensor<B, 3>,
        key: Tensor<B, 3>,
        value: Tensor<B, 3>,
        mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_length_q, d_model] = query.dims();
        let d_k = d_model / self.n_heads;
        
        // 线性投影到 Q, K, V
        let q = self.query_proj.forward(query).reshape([batch_size, seq_length_q, self.n_heads, d_k]);
        let k = self.key_proj.forward(key).reshape([batch_size, seq_length_k, self.n_heads, d_k]);
        let v = self.value_proj.forward(value).reshape([batch_size, seq_length_k, self.n_heads, d_k]);
        
        // 计算注意力分数
        let scores = q.matmul(k.transpose()) / (d_k as f64).sqrt();
        let scores = match mask {
            Some(mask) => scores.mask_fill(mask, f64::NEG_INFINITY),
            None => scores,
        };
        
        let attn = scores.softmax(3);  // 沿序列维度
        let output = attn.matmul(v);
        
        // 合并多头输出
        let output = output.reshape([batch_size, seq_length_q, d_model]);
        self.output_proj.forward(output)
    }
}
```

#### 专家：注意力机制、梯度计算与高效实现

**1. 注意力机制的数学细节**

标准缩放点积注意力的数学形式为：
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

- **缩放因子** $\frac{1}{\sqrt{d_k}}$：防止点积值过大导致 softmax 梯度消失
- **softmax 温度**：可引入温度参数 $\tau$：$\text{softmax}(x/\tau)$，控制注意力分布尖锐程度
- **注意力稀疏性**：可通过 top-k 筛选或稀疏 softmax 减少计算

**2. 自动微分与梯度计算**

注意力层包含三个可微操作：矩阵乘法、缩放、softmax、第二个矩阵乘法。

- **softmax 梯度**：若 $y = \text{softmax}(x)$，则 $\frac{\partial y_i}{\partial x_j} = y_i(\delta_{ij} - y_j)$
- **整体梯度流**：$\frac{\partial L}{\partial Q} = \frac{\partial L}{\partial \text{Attention}} \cdot \frac{\partial \text{Attention}}{\partial Q}$ 等

Burn 的自动微分后端将注意力分解为基本操作，分别计算梯度后组合。

**3. 高效注意力实现技术**

标准注意力 $O(n^2 d)$ 复杂度对长序列不可行，需优化：

- **FlashAttention** [Dao et al., 2022]：Burn 在 CubeCL GPU 后端实现了 FlashAttention（`crates/burn-cubecl/src/kernel/attention/base.rs`），支持 `FlashBlackboxAccelerated`、`FlashUnit`、`Fallback` 等多种策略，通过分块计算和重计算避免存储 $n \times n$ 注意力矩阵。注意这是 GPU 专属优化，NdArray CPU 后端不包含。
- **线性注意力**：使用核函数近似，复杂度 $O(nd^2)$
  $$ \text{LinearAttn}(Q, K, V) = \phi(Q)(\phi(K)^\top V) $$
  其中 $\phi$ 为特征映射（如 $\phi(x) = \text{elu}(x) + 1$）。Burn 目前未内置线性注意力模块，需自定义实现。

**4. 注意力变体与掩码机制**

- **因果注意力**（自回归）：使用下三角布尔掩码 $M_{ij} = \begin{cases} 0 & i \geq j \\ -\infty & i < j \end{cases}$
- **局部注意力**：滑动窗口掩码 $M_{ij} = \begin{cases} 0 & |i-j| \leq w \\ -\infty & \text{否则} \end{cases}$
- **跨步注意力**：稀疏模式减少计算，如 BigBird 的全局+局部+随机注意力

**5. 数值稳定性**

注意力 softmax 的数值稳定技巧与 §1.1 一致：Burn 的 softmax 实现通过 `x - x.detach().max()` 防溢出。混合精度下，`MixedPrecisionBackend` 自动管理 fp16/fp32 的精度转换。

**6. 与 §4.7 跳跃连接的联系**

Transformer 块中，注意力层通常与残差连接（§4.7）和层归一化（§4.6）结合：
$$
\text{Output} = \text{LayerNorm}(x + \text{Attention}(x))
$$
残差连接确保梯度直接回传，缓解梯度消失。

**7. 计算复杂度分析**

设序列长度 $n$，隐藏维度 $d$，头数 $h$：
- **标准注意力**：$O(n^2 d + nd^2)$（注意力 + 投影）
- **线性注意力**：$O(nd^2)$（无 $n^2$ 项）
- **内存占用**：标准注意力需存储 $n \times n$ 矩阵（$O(n^2)$），优化版本可降至 $O(n)$

---

### 4.9 令牌嵌入（Token Embeddings）

嵌入层将离散令牌（如单词、子词）映射到连续向量空间。

#### 入门：API 用法

**Burn 实现**：`burn::nn::Embedding`

```rust
use burn::nn::{Embedding, EmbeddingConfig};
use burn::prelude::*;

let device = Default::default();
let embedding = EmbeddingConfig::new(10000, 512)  // 词汇表大小 10000，嵌入维度 512
    .init(&device);

let token_ids = Tensor::<Backend, 2, Int>::from_ints([[1, 2, 3], [4, 5, 6]], &device);
let embeddings = embedding.forward(token_ids);  // 形状 [2, 3, 512]
```

#### 进阶：数学公式与源码对应

**嵌入公式**：
$$
E(i) = W_i \quad \text{其中 } W \in \mathbb{R}^{V \times d}
$$

**Burn 源代码**：`crates/burn-nn/src/modules/embedding.rs`

```rust
impl<B: Backend> Embedding<B> {
    pub fn forward<const D: usize>(&self, indices: Tensor<B, D, Int>) -> Tensor<B, D> {
        // 使用 gather 操作从权重矩阵中提取对应行的向量
        self.weight.gather(0, indices)
    }
}
```

#### 专家：嵌入技巧与预训练嵌入

**嵌入技巧**：
- **权重绑定**：共享嵌入层和输出层的权重
- **层归一化**：在嵌入后应用层归一化稳定训练
- **缩放嵌入**：将嵌入乘以 $\sqrt{d}$ 以保持方差

**位置敏感嵌入**：
```rust
// 组合令牌嵌入和位置嵌入
let token_embeddings = embedding.forward(tokens);
let position_embeddings = position_encoding.forward(positions);
let embeddings = token_embeddings + position_embeddings;
```

---

### 4.10 位置编码（Positional Encoding）

位置编码为序列中的令牌添加位置信息，弥补自注意力机制的位置不变性。

#### 入门：API 用法

**正弦位置编码**：
```rust
use burn::nn::pos_encoding::{PositionalEncoding, PositionalEncodingConfig};
use burn::prelude::*;

let device = Default::default();
let pos_encoding = PositionalEncodingConfig::new(512, 10000)  // 维度 512，最大序列长度 10000
    .init(&device);

let batch_size = 32;
let seq_length = 128;
let d_model = 512;
let embeddings = Tensor::<Backend, 3>::random([batch_size, seq_length, d_model], Distribution::Default, &device);

// 添加位置编码
let output = pos_encoding.forward(embeddings);
```

#### 进阶：数学公式与源码对应

**正弦位置编码公式**：
$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)
$$
$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

**Burn 源代码**：`crates/burn-nn/src/modules/pos_encoding.rs`

```rust
impl<B: Backend> PositionalEncoding<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_length, d_model] = x.dims();
        
        // 生成位置编码矩阵
        let positions = Tensor::arange(0..seq_length as i64, &x.device())
            .reshape([seq_length, 1]);
        let div_term = Tensor::arange(0..d_model as i64, &x.device())
            .reshape([1, d_model])
            .float()
            .mul_scalar(-(10000.0f64).ln() / d_model as f64)
            .exp();
        
        let pos_encoding = positions.float().matmul(div_term);
        let pos_encoding = pos_encoding.sin_cos();  // 交替 sin/cos
        
        x + pos_encoding.unsqueeze::<3>(0)
    }
}
```

#### 专家：位置编码变体

**位置编码类型**：
- **绝对位置编码**：正弦编码、学习的位置嵌入
- **相对位置编码**：考虑令牌相对距离（T5, Transformer-XL）
- **旋转位置编码**（RoPE）：通过旋转矩阵编码相对位置
- **ALiBi**：基于线性偏置的相对位置编码

**RoPE 实现**：`crates/burn-nn/src/modules/rope_encoding.rs`
```rust
// 实际类名为 RotaryEncoding（非 RopeEncoding）
impl<B: Backend> RotaryEncoding<B> {
    pub fn forward(&self, x: Tensor<B, 3>, positions: Tensor<B, 1, Int>) -> Tensor<B, 3> {
        // 对 query 和 key 施加旋转位置编码
        let rotated = self.apply_rotary_pos_emb(x, positions);
        rotated
    }
}
```

---

## 5. 模型架构（Model Architectures）

### 5.1 多层感知机（Multilayer Perceptron, MLP）

MLP 是最简单的深度学习架构，由全连接层与激活函数交替堆叠而成。

#### 入门：API 用法

**简单 MLP 实现**：
```rust
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

#[derive(Module, Debug)]
struct MLP<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    linear3: Linear<B>,
}

impl<B: Backend> MLP<B> {
    fn new(d_input: usize, d_hidden: usize, d_output: usize, device: &B::Device) -> Self {
        Self {
            linear1: LinearConfig::new(d_input, d_hidden).init(device),
            linear2: LinearConfig::new(d_hidden, d_hidden).init(device),
            linear3: LinearConfig::new(d_hidden, d_output).init(device),
        }
    }
    
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(x);
        let x = burn::tensor::activation::relu(x);
        let x = self.linear2.forward(x);
        let x = burn::tensor::activation::relu(x);
        self.linear3.forward(x)
    }
}
```

#### 进阶：通用近似定理

**Cybenko 定理**：具有单隐藏层和非多项式激活函数的 MLP 可以在紧致集上以任意精度近似任何连续函数。

**数学表述**：
对于任意连续函数 $f: [0,1]^n \to \mathbb{R}$ 和 $\epsilon > 0$，存在单隐藏层 MLP $g$ 使得：
$$
\sup_{x \in [0,1]^n} |f(x) - g(x)| < \epsilon
$$

#### 专家：深度与表示能力

**深度优势**：
- **理论结果**：深度网络可以指数级更高效地表示某些函数类
- **实践观察**：更深的网络通常具有更好的泛化能力
- **优化挑战**：深度增加导致梯度消失/爆炸问题

**宽度与深度权衡**：
$$
\text{参数数量} \propto \text{宽度}^2 \times \text{深度}
$$

---

### 5.2 卷积神经网络（Convolutional Neural Networks, CNN）

CNN 是处理图像等网格结构数据的标准架构，结合卷积层、池化层和全连接层。

#### 入门：LeNet 类架构

**VGG 风格 CNN**：
```rust
use burn::nn::{Conv2d, Conv2dConfig, Linear, LinearConfig, MaxPool2d, MaxPool2dConfig};
use burn::prelude::*;

#[derive(Module, Debug)]
struct CNN<B: Backend> {
    conv1: Conv2d<B>,
    pool1: MaxPool2d<B>,
    conv2: Conv2d<B>,
    pool2: MaxPool2d<B>,
    fc1: Linear<B>,
    fc2: Linear<B>,
}

impl<B: Backend> CNN<B> {
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        // 特征提取器
        let x = self.conv1.forward(x);
        let x = burn::tensor::activation::relu(x);
        let x = self.pool1.forward(x);
        
        let x = self.conv2.forward(x);
        let x = burn::tensor::activation::relu(x);
        let x = self.pool2.forward(x);
        
        // 分类器：使用 reshape 将 4D 特征图展平为 2D
        let [b, c, h, w] = x.dims();
        let x = x.reshape([b, c * h * w]);
        let x = self.fc1.forward(x);
        let x = burn::tensor::activation::relu(x);
        self.fc2.forward(x)
    }
}
```

#### 进阶：残差网络（ResNet）

**残差块实现**：
```rust
use burn::nn::{Conv2d, Conv2dConfig};
use burn::nn::norm::{BatchNorm, BatchNormConfig};

#[derive(Module, Debug)]
struct ResidualBlock<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B>,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B>,
    shortcut: Option<Conv2d<B>>,  // 维度不匹配时的投影
}

impl<B: Backend> ResidualBlock<B> {
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let residual = x.clone();
        
        let mut y = self.conv1.forward(x);
        y = self.bn1.forward(y);
        y = burn::tensor::activation::relu(y);
        
        y = self.conv2.forward(y);
        y = self.bn2.forward(y);
        
        // 处理残差连接
        let shortcut = match &self.shortcut {
            Some(conv) => conv.forward(residual),
            None => residual,
        };
        
        burn::tensor::activation::relu(y + shortcut)
    }
}
```

#### 专家：ResNet 架构细节

**ResNet 变体**：
| 变体 | 层数 | 瓶颈设计 | 参数数量 |
|------|------|----------|----------|
| ResNet-18 | 18 | 否 | 11M |
| ResNet-34 | 34 | 否 | 21M |
| ResNet-50 | 50 | 是 | 25M |
| ResNet-101 | 101 | 是 | 44M |
| ResNet-152 | 152 | 是 | 60M |

**瓶颈设计**：
$$
\text{Bottleneck}(x) = W_1 \cdot \text{ReLU}(BN(W_3 \cdot \text{ReLU}(BN(W_2 \cdot x))))
$$
其中 $W_1: 64 \to 256$, $W_2: 256 \to 64$, $W_3: 64 \to 64$（通道数）。

**初始化策略**：
- 最后一层 BN 的 $\gamma$ 初始化为 0
- 使初始残差块接近恒等映射

---

### 5.3 Transformer 架构

Transformer 是基于自注意力的序列建模架构，已成为 NLP 和 CV 领域的主导模型。

#### 入门：Transformer 编码器

**Burn 实现**：`burn::nn::transformer::TransformerEncoder`

```rust
use burn::nn::transformer::{TransformerEncoder, TransformerEncoderConfig};
use burn::prelude::*;

let device = Default::default();
let encoder = TransformerEncoderConfig::new(512, 2048, 8, 6)  // d_model=512, d_ff=2048, heads=8, layers=6
    .init(&device);

let input = Tensor::<Backend, 3>::random([32, 128, 512], Distribution::Default, &device);
let output = encoder.forward(input);  // 形状 [32, 128, 512]
```

#### 进阶：Transformer 块分解

**编码器层结构**：
```rust
#[derive(Module, Debug)]
struct TransformerEncoderLayer<B: Backend> {
    self_attn: MultiHeadAttention<B>,
    feed_forward: PositionWiseFeedForward<B>,
    norm1: LayerNorm<B>,
    norm2: LayerNorm<B>,
    dropout: Dropout,
}

impl<B: Backend> TransformerEncoderLayer<B> {
    fn forward(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 2, Bool>>) -> Tensor<B, 3> {
        // 残差连接 1: 自注意力
        let residual = x.clone();
        let x = self.norm1.forward(x);
        let x = self.self_attn.forward(x.clone(), x, x, mask);
        let x = self.dropout.forward(x);
        let x = x + residual;
        
        // 残差连接 2: 前馈网络
        let residual = x.clone();
        let x = self.norm2.forward(x);
        let x = self.feed_forward.forward(x);
        let x = self.dropout.forward(x);
        x + residual
    }
}
```

#### 专家：Transformer 架构设计、训练与优化

**1. Transformer 架构设计原理**

Transformer 的核心设计选择及其数学原理：

- **残差连接**（§4.7）：$x_{l+1} = x_l + F(x_l)$ 确保梯度直接回传，缓解梯度消失
- **层归一化**（§4.6）：在残差连接前归一化（Pre-LN）或后归一化（Post-LN），影响训练稳定性
- **缩放注意力**：$\frac{QK^\top}{\sqrt{d_k}}$ 防止点积值过大导致 softmax 饱和
- **前馈网络**：两层线性变换 + 激活函数，提供非线性容量

**2. 主要 Transformer 变体**

| 变体 | 架构 | 预训练目标 | 应用 |
|------|------|------------|------|
| **原始 Transformer** | 编码器-解码器 | 序列到序列 | 机器翻译 |
| **BERT** | 仅编码器 | 掩码语言建模 (MLM) | 文本分类、NER |
| **GPT** | 仅解码器 | 因果语言建模 (CLM) | 文本生成 |
| **T5** | 编码器-解码器 | 文本到文本 | 多任务学习 |

**3. 前馈网络实现细节**

前馈网络（FFN/PWFF）是 Transformer 的关键组件，Burn 中实现为 `PositionWiseFeedForward`（`crates/burn-nn/src/modules/transformer/pwff.rs`）：

```rust
// PositionWiseFeedForwardConfig::new(d_model, d_ff)  → 内部两层 Linear + Dropout + 激活函数
#[derive(Module, Debug)]
pub struct PositionWiseFeedForward<B: Backend> {
    linear_inner: Linear<B>,   // d_model → d_ff
    linear_outer: Linear<B>,   // d_ff → d_model
    dropout: Dropout,
    activation: Activation<B>, // 默认 GELU
}
```

**激活函数选择**：
- **GELU**：$x \Phi(x)$，其中 $\Phi$ 为标准正态 CDF，平滑近似 ReLU
- **Swish**：$x \cdot \sigma(\beta x)$，可学习平滑度参数
- **ReLU**：计算简单但输出非零均值

**4. 位置编码策略**

位置编码使模型感知序列顺序：

- **绝对位置编码**：正弦/余弦函数（原始 Transformer）或可学习嵌入
  ```rust
  fn sinusoidal_pos_encoding(seq_len: usize, d_model: usize) -> Tensor<B, 2> {
      let positions = Tensor::arange(0..seq_len);
      let dimensions = Tensor::arange(0..d_model);
      let angles = positions.unsqueeze(1) / (10000.0.pow(dimensions / d_model));
      angles.sin() // 偶索引
          .masked_fill(dimensions % 2 == 1, angles.cos())  // 奇索引
  }
  ```
- **相对位置编码**：注意力分数中加入相对位置偏置 $B_{i-j}$
- **旋转位置编码**：RoPE，通过旋转矩阵编码相对位置

**5. 训练与推理优化**

- **学习率调度**：预热 + 余弦退火的组合通过 `ComposedLrScheduler` 实现（见 [`burn_foundations.md` §3.4](burn_foundations.md#34-学习率调度learning-rate-scheduling)）
- **梯度裁剪**：`GradientClipping::Norm(threshold)` 防止注意力分数导致的梯度爆炸（见 [`burn_foundations.md` §4.2](burn_foundations.md#42-梯度裁剪gradient-clipping)）
- **混合精度训练**：`Autodiff<MixedPrecisionBackend<Wgpu>>` 在 fp16/bf16 下执行前向传播，内部维护 fp32 主权重副本
- **KV 缓存**：`TransformerEncoder::new_autoregressive_cache()` 在自回归推理时缓存历史 K, V，避免重复计算
- **激活 checkpointing**：`Autodiff<B, BalancedCheckpointing>` 在反向传播时重计算部分激活，以计算换内存

**6. 与 §5.2 卷积神经网络的对比**

| 特性 | CNN | Transformer |
|------|-----|-------------|
| **归纳偏置** | 局部性、平移等变 | 序列建模、全局依赖 |
| **计算复杂度** | $O(n)$ | $O(n^2)$（注意力） |
| **并行性** | 高度并行 | 序列依赖（解码器） |
| **数据需求** | 较少 | 大量 |

**7. Burn 中的 Transformer 实现**

Burn 的 Transformer 模块设计灵活，支持自定义：
- `TransformerEncoderConfig`：编码器配置
- `TransformerDecoderConfig`：解码器配置  
- `TransformerEncoderDecoderConfig`：完整编码器-解码器

支持多种注意力机制、位置编码和归一化方案。

---

### 5.4 Vision Transformer (ViT)

ViT 将 Transformer 架构应用于图像分类，将图像分割为补丁序列。

#### 入门：ViT 实现

**补丁嵌入**：
```rust
use burn::nn::{Conv2d, Conv2dConfig, Linear, LinearConfig, LayerNorm, LayerNormConfig};

#[derive(Module, Debug)]
struct PatchEmbedding<B: Backend> {
    projection: Conv2d<B>,  // 使用卷积实现补丁投影
    cls_token: Param<Tensor<B, 1>>,
    position_embedding: Param<Tensor<B, 2>>,
}

impl<B: Backend> PatchEmbedding<B> {
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        // 将图像分割为补丁并展平
        // 输入: [B, C, H, W] -> 输出: [B, N, D]
        // 其中 N = (H/P) * (W/P), D = C * P * P
        
        let patches = self.projection.forward(x);  // 使用核大小=P, 步幅=P 的卷积
        let [b, d, h, w] = patches.dims();
        let patches = patches.reshape([b, d, h * w]).transpose();
        
        // 添加 [CLS] 令牌
        let cls_tokens = self.cls_token.val().unsqueeze::<3>(0).repeat(0, b);
        let embeddings = Tensor::cat(vec![cls_tokens, patches], 1);
        
        // 添加位置编码
        embeddings + self.position_embedding.val()
    }
}
```

#### 进阶：ViT 架构细节

**ViT 公式**：
1. **补丁分割**：$X \in \mathbb{R}^{H \times W \times C} \to X_p \in \mathbb{R}^{N \times (P^2 C)}$
2. **线性投影**：$Z_0 = [x_{\text{class}}; x_p^1 E; \ldots; x_p^N E] + E_{\text{pos}}$
3. **Transformer 编码器**：$Z_l' = \text{MSA}(\text{LN}(Z_{l-1})) + Z_{l-1}$
4. **MLP 头**：$y = \text{MLP}(\text{LN}(Z_L^0))$

**Burn 实现概览**：
```rust
#[derive(Module, Debug)]
struct VisionTransformer<B: Backend> {
    patch_embedding: PatchEmbedding<B>,
    encoder: TransformerEncoder<B>,
    norm: LayerNorm<B>,
    head: Linear<B>,
}

impl<B: Backend> VisionTransformer<B> {
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        // 补丁嵌入
        let x = self.patch_embedding.forward(x);
        
        // Transformer 编码器
        let x = self.encoder.forward(x);
        
        // 提取 [CLS] 令牌并分类
        let cls_token = x.slice([0.., 0..1, 0..]);  // 第一个位置是 [CLS]
        let cls_token = self.norm.forward(cls_token);
        self.head.forward(cls_token.squeeze(1))
    }
}
```

#### 专家：ViT 变体与混合架构

**ViT 变体与 Burn 实现映射**：
| 变体 | 关键创新 | Burn 中实现方式 |
|------|---------|---------------|
| **ViT** | 补丁嵌入 + CLS 令牌 | `Unfold4d` 切片 + 可学习 `Param<Tensor>` |
| **DeiT** | 蒸馏令牌 | 额外 `dist_token: Param<Tensor<B, 1>>` 并行于 CLS |
| **Swin Transformer** | 移位窗口 + 分层结构 | 多尺度 `Unfold4d` + `MultiHeadAttention` + 自定义窗口掩码 |
| **FlexiViT** | 可变补丁大小 | 统一 `patch_size` 配置，动态计算序列长度 |

**补丁嵌入的 Burn 实现**：使用 `Unfold4d`（`crates/burn-nn/src/modules/unfold.rs`）按核大小=补丁大小、步幅=补丁大小的配置切片，等价于 `Conv2d` 补丁投影的展开式实现：

```rust
// 使用 Conv2d 方案（文档入门示例）：直接投影 + reshape
let patches = Conv2d::forward(&self.proj, x);
// 或使用 Unfold4d 方案：先切片为补丁窗口，再线性投影
let patches = Unfold4d::forward(&self.unfold, x);  // [B, C*P*P, N]
```

**混合架构模式**——CNN 提取局部特征 + Transformer 建模全局关系：
```rust
struct HybridEncoder<B: Backend> {
    cnn_stem: Conv2d<B>,               // 输入层：局部特征提取
    transformer: TransformerEncoder<B>, // 全局关系建模
    head: Linear<B>,
}
```

**位置编码策略**：
- **可学习位置编码**（ViT 标准）：用 `Param<Tensor<B, 2>>` 存储 `[num_patches+1, d_model]` 的可学习嵌入，与补丁嵌入相加
- **无位置编码**：`Unfold4d` 的滑动窗口天然保留了空间位置信息，浅层可省略位置编码
- **RotaryEncoding**（§4.10）：用于 ViT 变体中的注意力层位置编码

---

### 5.5 编码器-解码器架构

编码器-解码器架构用于序列到序列任务，如机器翻译、图像分割。

#### 入门：Seq2Seq Transformer

**编码器-解码器 Transformer**：
```rust
use burn::nn::transformer::{TransformerEncoder, TransformerDecoder, TransformerDecoderConfig};

let encoder = TransformerEncoderConfig::new(512, 2048, 8, 6).init(&device);
let decoder = TransformerDecoderConfig::new(512, 2048, 8, 6).init(&device);

// 编码输入序列
let memory = encoder.forward(src_embeddings, src_mask);

// 解码生成输出序列  
let output = decoder.forward(tgt_embeddings, memory, tgt_mask, memory_mask);
```

#### 进阶：解码器自回归生成

**自回归解码**：
```rust
impl<B: Backend> TransformerDecoder<B> {
    fn generate(&self, memory: Tensor<B, 3>, max_len: usize) -> Tensor<B, 2> {
        let device = memory.device();
        let [batch_size, _, d_model] = memory.dims();
        
        // 初始化为起始令牌
        let mut output_tokens = Tensor::<B, 2, Int>::zeros([batch_size, 1], &device);
        
        for step in 0..max_len {
            // 获取当前输出嵌入
            let tgt_embeddings = self.embedding.forward(output_tokens.clone());
            
            // 生成因果掩码
            let tgt_mask = generate_autoregressive_mask(batch_size, step + 1, &device);
            
            // 解码一步
            let decoder_output = self.forward(tgt_embeddings, memory.clone(), Some(tgt_mask), None);
            
            // 预测下一个令牌（贪婪解码）
            let next_token_logits = decoder_output.slice([0.., step..step+1, 0..]);
            let next_token = next_token_logits.argmax(2);
            
            // 添加到序列
            output_tokens = Tensor::cat(vec![output_tokens, next_token], 1);
        }
        
        output_tokens
    }
}
```

#### 专家：编码器-解码器变体

**架构变体**：
- **原始 Transformer**：6 层编码器 + 6 层解码器
- **BERT**：仅编码器，用于理解任务
- **GPT**：仅解码器，用于生成任务
- **T5**：编码器-解码器，统一文本到文本框架

**注意力掩码类型**：
1. **编码器掩码**：填充掩码，忽略填充令牌
2. **解码器自注意力掩码**：因果掩码，防止信息泄漏
3. **解码器-编码器注意力掩码**：通常与编码器掩码相同

**训练技巧**：
- **教师强制**：训练时使用真实目标作为解码器输入
- **计划采样**：逐步从教师强制过渡到自回归生成
- **波束搜索**：推理时保持多个候选序列

---

### 5.6 现代架构趋势

#### 入门：高效架构设计

**混合架构**：结合 CNN 的局部性优势和 Transformer 的全局建模能力。

**轻量级架构**：通过深度可分离卷积、注意力优化等技术减少计算开销。

#### 进阶：架构搜索与自动化

**神经架构搜索**（NAS）：自动化寻找最优架构。

**Burn 中的可配置架构**：
```rust
#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = "resnet18")]
    pub arch: Architecture,  // 架构选择
    
    #[config(default = 224)]
    pub input_size: usize,   // 输入尺寸
    
    #[config(default = 1000)]
    pub num_classes: usize,  // 输出类别数
    
    #[config(default = true)]
    pub pretrained: bool,    // 是否使用预训练权重
}

pub enum Architecture {
    ResNet18,
    ResNet50,
    ViTBase16,
    SwinTiny,
    EfficientNetB0,
}
```

#### 专家：可扩展性与蒸馏

**通过 `#[derive(Config)]` 实现架构缩放**：Burn 的 Config 模式天然支持参数化模型缩放——层数、通道数、头数均通过配置字段控制：

```rust
#[derive(Config)]
pub struct ScaledTransformerConfig {
    #[config(default = "6")]
    pub n_layers: usize,       // 控制深度
    #[config(default = "512")]
    pub d_model: usize,        // 控制宽度
    #[config(default = "8")]
    pub n_heads: usize,        // 控制注意力并行度
    #[config(default = "224")]
    pub image_size: usize,     // 控制输入分辨率
}
```

一个配置结构体即可覆盖从 `Tiny` 到 `Large` 的全系列模型。实际工程中，Burn 的 `TransformerEncoderConfig` 正是通过 `n_layers`、`d_model`、`n_heads`、`d_ff` 四个参数控制模型规模。

**知识蒸馏**——用小模型学习大模型输出分布：

```rust
// 使用 Burn 内置的损失模块：KLDivLoss + CrossEntropyLoss
use burn::nn::loss::{KLDivLoss, KLDivLossConfig, CrossEntropyLoss, CrossEntropyLossConfig, Reduction};
use burn::tensor::activation;

fn distillation_loss<B: Backend>(
    student_logits: Tensor<B, 2>,
    teacher_logits: Tensor<B, 2>,
    targets: Tensor<B, 1, Int>,
    temperature: f64,
    alpha: f64,
) -> Tensor<B, 1> {
    // 软目标：KL(softmax(teacher/T) || softmax(student/T)) * T²
    let soft_teacher = activation::softmax(teacher_logits.div_scalar(temperature), 1);
    let log_soft_student = activation::log_softmax(student_logits.clone().div_scalar(temperature), 1);
    let kl_loss = KLDivLossConfig { log_target: false }.init();
    let soft_loss = kl_loss.forward(log_soft_student, soft_teacher, Reduction::BatchMean)
        .mul_scalar(temperature * temperature);

    // 硬目标：标准交叉熵
    let ce_loss = CrossEntropyLossConfig::new().init(&student_logits.device());
    let hard_loss = ce_loss.forward(student_logits, targets);

    soft_loss.mul_scalar(alpha).add(hard_loss.mul_scalar(1.0 - alpha))
}
```

蒸馏时通常冻结教师模型（`teacher = teacher.no_grad()`），仅更新学生模型参数。

---

## 6. 实践指南

### 6.1 构建自定义层

1. **定义配置结构体**：使用 `#[derive(Config)]`
2. **实现层结构体**：使用 `#[derive(Module)]`
3. **实现初始化方法**：`Config::init()`
4. **实现前向传播**：`forward()` 方法
5. **添加测试**：验证正确性和数值稳定性

### 6.2 组合复杂架构

**模式匹配架构构建**：
```rust
fn build_model<B: Backend>(config: ModelConfig, device: &B::Device) -> impl Module<B> {
    match config.arch {
        Architecture::ResNet18 => ResNet::<B>::resnet18(config.num_classes, device),
        Architecture::ResNet50 => ResNet::<B>::resnet50(config.num_classes, device),
        Architecture::ViTBase16 => VisionTransformer::<B>::vit_base_16(config.num_classes, device),
        Architecture::SwinTiny => SwinTransformer::<B>::swin_tiny(config.num_classes, device),
    }
}
```

### 6.3 迁移学习与微调

**加载预训练权重**：
```rust
let pretrained = CompactRecorder::new()
    .load::<ResNetRecord<B>>("resnet50.bin", &device)
    .unwrap();

let mut model = ResNetConfig::new(1000).init(&device);
model = model.load_record(pretrained);

// 替换分类头
model.fc = LinearConfig::new(2048, num_classes).init(&device);
```

### 6.4 性能优化技巧

**激活检查点**：
```rust
use burn::backend::{Autodiff, BalancedCheckpointing};

type Backend = Autodiff<Wgpu, BalancedCheckpointing>;  // 平衡内存与计算
```

**混合精度训练**：
```rust
use burn::backend::{Autodiff, mixed_precision::MixedPrecisionBackend};

type Backend = Autodiff<MixedPrecisionBackend<Wgpu>>;
```

---

## 7. 总结与学习建议

### 7.1 关键收获

1. **模块化设计**：Burn 的 `#[derive(Module)]` 实现清晰的层抽象
2. **架构多样性**：从简单 MLP 到复杂 Transformer 的统一实现
3. **性能优化**：通过后端抽象支持多种硬件和优化策略
4. **可扩展性**：易于实现新层和组合新架构

### 7.2 学习路径建议

- **初学者**：从 MLP 和简单 CNN 开始，掌握基本层 API
- **中级开发者**：深入理解残差连接、注意力机制实现细节
- **高级开发者**：研究数值稳定性、性能优化和自定义后端开发

### 7.3 扩展学习资源

1. **Burn 官方文档**：https://burn.dev/docs/burn
2. **模型实现示例**：`burn/examples/` 目录
3. **预训练模型**：Hugging Face 集成和模型库
4. **性能基准**：不同后端和架构的基准测试

---

**相关文档**：
- [burn_foundations.md](burn_foundations.md)：训练基础与优化算法
- [4_deepmodels.md](../4_deepmodels.md)：原书第四章"模型组件"
- [5_deepmodels.md](../5_deepmodels.md)：原书第五章"架构"