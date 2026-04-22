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

#### 专家：Visitor/Mapper 模式与参数管理

基础定义可参考 `burn_foundations.md` 对 `ModuleVisitor` / `ModuleMapper` 的说明；这里聚焦训练时的关键路径：`visit()` 如何收集梯度，`map()` 如何应用梯度更新参数。

**1. 叶节点行为：`Param<Tensor<...>>` 是递归终点**

**Burn 源代码**：`crates/burn-core/src/module/param/tensor.rs`

```rust
impl<const D: usize, B: Backend> Module<B> for Param<Tensor<B, D>> {
    fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V) {
        visitor.visit_float(self)
    }

    fn map<M: ModuleMapper<B>>(self, mapper: &mut M) -> Self {
        mapper.map_float(self)
    }
}
```

每个可训练参数都被包装为 `Param<T>`，并带有唯一 `ParamId`（见 `crates/burn-core/src/module/param/base.rs`）。优化器后续正是通过 `ParamId` 在“参数 ↔ 梯度”之间做精确匹配。

**2. `#[derive(Module)]` 自动生成结构体级递归遍历**

**Burn 源代码**：`crates/burn-derive/src/module/codegen_struct.rs`

```rust
// 生成 visit: enter -> visit 子字段 -> exit
visitor.enter_module(name, container_type);
burn::module::Module::visit(&self.field, visitor);
visitor.exit_module(name, container_type);

// 生成 map: enter -> map 子字段 -> exit -> 重建 Self
mapper.enter_module(name, container_type);
let field = burn::module::Module::map(self.field, mapper);
mapper.exit_module(name, container_type);
Self { field, ... }
```

因此不需要手动枚举每一层参数，模型树会自动递归下钻到每个 `Param` 叶子节点。

**3. 梯度收集链路（`visit`）**

从训练循环看，关键三步是：

```rust
let grads = loss.backward();
let grads = GradientsParams::from_grads(grads, &model);
model = optim.step(lr, model, grads);
```

对应源码链路如下：

1. `loss.backward()` 产生自动微分后端的梯度表。
2. `GradientsParams::from_grads(...)` 调用 `module.visit(...)`（`crates/burn-optim/src/optim/grads.rs`）。
3. `GradientsParamsConverter::visit_float(...)` 在每个参数处执行：
   - `param.val().grad_remove(self.grads)` 取出该参数对应梯度；
   - `register(param.id, grad)` 写入 `GradientsParams`（键为 `ParamId`）。

**4. 梯度应用链路（`map`）**

`optim.step(lr, model, grads)` 内部会创建 `SimpleOptimizerMapper` 并调用 `module.map(&mut mapper)`（`crates/burn-optim/src/optim/simple/adaptor.rs`）。

`map_float` 的核心流程：

- `param.consume()` 取出 `(id, tensor, mapper)`；
- `grads.remove(id)` 按 `ParamId` 取梯度；
- `optimizer.step(lr, tensor.inner(), grad, state)` 计算新参数；
- `Param::from_mapped_value(id, tensor, mapper)` 重建参数节点。

这意味着 `map()` 不是“原地改值”，而是“遍历并重建”整个模块树，最终返回一个参数已更新的新模型实例。

**流程总览**：
`backward()` → `from_grads()` + `visit()` 收集梯度 → `optim.step()` + `map()` 应用梯度。

这一机制让优化器实现只需关心“单个参数如何更新”，而参数树遍历与重建由 Module 系统统一处理。 

**参数包装实现**：`crates/burn-core/src/module/param/base.rs`

```rust
pub struct Param<T: Parameter> {
    pub id: ParamId,                       // 唯一标识符
    state: SyncOnceCell<T>,                // 延迟初始化的张量值
    initialization: Option<...>,           // 惰性初始化配置
    param_mapper: ParamMapper<T>,          // load/save 变换器
    pub(crate) require_grad: bool,         // 是否需要梯度
}
```

`ParamId` 是优化器匹配梯度的核心键，`require_grad` 则控制该参数是否参与梯度计算与更新。 

**最小端到端示例来源**：`burn/examples/custom-training-loop/src/lib.rs`

```rust
let grads = loss.backward();
let grads = GradientsParams::from_grads(grads, &model);
model = optim.step(config.lr, model, grads);
```

这三行正好对应上述“收集（visit）→ 应用（map）”的完整路径。 


---

### 4.2 线性层（Linear Layers）

线性层（实际为仿射变换）是深度学习中最基本的模块，实现 $Y = XW + b$ 的变换。

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

**数学公式**：
对于输入 $X \in \mathbb{R}^{B \times D_{\text{in}}}$，权重 $W \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$，偏置 $b \in \mathbb{R}^{D_{\text{out}}}$：
$$
Y = XW + b
$$

**Burn 源代码**：`crates/burn-nn/src/modules/linear.rs`

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

// 底层线性变换实现 (crates/burn-tensor/src/tensor/module/linear.rs)
pub fn linear<B: Backend, const D: usize>(
    input: Tensor<B, D>,
    weight: Tensor<B, 2>,
    bias: Option<Tensor<B, 1>>,
) -> Tensor<B, D> {
    let output = input.matmul(weight);
    match bias {
        Some(bias) => output + bias.unsqueeze_dim(0),
        None => output,
    }
}
```

#### 专家：初始化策略、梯度计算与性能优化

**1. 初始化策略的数学原理**

线性层的初始化策略旨在防止训练初期的梯度消失或爆炸问题。Burn 通过 `Initializer` 枚举提供多种初始化方法，每种对应不同的数学假设：

```rust
pub enum Initializer {
    Constant { value: f64 },                     // 常数初始化
    Uniform { min: f64, max: f64 },             // 均匀分布 U(a,b)
    Normal { mean: f64, std: f64 },             // 正态分布 N(μ,σ²)
    KaimingUniform { gain: f64, fan_out_only: bool },  // He 初始化（均匀）
    KaimingNormal { gain: f64, fan_out_only: bool },   // He 初始化（正态）
    XavierUniform { gain: f64 },                // Xavier/Glorot 初始化（均匀）
    XavierNormal { gain: f64 },                 // Xavier/Glorot 初始化（正态）
}
```

**关键初始化公式**：
- **Xavier/Glorot 初始化**：$\sigma = \text{gain} \times \sqrt{\frac{2}{n_{\text{in}} + n_{\text{out}}}}$，适用于 tanh/sigmoid 激活
- **Kaiming/He 初始化**：$\sigma = \text{gain} \times \sqrt{\frac{2}{n_{\text{in}}}}$，适用于 ReLU 及其变体

**2. 自动微分与梯度计算**

线性层 $Y = XW + b$ 的前向传播简单，但反向传播需要计算三个梯度：
- $\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} W^\top$（输入梯度）
- $\frac{\partial L}{\partial W} = X^\top \frac{\partial L}{\partial Y}$（权重梯度）
- $\frac{\partial L}{\partial b} = \sum_{\text{batch}} \frac{\partial L}{\partial Y}$（偏置梯度）

Burn 的自动微分后端通过 `matmul` 操作的梯度规则实现这些计算。`Tensor::matmul()` 在 `Autodiff<B>` 后端下会自动注册反向传播函数，计算上述梯度。

**3. 性能优化与底层实现**

线性层的性能关键在矩阵乘法（GEMM）优化。Burn 的后端系统允许不同硬件平台实现优化的 `matmul`：

- **CPU 后端**：使用 BLAS 库（如 OpenBLAS、Intel MKL）的 `sgemm`/`dgemm` 函数
- **GPU 后端**：调用 cuBLAS（NVIDIA）或 rocBLAS（AMD）的批处理 GEMM
- **WASM 后端**：使用 SIMD 优化的 JavaScript 实现

**内存布局优化**：`LinearLayout` 枚举支持行优先（C 风格）和列优先（Fortran 风格）存储：
```rust
pub enum LinearLayout {
    Row,  // 内存连续存储行元素
    Col,  // 内存连续存储列元素
}
```

选择合适布局可以避免转置开销，直接匹配后端 BLAS 库的期望格式。

**4. 数值稳定性考量**

虽然线性层本身没有 softmax 那样的数值溢出风险，但大规模矩阵乘法仍需要注意：

- **混合精度训练**：使用 `f16` 或 `bf16` 可减少内存占用和加速计算，但需注意精度损失和溢出风险
- **梯度裁剪**：大权重矩阵可能导致梯度爆炸，需在优化器中设置梯度裁剪
- **条件数**：权重矩阵的条件数 $\kappa(W) = \|W\| \cdot \|W^{-1}\|$ 影响数值稳定性，正交初始化有助于保持条件数接近 1

**5. 与 §3.3 的联系**

线性层作为最基本的可训练模块，其梯度计算直接体现了 §3.3 中 SGD 的核心思想：$\theta \leftarrow \theta - \eta \nabla_\theta L$。在 Burn 中，这一过程分成两个阶段：先由 `ModuleVisitor` 通过 `visit()` 遍历线性层的 `Param<Weight>` 与 `Param<Bias>` 收集梯度（`GradientsParams`，键为 `ParamId`），再由 `ModuleMapper` 通过 `map()` 按同一 `ParamId` 应用更新并重建参数节点。这样优化器既不需要手动列举每层参数，也能保证梯度和参数一一对应。

> 详细调用链（`backward() -> from_grads() -> visit() -> optim.step() -> map()`）见 [§4.1 层的概念（专家）](#41-层的概念the-concept-of-layers)。

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

**Burn 源代码**：`crates/burn-nn/src/modules/conv/base.rs`

```rust
impl<B: Backend, const D: usize> Conv<B, D> {
    pub fn forward<const D2: usize>(&self, input: Tensor<B, D2>) -> Tensor<B, D2> {
        conv::convolution(
            input,
            self.weight.val(),
            self.bias.as_ref().map(|b| b.val()),
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
    }
}
```

#### 专家：卷积算法、梯度计算与性能优化

**1. 卷积算法的实现策略**

卷积操作的计算复杂度高（$O(B \cdot C_{\text{in}} \cdot C_{\text{out}} \cdot H \cdot W \cdot K \cdot L)$），因此 Burn 后端采用多种优化策略：

- **im2col + GEMM**：将卷积展开为大型矩阵乘法，利用 BLAS 库的优化 GEMM 函数
  ```rust
  // 概念性伪代码
  let im2col = input.im2col(kernel_size, stride, padding, dilation);
  let output = im2col.matmul(weight.reshape([C_out, C_in * K * L]));
  ```
- **Winograd 算法**：对小型核（3×3）使用 Winograd 变换减少乘法次数
- **FFT 卷积**：对大型核使用快速傅里叶变换，将时域卷积转为频域点乘

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

**4. 性能优化技术**

- **内存布局**：NHWC（通道最后）与 NCHW（通道优先）布局对 GPU/CPU 性能影响不同
- **核融合**：将卷积 + 批归一化 + 激活函数融合为单个 GPU 核函数，减少内存传输
- **张量核心**：利用 NVIDIA Tensor Cores 进行混合精度（fp16/bf16）卷积
- **分组卷积优化**：对 `groups > 1` 使用专门的核函数避免冗余计算

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

**Burn 源代码**：`crates/burn-nn/src/modules/pool/`

```rust
// 最大池化实现（简化）
pub fn max_pool2d<B: Backend>(
    x: Tensor<B, 4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
) -> Tensor<B, 4> {
    // 使用滑动窗口提取局部块，计算最大值
    x.unfold(kernel_size, stride, padding, dilation)
     .max_dim(2)  // 沿空间维度取最大值
     .max_dim(3)
}
```

#### 专家：池化策略与梯度传播

**梯度传播特性**：
- **最大池化**：梯度仅流向最大值所在位置
- **平均池化**：梯度均匀分配到所有输入位置

**自适应池化**：`AdaptiveAvgPool2d` 自动调整核大小以获得指定输出尺寸。

---

### 4.5 Dropout 层

Dropout 是一种正则化技术，通过随机丢弃激活防止过拟合和促进独立特征学习。

#### 入门：API 用法

**Burn 实现**：`burn::nn::Dropout`

```rust
use burn::nn::Dropout;
use burn::prelude::*;

let dropout = Dropout::new(0.5);  // 丢弃概率 50%
let device = Default::default();

// 训练模式
let input = Tensor::<Backend, 2>::random([32, 128], Distribution::Default, &device);
let output_train = dropout.forward(input.clone());

// 评估模式
dropout.eval();
let output_eval = dropout.forward(input);
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
impl<B: Backend> Dropout<B> {
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        if !self.is_active {
            return input;  // 评估模式：直接返回输入
        }
        
        let mask = Tensor::<B, D, Bool>::random(
            input.shape(),
            Distribution::Bernoulli(self.prob),
            &input.device(),
        );
        
        input.mask_fill(mask.clone(), 0.0) / (1.0 - self.prob)
    }
}
```

#### 专家：Dropout 变体与空间 Dropout

**Dropout 变体**：
- **标准 Dropout**：独立丢弃每个激活
- **空间 Dropout**（Spatial Dropout）：丢弃整个特征图（通道），适用于卷积层
- **Dropout2d**：`burn::nn::Dropout2d` 实现空间 Dropout

**训练/评估模式切换**：
```rust
impl<B: Backend> Dropout<B> {
    pub fn train(&mut self) { self.is_active = true; }
    pub fn eval(&mut self) { self.is_active = false; }
}
```

---

### 4.6 归一化层（Normalization Layers）

归一化层通过标准化激活的分布来稳定训练过程。

#### 入门：API 用法

**Batch Normalization**：
```rust
use burn::nn::norm::{BatchNorm1d, BatchNorm1dConfig};
use burn::prelude::*;

let device = Default::default();
let bn = BatchNorm1dConfig::new(128)  // 特征维度 128
    .with_eps(1e-5)                   // 数值稳定性常数
    .with_momentum(0.1)               // 移动平均动量
    .init(&device);

let input = Tensor::<Backend, 2>::random([32, 128], Distribution::Default, &device);
let output = bn.forward(input);
```

**Layer Normalization**：
```rust
use burn::nn::norm::{LayerNorm, LayerNormConfig};

let ln = LayerNormConfig::new(128)  // 特征维度 128
    .with_eps(1e-5)
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

**Burn 源代码**：`crates/burn-nn/src/modules/norm/`

```rust
// BatchNorm 前向传播（简化）
pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
    if self.training {
        // 训练模式：计算批次统计量，更新移动平均
        let mean = input.mean_dim(D - 1);
        let var = input.var_dim(D - 1, self.eps);
        
        // 更新移动统计量
        self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * mean;
        self.running_var = self.momentum * self.running_var + (1.0 - self.momentum) * var;
        
        // 归一化并缩放
        let normalized = (input - mean) / (var + self.eps).sqrt();
        normalized * self.weight + self.bias
    } else {
        // 评估模式：使用移动统计量
        let normalized = (input - self.running_mean) / (self.running_var + self.eps).sqrt();
        normalized * self.weight + self.bias
    }
}
```

#### 专家：归一化变体与应用场景

**归一化层对比**：
| 类型 | 归一化维度 | 适用场景 | Burn 实现 |
|------|-----------|----------|-----------|
| **BatchNorm** | 批次维度 | 卷积网络、大批次训练 | `BatchNorm1d/2d/3d` |
| **LayerNorm** | 特征维度 | Transformer、RNN、小批次 | `LayerNorm` |
| **InstanceNorm** | 空间维度 | 风格迁移、生成模型 | `InstanceNorm1d/2d/3d` |
| **GroupNorm** | 分组特征 | 小批次训练 | `GroupNorm` |

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

**残差连接变体**：
- **Pre-activation**：归一化和激活在卷积之前（ResNet v2）
- **Bottleneck**：使用 1x1 卷积减少和恢复维度
- **Dense connections**：连接所有前序层（DenseNet）

**初始化策略**：残差块的最后一层通常使用零初始化，使网络初始状态接近恒等映射。

**梯度流动**：
$$
\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \left(1 + \frac{\partial \mathcal{F}}{\partial x}\right)
$$

即使 $\frac{\partial \mathcal{F}}{\partial x} \approx 0$，梯度仍可通过 $1$ 项回传。

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

- **FlashAttention** [Dao et al., 2022]：通过分块计算和重计算避免存储 $n \times n$ 注意力矩阵
  ```rust
  // 概念性伪代码
  fn flash_attention(q, k, v, block_size) {
      for i in 0..n/block_size {
          for j in 0..n/block_size {
              // 分块计算注意力分数
              let scores = q_block[i].matmul(k_block[j].t());
              // 在线 softmax（数值稳定）
              let attn = online_softmax(scores);
              output_block[i] += attn.matmul(v_block[j]);
          }
      }
  }
  ```
- **内存高效注意力**：通过重计算中间结果减少内存占用
- **线性注意力**：使用核函数近似，复杂度 $O(nd^2)$
  $$ \text{LinearAttn}(Q, K, V) = \phi(Q)(\phi(K)^\top V) $$
  其中 $\phi$ 为特征映射（如 $\phi(x) = \text{elu}(x) + 1$）

**4. 注意力变体与掩码机制**

- **因果注意力**（自回归）：使用下三角布尔掩码 $M_{ij} = \begin{cases} 0 & i \geq j \\ -\infty & i < j \end{cases}$
- **局部注意力**：滑动窗口掩码 $M_{ij} = \begin{cases} 0 & |i-j| \leq w \\ -\infty & \text{否则} \end{cases}$
- **跨步注意力**：稀疏模式减少计算，如 BigBird 的全局+局部+随机注意力

**5. 数值稳定性考量**

注意力层的数值稳定性关键在 softmax：
- **在线 softmax**：分块计算时需维护 running max 和 sum 保证数值稳定
- **混合精度**：使用 fp16 计算但 fp32 累加器，防止下溢
- **梯度裁剪**：注意力分数可能产生极大梯度，需裁剪

**6. 多头注意力的并行化**

- **张量并行**：将头分配到不同设备，通过 all-gather 通信合并结果
- **序列并行**：将序列分块分配到不同设备，需要跨设备通信注意力分数
- **流水线并行**：将注意力层分配到不同设备，需要激活 checkpointing

**7. 与 §4.7 跳跃连接的联系**

Transformer 块中，注意力层通常与残差连接（§4.7）和层归一化（§4.6）结合：
$$
\text{Output} = \text{LayerNorm}(x + \text{Attention}(x))
$$
残差连接确保梯度直接回传，缓解梯度消失。

**8. 计算复杂度分析**

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
impl<B: Backend> RopeEncoding<B> {
    pub fn forward(&self, x: Tensor<B, 3>, positions: Tensor<B, 1, Int>) -> Tensor<B, 3> {
        // 应用旋转矩阵到查询和键
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
    flatten: Flatten,
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
        
        // 分类器
        let x = self.flatten.forward(x);
        let x = self.fc1.forward(x);
        let x = burn::tensor::activation::relu(x);
        self.fc2.forward(x)
    }
}
```

#### 进阶：残差网络（ResNet）

**残差块实现**：
```rust
use burn::nn::{Conv2d, Conv2dConfig, BatchNorm2d, BatchNorm2dConfig};

#[derive(Module, Debug)]
struct ResidualBlock<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm2d<B>,
    conv2: Conv2d<B>,
    bn2: BatchNorm2d<B>,
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
    feed_forward: FeedForward<B>,
    norm1: LayerNorm<B>,
    norm2: LayerNorm<B>,
    dropout: Dropout<B>,
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

前馈网络（FFN）是 Transformer 的关键组件，提供非线性变换：
```rust
#[derive(Module, Debug)]
struct FeedForward<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    dropout: Dropout<B>,
}

impl<B: Backend> FeedForward<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.linear1.forward(x);
        let x = burn::tensor::activation::gelu(x);  // GPT 使用 GELU
        let x = self.dropout.forward(x);
        self.linear2.forward(x)
    }
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

**5. Transformer 训练技巧**

- **学习率预热**：前 $w$ 步线性增加学习率至初始值，稳定训练初期
  $$ \eta_t = \frac{t}{w} \cdot \eta_{\text{max}}, \quad t < w $$
- **权重衰减**：L2 正则化防止过拟合，通常设置为 0.01-0.1
- **梯度裁剪**：限制梯度范数防止梯度爆炸
  $$ g \leftarrow g \cdot \frac{\text{clip\_norm}}{\max(\|g\|, \text{clip\_norm})} $$
- **标签平滑**：将 one-hot 标签软化，防止模型过度自信

**6. 缩放定律与模型扩展**

Kaplan et al. (2020) 提出 Transformer 的幂律缩放：
$$ L(N, D) \approx \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D} $$
其中 $N$ 为参数数，$D$ 为数据量，$\alpha_N \approx 0.076$, $\alpha_D \approx 0.095$。

**7. 并行化策略**

大规模 Transformer 需要多设备并行：

- **数据并行**：不同设备处理不同批次，同步梯度（all-reduce）
- **张量并行**：将权重矩阵分块到不同设备，需要通信激活和梯度
- **流水线并行**：将层分配到不同设备，需要流水线气泡和激活 checkpointing
- **序列并行**：将序列分块分配到不同设备，需要注意力通信

**8. 推理优化技术**

- **键值缓存**：自回归生成时缓存先前时间步的 K, V，避免重复计算
- **量化**：将权重/激活从 fp32 降至 int8/int4，减少内存和计算
- **蒸馏**：用小模型学习大模型输出分布，保持性能减少计算
- **剪枝**：移除不重要的权重/注意力头，创建稀疏模型

**9. 数值稳定性考量**

- **深度 Transformer**：超过 100 层时需使用 Pre-LN 而非 Post-LN 保持稳定
- **混合精度训练**：使用 fp16 但维护 fp32 主副本防止梯度下溢
- **激活 checkpointing**：仅存储部分层激活，反向传播时重计算其余层

**10. 与 §5.2 卷积神经网络的对比**

| 特性 | CNN | Transformer |
|------|-----|-------------|
| **归纳偏置** | 局部性、平移等变 | 序列建模、全局依赖 |
| **计算复杂度** | $O(n)$ | $O(n^2)$（注意力） |
| **并行性** | 高度并行 | 序列依赖（解码器） |
| **数据需求** | 较少 | 大量 |

**11. Burn 中的 Transformer 实现**

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

**ViT 变体**：
- **DeiT**：数据高效图像 Transformer，使用蒸馏令牌
- **Swin Transformer**：分层移位窗口，线性计算复杂度
- **MobileViT**：移动设备优化，结合 CNN 和 Transformer

**混合架构**：
```rust
// CNN + Transformer 混合
struct HybridViT<B: Backend> {
    cnn_backbone: CNN<B>,           // 提取局部特征
    transformer: TransformerEncoder<B>, // 建模全局关系
    head: Linear<B>,
}
```

**位置编码策略**：
- **可学习位置编码**：ViT-B/16 标准配置
- **相对位置编码**：Swin Transformer
- **无位置编码**：某些研究表明可能不需要显式位置编码

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

**模型缩放定律**：
- **宽度缩放**：增加通道数
- **深度缩放**：增加层数  
- **分辨率缩放**：增加输入尺寸

**知识蒸馏**：
```rust
struct DistillationLoss {
    temperature: f32,
    alpha: f32,  // 蒸馏损失权重
}

impl DistillationLoss {
    fn forward(
        &self,
        student_logits: Tensor<B, 2>,
        teacher_logits: Tensor<B, 2>,
        targets: Tensor<B, 1, Int>,
    ) -> Tensor<B, 1> {
        // 软目标损失（蒸馏）
        let soft_loss = kl_divergence(
            student_logits.div_scalar(self.temperature).softmax(1),
            teacher_logits.div_scalar(self.temperature).softmax(1),
        ) * (self.temperature * self.temperature);
        
        // 硬目标损失（标准交叉熵）
        let hard_loss = cross_entropy_loss(student_logits, targets);
        
        soft_loss * self.alpha + hard_loss * (1.0 - self.alpha)
    }
}
```

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