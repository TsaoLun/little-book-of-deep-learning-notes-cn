# Burn框架中《深度学习小书》第三章渐进式分析

## 引言

《深度学习小书》（Little Book of Deep Learning）第三章"训练"涵盖了深度学习的核心训练概念，包括损失函数、优化算法、反向传播和训练协议等。本分析文档旨在提供一个渐进式的学习路径，将第三章的理论知识点与 Rust 深度学习框架 [Burn](https://github.com/tracel-ai/burn) 的代码实现进行对应。

本文档分为三个层次，适合不同阶段的学习者：

1. **第一部分：知识点到API映射（入门级）** - 适合初学者，展示如何通过 Burn 的高级 API 实现第三章的概念
2. **第二部分：公式与源代码对应（进阶级）** - 适合中级学习者，展示数学公式如何直接转换为 Burn 源代码
3. **第三部分：深入实现分析（专家级）** - 适合高级学习者和框架开发者，深入分析数值稳定性、梯度计算等底层实现细节

通过这三个层次的分析，读者可以逐步深入理解深度学习理论到工程实践的完整转换过程。

---

## 第一部分：知识点到API映射（入门级）

本部分基于文档 [burn_chapter3_mapping.md](../burn_chapter3_mapping.md)，展示如何通过 Burn 的高级 API 实现第三章的核心概念。适合初次接触 Burn 框架或希望快速上手的开发者。

### 1. 损失函数（Loss Functions）

#### 1.1 交叉熵损失（Cross-Entropy Loss）

交叉熵是分类任务的标准损失函数，用于衡量模型预测概率分布与真实分布之间的差异。

**Burn 实现**：`burn_nn::loss::CrossEntropyLoss`

```rust
use burn::prelude::*;
use burn_nn::loss::CrossEntropyLossConfig;

// 创建交叉熵损失函数
let device = Default::default();
let loss = CrossEntropyLossConfig::new()
    .with_smoothing(Some(0.1)) // 标签平滑
    .with_weights(Some(vec![1.0, 2.0, 3.0])) // 类别权重
    .init(&device);

// 使用示例
let logits = Tensor::random([batch_size, num_classes], Distribution::Normal(0., 1.), &device);
let targets = Tensor::from_ints([0, 2, 1], &device); // 类别索引
let loss_value = loss.forward(logits, targets);
```

**关键特性**：
- 支持标签平滑（label smoothing）
- 支持类别权重（class weights）
- 支持填充令牌忽略（pad tokens）
- 可选择输入为 logits 或概率

#### 1.2 对比损失（Contrastive Loss）

对比损失用于度量学习，使相似样本在嵌入空间中更接近，不相似样本更远离。

**Burn 实现**：`burn_nn::loss::CosineEmbeddingLoss`

```rust
use burn_nn::loss::CosineEmbeddingLossConfig;

let loss = CosineEmbeddingLossConfig::new()
    .with_margin(0.5) // 边界值
    .init();

let input1 = Tensor::random([batch_size, embedding_dim], Distribution::Default, &device);
let input2 = Tensor::random([batch_size, embedding_dim], Distribution::Default, &device);
let target = Tensor::from_ints([1, -1, 1, -1], &device); // 1表示相似，-1表示不相似

let loss_value = loss.forward(input1, input2, target);
```

#### 1.3 其他损失函数

Burn 还提供了多种常见损失函数：

- **均方误差（MSE）**：`burn_nn::loss::MseLoss`
- **平均绝对误差（MAE/L1）**：`burn_nn::loss::LpLoss`
- **Huber 损失**：`burn_nn::loss::HuberLoss`
- **KL 散度**：`burn_nn::loss::KlDivLoss`
- **连接主义时间分类（CTC）**：`burn_nn::loss::CtcLoss`

### 2. 自回归模型（Autoregressive Models）

自回归模型通过链式法则分解序列数据的联合概率，常用于语言建模和序列生成。

#### 2.1 因果模型（Causal Models）

因果模型确保每个时间步的预测仅依赖于之前的输入，不泄露未来信息。

**Burn 实现**：通过 Transformer 的因果注意力掩码实现

```rust
use burn_nn::transformer::TransformerEncoderConfig;
use burn_nn::attention::CausalMask;

// 创建因果 Transformer 编码器
let transformer = TransformerEncoderConfig::new(d_model, nhead)
    .with_causal_mask(true) // 启用因果掩码
    .init(&device);

// 对于自回归生成，可以使用自回归采样
let mut output = Vec::new();
let mut current_input = start_token;

for _ in 0..seq_len {
    let logits = transformer.forward(current_input);
    let next_token = logits.argmax(dim::<2>(-1)); // 贪婪采样
    output.push(next_token);
    current_input = next_token; // 将预测作为下一时间步的输入
}
```

#### 2.2 分词器（Tokenizers）

Burn 本身不包含分词器实现，但可以与外部分词器库（如 `tokenizers`）结合使用。

```rust
use tokenizers::Tokenizer;

let tokenizer = Tokenizer::from_pretrained("bert-base-uncased").unwrap();
let encoding = tokenizer.encode("Hello, world!", false).unwrap();
let token_ids = encoding.get_ids();
```

### 3. 梯度下降（Gradient Descent）

#### 3.1 随机梯度下降（Stochastic Gradient Descent）

**Burn 实现**：`burn_optim::Sgd`

```rust
use burn_optim::{SgdConfig, Optimizer};

let config = SgdConfig::new()
    .with_momentum(0.9) // 动量
    .with_weight_decay(0.0001) // 权重衰减
    .with_nesterov(true); // Nesterov 动量

let mut optimizer = config.init(&model);

// 训练循环中的优化步骤
let grads = loss.backward();
optimizer.step(&model, grads);
```

#### 3.2 Adam 优化器

**Burn 实现**：`burn_optim::Adam` 和 `burn_optim::AdamW`

```rust
use burn_optim::AdamWConfig;

let config = AdamWConfig::new()
    .with_betas(0.9, 0.999)
    .with_eps(1e-8)
    .with_weight_decay(0.01)
    .with_cautious_weight_decay(true);

let mut optimizer = config.init(&model);
```

#### 3.3 学习率调度（Learning Rate Scheduling）

**Burn 实现**：`burn_optim::lr_scheduler`

```rust
use burn_optim::lr_scheduler::{
    CosineAnnealingLrSchedulerConfig,
    LinearLrSchedulerConfig,
    ComposedLrSchedulerConfig
};

let scheduler = ComposedLrSchedulerConfig::new()
    .linear(LinearLrSchedulerConfig::new(1e-4, 1e-2, 1000)) // 预热
    .cosine(CosineAnnealingLrSchedulerConfig::new(1e-2, 1e-5, 10000)) // 余弦退火
    .init()
    .unwrap();

// 在每个训练步骤更新学习率
let lr = scheduler.step();
optimizer.set_learning_rate(lr);
```

### 4. 反向传播（Backpropagation）

#### 4.1 自动微分（Autodiff）

Burn 通过 `Autodiff` 后端装饰器提供自动微分功能。

```rust
use burn::backend::{Autodiff, Wgpu};
use burn::tensor::{Tensor, Distribution};

type Backend = Autodiff<Wgpu>;

let device = Default::default();
let x: Tensor<Backend, 2> = Tensor::random([32, 32], Distribution::Default, &device);
let y: Tensor<Backend, 2> = Tensor::random([32, 32], Distribution::Default, &device).require_grad();

let z = x.matmul(y);
let loss = z.mean();
let grads = loss.backward(); // 自动计算梯度
```

#### 4.2 梯度裁剪（Gradient Clipping）

防止梯度爆炸的常用技术。

```rust
use burn_optim::grad_clipping::GradientClipping;

let clipping = GradientClipping::Norm(1.0); // 梯度范数裁剪到 1.0
optimizer = optimizer.with_grad_clipping(clipping);
```

### 5. 深度的价值（The Value of Depth）

深度神经网络通过堆叠多个层来学习层次化表示。

**Burn 实现**：模块化层组合

```rust
use burn_nn::{
    Linear, LinearConfig,
    Relu,
    Sequential, 
    Conv2d, Conv2dConfig,
    BatchNorm2d, BatchNorm2dConfig
};

// 构建深度卷积网络
let model = Sequential::new()
    .add(Conv2dConfig::new([1, 32], [3, 3]).init(&device))
    .add(BatchNorm2dConfig::new(32).init(&device))
    .add(Relu::new())
    .add(Conv2dConfig::new([32, 64], [3, 3]).init(&device))
    .add(BatchNorm2dConfig::new(64).init(&device))
    .add(Relu::new())
    .add(LinearConfig::new(64 * 28 * 28, 10).init(&device));
```

### 6. 训练协议（Training Protocols）

#### 6.1 数据集分割

Burn 提供了灵活的数据集 API。

```rust
use burn::data::dataset::{
    Dataset, 
    transform::{PartialDataset, MapperDataset},
    vision::MnistDataset
};

let dataset = Arc::new(MnistDataset::train());
let train_dataset = PartialDataset::new(dataset.clone(), 0, 50000);
let valid_dataset = PartialDataset::new(dataset.clone(), 50000, 60000);
```

#### 6.2 数据加载器

```rust
use burn::data::dataloader::DataLoaderBuilder;

let dataloader = DataLoaderBuilder::new(batcher)
    .batch_size(64)
    .shuffle(42)
    .num_workers(4)
    .build(dataset);
```

#### 6.3 训练循环与评估

Burn 提供了高级训练 API `Learner`。

```rust
use burn::train::{
    Learner, 
    metric::{AccuracyMetric, LossMetric},
    renderer::MetricsRenderer
};

let learner = Learner::new(model, optimizer, lr_scheduler);

let result = learner.fit(
    train_dataloader,
    valid_dataloader,
    num_epochs,
    (AccuracyMetric::new(), LossMetric::new()),
    ARTIFACT_DIR
);
```

#### 6.4 早停（Early Stopping）

```rust
use burn::train::{
    MetricEarlyStoppingStrategy,
    StoppingCondition,
    metric::store::{Aggregate, Direction, Split}
};

let early_stopping = MetricEarlyStoppingStrategy::new(
    &LossMetric::new(),
    Aggregate::Mean,
    Direction::Lowest,
    Split::Valid,
    StoppingCondition::NoImprovementSince { n_epochs: 5 }
);
```

### 7. 规模的好处（The Benefit of Scale）

#### 7.1 多后端支持

Burn 支持多种硬件后端，便于在不同规模上训练和部署。

```rust
// CPU 后端
use burn_ndarray::NdArray;
type CpuBackend = NdArray<f32>;

// GPU 后端
use burn_wgpu::Wgpu;
type GpuBackend = Wgpu;

// CUDA 后端
use burn_cuda::Cuda;
type CudaBackend = Cuda;

// 自动微分包装
type TrainBackend = Autodiff<CudaBackend>;
```

#### 7.2 分布式训练

Burn 通过 `burn_collective` 支持分布式训练。

```rust
use burn_collective::{
    init_process_group,
    all_reduce,
    Backend as CollectiveBackend
};

init_process_group(backend, rank, world_size);
let reduced_grads = all_reduce(grads, CollectiveBackend::Nccl);
```

### 第一部分总结

本部分展示了如何通过 Burn 的高级 API 实现《深度学习小书》第三章中的核心概念。通过实际代码示例，读者可以快速上手 Burn 框架，理解如何将理论概念转化为具体的代码实现。下一部分将进一步深入，展示这些 API 背后的数学公式与源代码的直接对应关系。

---

## 第二部分：公式与源代码对应（进阶级）

本部分基于文档 [burn_chapter3_formulas.md](../burn_chapter3_formulas.md)，深入展示数学公式如何直接转换为 Burn 源代码。适合希望理解底层实现原理的中级学习者。

### 1. 损失函数的公式实现

#### 1.1 Softmax / Logits 转换

**数学公式**：
$$
\hat{P}(Y = y \mid X = x) = \frac{\exp f(x;w)_y}{\sum_z \exp f(x;w)_z}
$$

**Burn 源代码**：`crates/burn-tensor/src/tensor/activation/base.rs`

```rust
/// Applies the softmax function on the input tensor along the given dimension.
///
/// $$
/// \text{softmax}(x_i) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}
/// $$
pub fn softmax<const D: usize, B: Backend>(tensor: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    check!(TensorCheck::dim_ops::<D>("softmax", dim));

    // 数值稳定性处理：减去最大值防止指数溢出
    // x_i' = x_i - max(x)
    let tensor = tensor.clone() - tensor.detach().max_dim(dim);
    
    // 计算 exp(x_i') 和 sum(exp(x_j'))
    let tensor = tensor.exp();                    // exp(x_i')
    let tensor_tmp = tensor.clone().sum_dim(dim); // sum_j(exp(x_j'))
    
    // 返回 softmax(x_i) = exp(x_i') / sum_j(exp(x_j'))
    tensor.div(tensor_tmp)
}

/// Applies the log softmax function on the input tensor along the given dimension.
///
/// $$
/// \text{log_softmax}(x_i) = \log\left(\text{softmax}(x_i)\right)
/// = \log\left(\frac{\exp(x_i)}{\sum_j \exp(x_j)}\right)
/// $$
pub fn log_softmax<const D: usize, B: Backend>(tensor: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    check!(TensorCheck::dim_ops::<D>("log softmax", dim));

    // 数值稳定性处理：x_i' = x_i - max(x)
    let tensor = tensor.clone() - tensor.detach().max_dim(dim);
    
    // 计算 log(sum_j(exp(x_j'))) 
    let tensor_tmp = tensor.clone().exp().sum_dim(dim).log();
    
    // 返回 log_softmax(x_i) = x_i' - log(sum_j(exp(x_j')))
    // 等价于 log(exp(x_i') / sum_j(exp(x_j')))
    tensor.sub(tensor_tmp)
}
```

**代码解释**：
1. **数值稳定性**：通过减去最大值 (`max_dim`) 防止指数运算溢出
2. **softmax 实现**：$\text{softmax}(x_i) = \frac{\exp(x_i - \max(x))}{\sum_j \exp(x_j - \max(x))}$
3. **log_softmax 实现**：$\log\text{softmax}(x_i) = (x_i - \max(x)) - \log\sum_j \exp(x_j - \max(x))$
4. 使用 `detach()` 防止梯度传播到最大值计算，避免影响反向传播

#### 1.2 交叉熵损失（Cross-Entropy Loss）

**数学公式**：
$$
\mathcal{L}_{ce}(w) = -\frac{1}{N} \sum_{n=1}^{N} \log\hat{P}(Y = y_n | X = x_n)
= \frac{1}{N}\sum_{n=1}^{N}-\log\frac{\exp f(x_n;w)_{y_n}}{\sum_z \exp f(x_n;w)_z}
$$

**Burn 源代码**：`crates/burn-nn/src/loss/cross_entropy.rs`

```rust
fn forward_default(&self, logits: Tensor<B, 2>, targets: Tensor<B, 1, Int>) -> Tensor<B, 1> {
    let [batch_size] = targets.dims();

    // 1. 计算填充掩码（忽略填充令牌）
    let mask = self.padding_mask(&targets);
    
    // 2. 计算 log softmax: log(exp(x_y) / sum_z(exp(x_z)))
    // 对应公式中的 log(exp(f(x;w)_y) / sum_z exp(f(x;w)_z))
    let tensor = log_softmax(logits, 1);
    
    // 3. 提取真实类别的 log 概率
    // tensor.gather(1, targets.reshape([batch_size, 1])) 选择每个样本的真实类别 logits
    let tensor = tensor.gather(1, targets.clone().reshape([batch_size, 1]));
    
    // 4. 根据是否使用权重计算损失
    match &self.weights {
        Some(weights) => {
            // 加权交叉熵：weight * -log(p)
            let weights = weights.clone().gather(0, targets);
            let tensor = tensor.reshape([batch_size]) * weights.clone();
            let tensor = Self::apply_mask_1d(tensor, mask);
            // 对应加权平均：sum(weight * -log(p)) / sum(weights)
            tensor.sum().neg() / weights.sum()
        }
        None => {
            // 标准交叉熵：mean(-log(p))
            let tensor = Self::apply_mask_1d(tensor.reshape([batch_size]), mask);
            // 对应公式：-(1/N) * sum(log(p))
            tensor.mean().neg()
        }
    }
}
```

**代码与公式对应关系**：
| 公式部分 | 代码实现 | 说明 |
|---------|---------|------|
| $\log\hat{P}(Y = y_n \| X = x_n)$ | `log_softmax(logits, 1).gather(1, targets)` | 计算真实类别的 log 概率 |
| $-\frac{1}{N} \sum_{n=1}^{N}$ | `.mean().neg()` | 取平均后取负 |
| 加权版本 | `tensor.sum().neg() / weights.sum()` | 加权平均 |

#### 1.3 对比损失（Contrastive Loss / Cosine Embedding Loss）

**数学公式**（对比损失变种）：
对于三元组 $(x_a, x_b, x_c)$ 满足 $y_a = y_b \neq y_c$：
$$
\mathcal{L}_{\text{contrastive}} = \max(0, 1 - f(x_a, x_c; w) + f(x_a, x_b; w))
$$

**Burn 源代码**：`crates/burn-nn/src/loss/cosine_embedding.rs`

```rust
pub fn forward_no_reduction<B: Backend>(
    &self,
    input1: Tensor<B, 2>,
    input2: Tensor<B, 2>,
    target: Tensor<B, 1, Int>,
) -> Tensor<B, 1> {
    self.assertions(&input1, &input2, &target);

    // 计算余弦相似度：cos_sim = (x·y) / (‖x‖‖y‖)
    // 对应公式中的 f(x_a, x_b; w) 函数
    let cos_sim = cosine_similarity(input1, input2, 1, None);
    let cos_sim: Tensor<B, 1> = cos_sim.squeeze_dim(1);

    let mut loss = cos_sim.zeros_like();

    // 相似样本对（target == 1）：损失 = 1 - cos_sim
    // 对应最大化相似度（使 cos_sim → 1）
    let similar_mask = target.clone().equal_elem(1);
    let similar_loss = cos_sim.clone().neg().add_scalar(1);
    loss = loss.mask_where(similar_mask, similar_loss);

    // 不相似样本对（target == -1）：损失 = max(0, cos_sim - margin)
    // 对应最小化相似度（使 cos_sim < margin）
    let dissimilar_mask = target.equal_elem(-1);
    let dissimilar_loss = relu(cos_sim.clone().sub_scalar(self.margin));
    loss = loss.mask_where(dissimilar_mask, dissimilar_loss);

    loss
}
```

**代码与公式对应关系**：
| 公式部分 | 代码实现 | 说明 |
|---------|---------|------|
| $f(x_a, x_b; w)$ | `cosine_similarity(input1, input2, 1, None)` | 余弦相似度作为距离函数 |
| $1 - f(x_a, x_b; w)$ | `cos_sim.clone().neg().add_scalar(1)` | 相似样本损失 |
| $\max(0, f(x_a, x_c; w) - \text{margin})$ | `relu(cos_sim.clone().sub_scalar(self.margin))` | 不相似样本损失（带边界） |
| 样本类型选择 | `.mask_where(mask, loss)` | 根据 target 值选择相应损失 |

### 2. 梯度下降（Gradient Descent）

#### 2.1 梯度下降更新公式

**数学公式**：
$$
w_{n+1} = w_n - \eta \nabla\mathcal{L}|_w(w_n)
$$

**Burn 源代码**：`crates/burn-optim/src/optim/sgd.rs`

```rust
impl<B: Backend> SimpleOptimizer<B> for Sgd<B> {
    type State<const D: usize> = SgdState<B, D>;

    fn step<const D: usize>(
        &self,
        lr: LearningRate,           // 学习率 η
        tensor: Tensor<B, D>,       // 当前参数 w_n
        mut grad: Tensor<B, D>,     // 梯度 ∇ℒ|_w(w_n)
        state: Option<Self::State<D>>,
    ) -> (Tensor<B, D>, Option<Self::State<D>>) {
        let mut state_momentum = None;

        if let Some(state) = state {
            state_momentum = state.momentum;
        }

        // 权重衰减：grad = grad + λ * w_n
        if let Some(weight_decay) = &self.weight_decay {
            grad = weight_decay.transform(grad, tensor.clone());
        }

        // 动量更新：v_{t+1} = μ * v_t + (1 - τ) * grad
        if let Some(momentum) = &self.momentum {
            let (grad_out, state) = momentum.transform(grad, state_momentum);
            state_momentum = Some(state);
            grad = grad_out;
        }

        let state = SgdState::new(state_momentum);
        
        // 计算更新量：Δw = η * grad
        let delta = grad.mul_scalar(lr);
        
        // 参数更新：w_{n+1} = w_n - Δw
        (tensor - delta, Some(state))
    }
}
```

**代码与公式对应关系**：
| 公式部分 | 代码实现 | 说明 |
|---------|---------|------|
| $\eta \nabla\mathcal{L}\|_w(w_n)$ | `grad.mul_scalar(lr)` | 学习率乘以梯度 |
| $w_n - \eta \nabla\mathcal{L}\|_w(w_n)$ | `tensor - delta` | 参数更新 |
| 权重衰减项 | `weight_decay.transform(grad, tensor)` | L2 正则化：grad + λw |
| 动量项 | `momentum.transform(grad, state_momentum)` | 动量加速：v_{t+1} = μv_t + (1-τ)grad |

#### 2.2 动量（Momentum）实现

**数学公式**：
$$
\begin{aligned}
v_{t+1} &= \mu v_t + (1 - \tau) g_t \\
w_{t+1} &= w_t - \eta v_{t+1}
\end{aligned}
$$

**Burn 源代码**：`crates/burn-optim/src/optim/momentum.rs`

```rust
pub fn transform<B: Backend, const D: usize>(
    &self,
    grad: Tensor<B, D>,
    state: Option<MomentumState<B, D>>,
) -> (Tensor<B, D>, MomentumState<B, D>) {
    let state = match state {
        Some(state) => {
            // v_t = μ * v_{t-1} + (1 - τ) * g_t
            let velocity = state
                .velocity
                .mul_scalar(self.momentum)
                .add(grad.clone().mul_scalar(1.0 - self.dampening));
            
            MomentumState::new(velocity)
        }
        None => {
            // 初始状态：v_0 = g_0
            MomentumState::new(grad.clone())
        }
    };

    let grad = if self.nesterov {
        // Nesterov 动量：g_t' = g_t + μ * v_t
        grad.add(state.velocity.clone().mul_scalar(self.momentum))
    } else {
        // 标准动量：直接使用 v_t
        state.velocity.clone()
    };

    (grad, state)
}
```

### 3. 反向传播与自动微分

#### 3.1 链式法则实现

Burn 通过 `Autodiff` 后端装饰器自动实现反向传播的链式法则。

**数学原理**：
对于复合函数 $f = f^{(D)} \circ f^{(D-1)} \circ \cdots \circ f^{(1)}$，链式法则为：
$$
\frac{\partial \ell}{\partial x^{(d-1)}} = \frac{\partial \ell}{\partial x^{(d)}} \cdot \frac{\partial f^{(d)}}{\partial x^{(d-1)}}
$$

**Burn 使用示例**：

```rust
use burn::backend::{Autodiff, Wgpu};
use burn::tensor::{Tensor, Distribution};

type Backend = Autodiff<Wgpu>;

let device = Default::default();
let x: Tensor<Backend, 2> = Tensor::random([32, 32], Distribution::Default, &device);
let y: Tensor<Backend, 2> = Tensor::random([32, 32], Distribution::Default, &device).require_grad();

// 前向传播：计算复合函数
let z = x.matmul(y);      // 矩阵乘法
let w = z.relu();         // ReLU 激活
let loss = w.mean();      // 损失函数

// 自动反向传播：计算 ∂loss/∂x 和 ∂loss/∂y
let grads = loss.backward();
```

**实现机制**：
1. **计算图跟踪**：`Autodiff<B>` 包装后端，记录所有张量操作
2. **梯度计算**：调用 `backward()` 时，沿计算图反向传播梯度
3. **雅可比矩阵乘积**：自动计算每个操作的雅可比矩阵与上游梯度的乘积

### 4. 训练协议相关公式

#### 4.1 学习率调度：余弦退火

**数学公式**（Cosine Annealing）：
$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{T_{\text{cur}}}{T_{\max}}\pi\right)\right)
$$

**Burn 源代码**：`crates/burn-optim/src/lr_scheduler/cosine.rs`

```rust
fn step(&mut self, step: usize) -> LearningRate {
    if step >= self.t_max {
        return self.eta_min;
    }
    
    // 计算当前进度比例
    let progress = step as f64 / self.t_max as f64;
    
    // 余弦退火公式
    self.eta_min + (self.eta_max - self.eta_min) * 
        (1.0 + f64::cos(std::f64::consts::PI * progress)) / 2.0
}
```

#### 4.2 梯度裁剪：范数裁剪

**数学公式**：
$$
g_{\text{clipped}} = \begin{cases}
g & \text{if } \|g\| \leq \text{threshold} \\
\text{threshold} \cdot \frac{g}{\|g\|} & \text{otherwise}
\end{cases}
$$

**Burn 源代码**：`crates/burn-optim/src/grad_clipping.rs`

```rust
pub fn clip_gradient<B: Backend, const D: usize>(
    &self,
    gradient: Tensor<B, D>,
) -> Tensor<B, D> {
    match self {
        GradientClipping::Value(threshold) => {
            let norm = gradient.clone().powf_scalar(2.0).sum().sqrt();
            
            if norm.to_f64() <= *threshold {
                gradient
            } else {
                // 缩放梯度：g * threshold / norm
                gradient.mul_scalar(*threshold / norm.to_f64())
            }
        }
        GradientClipping::Norm(max_norm) => {
            let norm = gradient.clone().powf_scalar(2.0).sum().sqrt();
            let scale = max_norm / (norm.to_f64() + 1e-6);
            
            if scale < 1.0 {
                gradient.mul_scalar(scale)
            } else {
                gradient
            }
        }
    }
}
```

### 第二部分总结

本部分展示了深度学习数学公式到 Burn 源代码的直接对应关系。通过分析这些实现，我们可以看到：

1. **公式的直接转换**：数学公式如何逐行转换为 Rust 代码
2. **数值稳定性处理**：实际代码中如何处理数值计算问题
3. **框架设计模式**：Burn 如何组织不同组件的代码结构

下一部分将进一步深入，分析这些实现背后的设计原理和优化技巧。

---

## 第三部分：深入实现分析（专家级）

本部分基于文档 [burn_chapter3_formulas_detailed.md](../burn_chapter3_formulas_detailed.md)，深入分析 Burn 框架中深度学习公式的实现细节，重点关注数值稳定性、梯度计算和优化技巧。适合希望深入理解框架内部工作原理的高级学习者和框架开发者。

### 1. 基础数学运算的代码映射

#### 1.1 向量运算

| 数学运算 | Burn 代码 | 对应公式 |
|---------|-----------|----------|
| 向量点积 | `(x1 * x2).sum_dim(dim)` | $\sum_i x_{1,i} x_{2,i}$ |
| L2 范数 | `x.square().sum().sqrt()` | $\|x\|_2 = \sqrt{\sum_i x_i^2}$ |
| 余弦相似度 | `dot_product / (norm_x1 * norm_x2)` | $\frac{x_1 \cdot x_2}{\|x_1\| \|x_2\|}$ |

**余弦相似度的完整实现**（`crates/burn-tensor/src/tensor/linalg/cosine_similarity.rs`）：

```rust
pub fn cosine_similarity<B: Backend, const D: usize>(
    x1: Tensor<B, D>,
    x2: Tensor<B, D>,
    dim: i32,
    eps: Option<B::FloatElem>,
) -> Tensor<B, D> {
    let eps = eps.unwrap_or_else(|| B::FloatElem::from_elem(DEFAULT_EPSILON));
    
    // 转换为正维度索引
    let dim_idx = if dim < 0 { D as i32 + dim } else { dim } as usize;
    
    // 计算点积：∑(x1_i * x2_i)
    let dot_product = (x1.clone() * x2.clone()).sum_dim(dim_idx);
    
    // 计算L2范数：‖x1‖ 和 ‖x2‖
    let norm_x1 = l2_norm(x1, dim_idx);
    let norm_x2 = l2_norm(x2, dim_idx);
    
    // 分母加上epsilon防止除零
    let denominator = norm_x1.clamp_min(eps) * norm_x2.clamp_min(eps);
    
    // 返回余弦相似度
    dot_product / denominator
}
```

#### 1.2 Softmax 与数值稳定性

**数学公式**：
$$
\text{softmax}(x_i) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}
$$

**数值稳定版本**：
$$
\text{softmax}(x_i) = \frac{\exp(x_i - \max(x))}{\sum_j \exp(x_j - \max(x))}
$$

**Burn 实现分析**（`crates/burn-tensor/src/tensor/activation/base.rs`）：

```rust
pub fn softmax<const D: usize, B: Backend>(tensor: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    check!(TensorCheck::dim_ops::<D>("softmax", dim));

    // 减去最大值防止指数溢出：x_i' = x_i - max(x)
    let tensor = tensor.clone() - tensor.detach().max_dim(dim);
    
    // 计算 exp(x_i')
    let tensor = tensor.exp();
    
    // 计算 ∑_j exp(x_j')
    let tensor_tmp = tensor.clone().sum_dim(dim);
    
    // 返回 exp(x_i') / ∑_j exp(x_j')
    tensor.div(tensor_tmp)
}
```

**关键点**：
1. 使用 `detach()` 防止梯度流向 `max_dim` 计算
2. 减去最大值确保指数运算不会溢出
3. 先计算指数再求和，符合数学定义

### 2. 损失函数的公式实现

#### 2.1 交叉熵损失的完整推导

**理论公式**：
$$
\mathcal{L}_{ce} = -\frac{1}{N} \sum_{n=1}^N \log \frac{\exp(f(x_n)_{y_n})}{\sum_z \exp(f(x_n)_z)}
$$

**等价形式**（计算更稳定）：
$$
\mathcal{L}_{ce} = -\frac{1}{N} \sum_{n=1}^N \left[ f(x_n)_{y_n} - \log\sum_z \exp(f(x_n)_z) \right]
$$

**Burn 实现分解**：

```rust
fn forward_default(&self, logits: Tensor<B, 2>, targets: Tensor<B, 1, Int>) -> Tensor<B, 1> {
    let [batch_size] = targets.dims();
    let mask = self.padding_mask(&targets);
    
    // 步骤1：计算 log_softmax = f(x)_y - log∑exp(f(x)_z)
    let tensor = log_softmax(logits, 1);
    
    // 步骤2：提取真实类别的 log 概率
    let tensor = tensor.gather(1, targets.clone().reshape([batch_size, 1]));
    
    // 步骤3：计算负对数似然并平均
    let tensor = Self::apply_mask_1d(tensor.reshape([batch_size]), mask);
    tensor.mean().neg()  // -(1/N) * sum(log_prob)
}
```

**数学等价性证明**：
1. `log_softmax(logits, 1)` = $f(x)_y - \log\sum_z \exp(f(x)_z)$
2. `.gather(1, targets)` 选择每个样本的真实类别 $y_n$
3. `.mean().neg()` = $-\frac{1}{N}\sum_{n=1}^N$

#### 2.2 对比损失的三元组形式

**原始三元组损失公式**：
$$
\mathcal{L}_{\text{triplet}} = \max(0, d(x_a, x_p) - d(x_a, x_n) + \alpha)
$$
其中 $d$ 是距离函数，$\alpha$ 是边界（margin）。

**Burn 的余弦嵌入损失变种**：

```rust
// 相似样本对 (target == 1): 损失 = 1 - cos_sim
let similar_loss = cos_sim.clone().neg().add_scalar(1);

// 不相似样本对 (target == -1): 损失 = max(0, cos_sim - margin)
let dissimilar_loss = relu(cos_sim.clone().sub_scalar(self.margin));
```

**公式转换**：
- 相似样本：$1 - \cos(x_1, x_2)$，最小化时使 $\cos(x_1, x_2) \to 1$
- 不相似样本：$\max(0, \cos(x_1, x_2) - \text{margin})$，最小化时使 $\cos(x_1, x_2) < \text{margin}$

### 3. 优化算法的公式实现

#### 3.1 权重衰减（L2正则化）

**理论公式**：
总损失：$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \frac{\lambda}{2} \|w\|^2$

梯度：$\nabla_w \mathcal{L}_{\text{total}} = \nabla_w \mathcal{L}_{\text{data}} + \lambda w$

**Burn 实现**（`crates/burn-optim/src/optim/decay.rs`）：

```rust
pub fn transform<B: Backend, const D: usize>(
    &self,
    grad: Tensor<B, D>,      // ∇ℒ_data
    tensor: Tensor<B, D>,    // w
) -> Tensor<B, D> {
    // grad + λ * w
    tensor.mul_scalar(self.penalty).add(grad)
}
```

**注意**：这里的 `penalty` 对应公式中的 $\lambda$。

#### 3.2 动量优化器

**标准动量公式**：
$$
\begin{aligned}
v_{t+1} &= \mu v_t + (1 - \tau) g_t \\
w_{t+1} &= w_t - \eta v_{t+1}
\end{aligned}
$$

**Nesterov 动量公式**：
$$
\begin{aligned}
v_{t+1} &= \mu v_t + (1 - \tau) g_t \\
w_{t+1} &= w_t - \eta (g_t + \mu v_{t+1})
\end{aligned}
$$

**Burn 实现**（`crates/burn-optim/src/optim/momentum.rs`）：

```rust
pub fn transform<const D: usize>(
    &self,
    grad: Tensor<B, D>,      // g_t
    state: Option<MomentumState<B, D>>,
) -> (Tensor<B, D>, MomentumState<B, D>) {
    // 计算速度 v_t = μ * v_{t-1} + (1 - τ) * g_t
    let velocity = if let Some(state) = state {
        grad.clone()
            .mul_scalar(1.0 - self.dampening)      // (1 - τ) * g_t
            .add(state.velocity.mul_scalar(self.momentum))  // + μ * v_{t-1}
    } else {
        grad.clone()  // 初始状态：v_0 = g_0
    };

    // 选择更新规则
    let grad = match self.nesterov {
        true => velocity.clone().mul_scalar(self.momentum).add(grad),  // g_t + μ * v_t
        false => velocity.clone(),  // 直接使用 v_t
    };

    (grad, MomentumState::new(velocity))
}
```

**参数对应关系**：
- `self.momentum` = $\mu$（动量因子）
- `self.dampening` = $\tau$（阻尼因子）
- `self.nesterov` = 是否使用 Nesterov 动量

#### 3.3 梯度裁剪

##### 3.3.1 按值裁剪

**公式**：
$$
g_{\text{clipped}, i} = \begin{cases}
\text{threshold} & \text{if } g_i > \text{threshold} \\
-\text{threshold} & \text{if } g_i < -\text{threshold} \\
g_i & \text{otherwise}
\end{cases}
$$

**Burn 实现**：

```rust
fn clip_by_value<B: Backend, const D: usize>(
    &self,
    grad: Tensor<B, D>,
    threshold: f32,
) -> Tensor<B, D> {
    let greater_mask = grad.clone().greater_elem(threshold);
    let lower_mask = grad.clone().lower_elem(-threshold);

    let clipped_grad = grad.mask_fill(greater_mask, threshold);
    clipped_grad.mask_fill(lower_mask, -threshold)
}
```

##### 3.3.2 按范数裁剪

**公式**：
$$
g_{\text{clipped}} = \begin{cases}
g & \text{if } \|g\| \leq \text{threshold} \\
\frac{\text{threshold}}{\|g\|} g & \text{otherwise}
\end{cases}
$$

**Burn 实现**：

```rust
fn clip_by_norm<B: Backend, const D: usize>(
    &self,
    grad: Tensor<B, D>,
    threshold: f32,
) -> Tensor<B, D> {
    // 计算L2范数：‖g‖ = √(∑ g_i^2)
    let norm = Self::l2_norm(grad.clone());
    
    // 计算裁剪系数：threshold / (‖g‖ + ε)
    let clip_coef = threshold / norm.add_scalar(1e-6);
    
    // 系数限制在 [0, 1]，只有范数超过阈值时才裁剪
    let clip_coef_clamped = clip_coef.clamp_max(1.0);
    
    // 应用裁剪：g * min(1, threshold/‖g‖)
    grad.mul(clip_coef_clamped.unsqueeze())
}
```

### 4. 自动微分的链式法则实现

#### 4.1 计算图构建

Burn 的 `Autodiff<B>` 后端装饰器自动跟踪所有张量操作：

```rust
type Backend = Autodiff<Wgpu>;

let x: Tensor<Backend, 2> = Tensor::random([32, 32], Distribution::Default, &device);
let y: Tensor<Backend, 2> = Tensor::random([32, 32], Distribution::Default, &device).require_grad();

// 前向传播（自动构建计算图）
let z = x.matmul(y);      // 节点1：矩阵乘法
let w = z.relu();         // 节点2：ReLU激活  
let loss = w.mean();      // 节点3：平均值

// 反向传播（自动应用链式法则）
let grads = loss.backward();  // 计算 ∂loss/∂x 和 ∂loss/∂y
```

#### 4.2 链式法则的具体应用

对于复合函数 $f(g(x))$，链式法则为：
$$
\frac{df}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}
$$

**以 ReLU 为例**：

```rust
// ReLU 的前向传播
fn relu_forward(x: Tensor) -> Tensor {
    x.maximum(0.0)
}

// ReLU 的反向传播
fn relu_backward(grad_output: Tensor, x: Tensor) -> Tensor {
    let mask = x.greater_elem(0.0);  // ∂relu/∂x = 1 if x > 0 else 0
    grad_output * mask
}
```

**矩阵乘法的导数**：
对于 $z = x y$，有：
$$
\frac{\partial z}{\partial x} = y^T, \quad \frac{\partial z}{\partial y} = x^T
$$

### 5. 学习率调度的数学实现

#### 5.1 余弦退火调度

**公式**：
$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{T_{\text{cur}}}{T_{\max}}\pi\right)\right)
$$

**Burn 实现**：

```rust
fn step(&mut self, step: usize) -> LearningRate {
    if step >= self.t_max {
        return self.eta_min;
    }
    
    let progress = step as f64 / self.t_max as f64;
    let cosine = f64::cos(std::f64::consts::PI * progress);
    
    self.eta_min + (self.eta_max - self.eta_min) * (1.0 + cosine) / 2.0
}
```

#### 5.2 线性预热

**公式**：
$$
\eta_t = \eta_{\text{start}} + (\eta_{\text{end}} - \eta_{\text{start}}) \times \frac{t}{T_{\text{warmup}}}
$$

**Burn 实现**：

```rust
fn step(&mut self, step: usize) -> LearningRate {
    if step >= self.num_steps {
        return self.end_learning_rate;
    }
    
    let ratio = step as f64 / self.num_steps as f64;
    self.start_learning_rate + (self.end_learning_rate - self.start_learning_rate) * ratio
}
```

### 6. 关键实现模式总结

#### 6.1 数值稳定性模式

| 问题 | 解决方案 | Burn 代码示例 |
|------|----------|---------------|
| 指数溢出 | 减去最大值 | `x - x.detach().max_dim(dim)` |
| 除零错误 | 添加 epsilon | `x / (y + 1e-6)` |
| 对数负数 | 钳制最小值 | `x.clamp_min(1e-8).log()` |

#### 6.2 并行计算模式

| 运算类型 | Burn 实现 | 对应数学 |
|----------|-----------|----------|
| 逐元素运算 | `x.exp()`, `x.log()` | $e^x$, $\log x$ |
| 归约运算 | `x.sum_dim(dim)`, `x.mean()` | $\sum_i x_i$, $\frac{1}{N}\sum_i x_i$ |
| 广播运算 | `x.reshape(shape).repeat_dim(dim, n)` | 维度扩展 |

#### 6.3 梯度计算模式

| 操作类型 | 前向传播 | 反向传播 |
|----------|----------|----------|
| 线性变换 | `x.matmul(y)` | $\frac{\partial}{\partial x} = y^T$, $\frac{\partial}{\partial y} = x^T$ |
| 激活函数 | `x.relu()` | `mask = x > 0; grad * mask` |
| 损失函数 | `cross_entropy(logits, targets)` | $\frac{\partial\mathcal{L}}{\partial\text{logits}} = \text{softmax} - \text{one\_hot}$ |

### 7. 从公式到代码的实践指南

#### 7.1 实现新损失函数的步骤

1. **写出数学公式**：明确输入输出和计算过程
2. **考虑数值稳定性**：识别可能溢出或不稳定的操作
3. **转换为张量操作**：将求和、乘积等转换为 Burn API 调用
4. **实现前向传播**：实现 `forward` 方法
5. **验证梯度**：使用自动微分验证梯度正确性

#### 7.2 实现新优化器的步骤

1. **定义更新公式**：明确参数更新规则
2. **实现状态管理**：定义优化器状态结构
3. **实现 `step` 方法**：应用更新公式
4. **添加配置支持**：实现 `Config` trait
5. **测试收敛性**：在简单问题上测试优化器

#### 7.3 调试技巧

1. **梯度检查**：比较自动微分与数值梯度
2. **中间值检查**：在关键点打印张量值
3. **数值范围检查**：检查指数、对数运算的输入范围
4. **形状检查**：验证所有张量操作的形状匹配

### 第三部分总结

本部分深入分析了 Burn 框架中深度学习公式的底层实现细节。通过分析这些实现，我们可以：

1. **理解公式的实际计算**：看到数学公式如何转化为高效、稳定的代码
2. **学习数值稳定性技巧**：掌握防止数值问题的最佳实践
3. **掌握优化技术**：理解各种优化算法的具体实现细节
4. **扩展框架功能**：基于现有模式实现新的损失函数或优化器

这种公式到代码的映射不仅有助于理解 Burn 框架，也为在其他框架中实现类似功能提供了参考。

---

## 总结与学习建议

本文档通过三个渐进式的层次，全面分析了《深度学习小书》第三章"训练"中的核心概念在 Burn 框架中的实现：

### 学习路径建议

1. **初学者**：从第一部分开始，通过高级 API 示例快速上手 Burn 框架，理解如何将理论概念转化为代码
2. **中级学习者**：在掌握 API 使用后，深入学习第二部分，理解数学公式到源代码的直接转换
3. **高级学习者/框架开发者**：研究第三部分的深入分析，掌握数值稳定性、梯度计算等底层实现细节

### 关键收获

1. **理论与实践的结合**：深度学习理论（损失函数、优化算法、反向传播）与现代框架实现（Burn）的对应关系
2. **数值稳定性的重要性**：实际代码中如何处理指数溢出、除零错误等数值计算问题
3. **框架设计模式**：Burn 的模块化设计、自动微分机制和多后端支持
4. **代码可读性与效率**：Rust 语言特性如何用于构建高效且安全的深度学习框架

### 扩展学习

1. **深入 Burn 源代码**：基于本文档的分析，进一步探索 Burn 的其他模块（如神经网络层、数据加载器等）
2. **对比其他框架**：将 Burn 的实现与 PyTorch、TensorFlow 等框架进行对比，理解不同设计哲学
3. **实现自定义组件**：基于本文档中的模式，尝试实现自定义的损失函数或优化器
4. **性能优化研究**：分析 Burn 在不同后端（CPU、GPU）上的性能表现，理解优化技巧

### 参考资料

1. **Burn 官方资源**：
   - 源代码：https://github.com/tracel-ai/burn
   - 文档：https://burn.dev/docs/burn
   - 示例：https://github.com/tracel-ai/burn/tree/main/examples

2. **《深度学习小书》**：
   - 第三章原文：`../3_foundations.md`
   - 完整书籍：https://littlebookofdeeplearning.com/

3. **相关技术文档**：
   - 自动微分原理：Baydin et al., "Automatic Differentiation in Machine Learning: a Survey" (2015)
   - 优化算法综述：Ruder, "An overview of gradient descent optimization algorithms" (2016)
   - 数值稳定性：Higham, "Accuracy and Stability of Numerical Algorithms" (2002)

---

**相关文档**：
- [burn_chapter3_mapping.md](../burn_chapter3_mapping.md)：高级知识点到代码的映射
- [burn_chapter3_formulas.md](../burn_chapter3_formulas.md)：公式与源代码的直接对应
- [burn_chapter3_formulas_detailed.md](../burn_chapter3_formulas_detailed.md)：深入的公式分析

**注**：本文档整合了上述三个文档的内容，提供了一个从入门到精通的渐进式学习路径。建议按照文档的三个部分顺序学习，逐步深入理解 Burn 框架的实现原理。

