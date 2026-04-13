# 《深度学习小书》第三章知识点在 Burn 框架中的映射

本文档将《深度学习小书》（Little Book of Deep Learning）第三章“训练”中的核心知识点与 Rust 深度学习框架 [Burn](https://github.com/tracel-ai/burn) 的代码实现进行对应。通过实际代码示例，展示这些理论概念如何在现代深度学习框架中具体实现。

## 1. 损失函数（Loss Functions）

### 1.1 交叉熵损失（Cross-Entropy Loss）

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

### 1.2 对比损失（Contrastive Loss）

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

### 1.3 其他损失函数

Burn 还提供了多种常见损失函数：

- **均方误差（MSE）**：`burn_nn::loss::MseLoss`
- **平均绝对误差（MAE/L1）**：`burn_nn::loss::LpLoss`
- **Huber 损失**：`burn_nn::loss::HuberLoss`
- **KL 散度**：`burn_nn::loss::KlDivLoss`
- **连接主义时间分类（CTC）**：`burn_nn::loss::CtcLoss`

## 2. 自回归模型（Autoregressive Models）

自回归模型通过链式法则分解序列数据的联合概率，常用于语言建模和序列生成。

### 2.1 因果模型（Causal Models）

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

### 2.2 分词器（Tokenizers）

Burn 本身不包含分词器实现，但可以与外部分词器库（如 `tokenizers`）结合使用。

```rust
use tokenizers::Tokenizer;

let tokenizer = Tokenizer::from_pretrained("bert-base-uncased").unwrap();
let encoding = tokenizer.encode("Hello, world!", false).unwrap();
let token_ids = encoding.get_ids();
```

## 3. 梯度下降（Gradient Descent）

### 3.1 随机梯度下降（Stochastic Gradient Descent）

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

### 3.2 Adam 优化器

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

### 3.3 学习率调度（Learning Rate Scheduling）

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

## 4. 反向传播（Backpropagation）

### 4.1 自动微分（Autodiff）

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

### 4.2 梯度裁剪（Gradient Clipping）

防止梯度爆炸的常用技术。

```rust
use burn_optim::grad_clipping::GradientClipping;

let clipping = GradientClipping::Norm(1.0); // 梯度范数裁剪到 1.0
optimizer = optimizer.with_grad_clipping(clipping);
```

## 5. 深度的价值（The Value of Depth）

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

## 6. 训练协议（Training Protocols）

### 6.1 数据集分割

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

### 6.2 数据加载器

```rust
use burn::data::dataloader::DataLoaderBuilder;

let dataloader = DataLoaderBuilder::new(batcher)
    .batch_size(64)
    .shuffle(42)
    .num_workers(4)
    .build(dataset);
```

### 6.3 训练循环与评估

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

### 6.4 早停（Early Stopping）

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

## 7. 规模的好处（The Benefit of Scale）

### 7.1 多后端支持

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

### 7.2 分布式训练

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

## 总结

Burn 框架全面实现了《深度学习小书》第三章中介绍的核心训练概念：

1. **损失函数**：提供了丰富的损失函数实现，支持现代训练技巧
2. **优化算法**：实现了 SGD、Adam 等主流优化器及其变种
3. **自动微分**：通过后端装饰器模式提供透明的反向传播
4. **训练工具**：提供了完整的数据流水线、训练循环和监控工具
5. **可扩展性**：支持多后端和分布式训练，适应不同规模需求

通过 Burn 的模块化设计，这些组件可以灵活组合，构建从研究原型到生产部署的完整深度学习流程。

## 参考资料

1. Burn 官方文档：https://burn.dev/docs/burn
2. Burn GitHub 仓库：https://github.com/tracel-ai/burn
3. 《深度学习小书》第三章：训练