# Burn框架中《深度学习小书》第三章渐进式分析

## 引言

《深度学习小书》（Little Book of Deep Learning）第三章"训练"涵盖了深度学习的核心训练概念，包括损失函数、优化算法、反向传播和训练协议等。本分析文档旨在提供一个渐进式的学习路径，将第三章的理论知识点与 Rust 深度学习框架 [Burn](https://github.com/tracel-ai/burn) 的代码实现进行对应。

**文档组织方式**：按知识点组织，每个知识点内部分三个深度层次循序渐进：

- **入门**：通过 Burn 高级 API 快速上手，适合初学者
- **进阶**：数学公式与源代码的直接对应，适合中级学习者
- **专家**：数值稳定性、底层实现细节，适合框架开发者

---

## 1. 损失函数（Loss Functions）

### 1.1 Softmax 转换

Softmax 是分类任务的基础，将 logits 转换为概率分布，是交叉熵损失的前置步骤。

> logits 指的是模型输出的未归一化分数（非归一化概率的对数）。Softmax 函数将这些 logits 转换为概率分布，使得所有类别概率之和为 1。

#### 入门：API 用法

**Burn 实现**：`burn_tensor::activation::softmax`

```rust
use burn::tensor::{Tensor, activation};

// 将 logits 转换为概率分布
let logits = Tensor::<Backend, 2>::random([batch_size, num_classes], Distribution::Normal(0., 1.), &device);
let probabilities = activation::softmax(logits, 1); // 沿类别维度（dim=1）计算 softmax
```

#### 进阶：数学公式与源码对应

**数学公式**：
$$
\hat{P}(Y = y \mid X = x) = \frac{\exp f(x;w)_y}{\sum_z \exp f(x;w)_z}
$$

**符号解释**：
- *$X$*：输入变量，$x$ 是具体的输入样本
- *$Y$*：输出类别变量，$y$ 是具体的类别索引
- *$f(x;w)_y$*：模型$f$ 在参数$w$ 下，输入$x$ 时输出的第$y$ 个 logit 值
- *${z}$*：求和变量，遍历所有可能的类别
- *$\hat{P}(Y = y \mid X = x)$*：给定输入 $x$ 时预测类别为 $y$ 的条件概率估计

**Burn 源代码**：`crates/burn-tensor/src/tensor/activation/base.rs`

```rust
pub fn softmax<const D: usize, B: Backend>(tensor: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    check!(TensorCheck::dim_ops::<D>("softmax", dim));

    // 数值稳定性处理：减去最大值防止指数溢出（详见专家部分）
    // x_i' = x_i - max(x)
    let tensor = tensor.clone() - tensor.detach().max_dim(dim);
    
    // 计算 exp(x_i') 和 sum(exp(x_j'))
    let tensor = tensor.exp();                    // exp(x_i')
    let tensor_tmp = tensor.clone().sum_dim(dim); // sum_j(exp(x_j'))
    
    // 返回 softmax(x_i) = exp(x_i') / sum_j(exp(x_j'))
    tensor.div(tensor_tmp)
}
```

> Burn 还提供了 `log_softmax` 函数，用于稳定地计算 $\log(\text{softmax}(x))$，它将在 1.2 节交叉熵损失中使用。其具体实现将在交叉熵损失部分详细介绍。

**代码解释**：
1. **softmax 实现**：$\text{softmax}(x_i) = \frac{\exp(x_i - \max(x))}{\sum_j \exp(x_j - \max(x))}$
2. **log_softmax** 使用相同的数值稳定性技巧，其具体实现将在 1.2 节交叉熵损失中展示。

#### 专家：数值稳定性细节

softmax 和 log_softmax 的实现中包含两个关键的数值稳定性技巧：

1. **`detach()`**：防止梯度流向 `max_dim` 计算。减去最大值只是为了数值稳定，不改变数学结果，因此不应参与反向传播的梯度图。
2. **减去最大值**：确保指数运算的输入 *$\leq 0$*，防止 `exp` 溢出至 `inf`。由于 $e^0 = 1$，减去最大值后最大的指数结果为 1，其余均小于 1，从而避免浮点溢出。

`log_softmax` 的具体实现和用途将在 1.2 节交叉熵损失中详细说明。

---

### 1.2 交叉熵损失（Cross-Entropy Loss）

交叉熵是分类任务的标准损失函数，衡量模型预测概率分布与真实分布之间的差异。

#### 入门：API 用法

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

#### 进阶：数学公式与源码对应

交叉熵损失的核心是衡量模型预测概率分布与真实分布之间的差异。在分类任务中，我们通过 softmax 将模型输出的 logits 转换为概率分布，然后使用交叉熵来评估这个预测分布的质量。这自然引出一个关键问题：**为什么交叉熵需要对 softmax 输出求对数？**

交叉熵的定义为 $H(p,q) = -\sum_i p_i \log q_i$，其中 *$p$* 是真实分布，*$q$* 是预测分布。这个公式来源于信息论，用于衡量两个概率分布之间的差异。

在机器学习中，我们将交叉熵作为损失函数 $\mathcal{L}_{ce}$。对于分类任务：
- 真实分布 *$p$* 是 one-hot 向量（仅在真实类别位置为 1，其余为 0）
- 预测分布 *$q$* 是 softmax 输出的概率分布

当 *$p$* 是 one-hot 向量时（设真实类别为 $y$，即 *$p_y = 1$*，$p_i = 0$ 当 $i \neq y$），交叉熵简化为：

$$
H(p,q) = -\sum_i p_i \log q_i = -1 \cdot \log q_y - \sum_{i \neq y} 0 \cdot \log q_i = -\log q(y)
$$

即

$$
-\log \hat{P}(Y = y | X = x)
$$

其中 $y$ 是真实类别索引。这个简化揭示了交叉熵损失的本质：我们只需关注真实类别的预测概率的对数。这也是为什么代码中直接使用 `log_softmax` 来计算交叉熵损失（其实现见下文）——它直接提供了所需的 $\log q(y)$。

**从信息论交叉熵到机器学习损失函数**：
上面的 $-\log q(y)$ 是针对单个样本的交叉熵。在机器学习中，我们通常在整个训练集上定义损失函数。假设有 $N$ 个独立同分布的样本，交叉熵损失 $\mathcal{L}_{ce}$ 定义为所有样本交叉熵的平均值：

基于上述分析，我们可以形式化地写出交叉熵损失的完整数学公式：

$$
\mathcal{L}_{ce}(w) = -\frac{1}{N} \sum_{n=1}^{N} \log\hat{P}(Y = y_n | X = x_n)
= -\frac{1}{N}\sum_{n=1}^{N}\log\frac{\exp f(x_n;w)_{y_n}}{\sum_z \exp f(x_n;w)_z}
$$

**符号解释**：
- *$\mathcal{L}_{ce}(w)$*：参数为 $w$ 时的交叉熵损失，这是信息论交叉熵 $H(p,q)$ 在分类任务中的具体应用形式
- *$N$*：训练样本总数
- *$n$*：样本索引，$n = 1, 2, \dots, N$
- *$x_n$*：第 $n$ 个输入样本
- *$y_n$*：第 $n$ 个样本的真实类别
- *$f(x_n;w)_{y_n}$*：模型$f$ 在参数 $w$ 下，输入 $x_n$ 时在真实类别 $y_n$ 上的 logit 输出
- *${z}$*：求和变量，遍历所有可能的类别
- *$\hat{P}(Y = y_n | X = x_n)$*：给定输入 $x_n$ 时预测类别为 $y_n$ 的条件概率估计（即 softmax 输出）

**等价形式（计算更稳定）**：
上面的公式 

$$
\mathcal{L}_{ce}(w) = -\frac{1}{N}\sum_{n=1}^{N}\log\frac{\exp f(x_n;w)_{y_n}}{\sum_z \exp f(x_n;w)_z}
$$ 

可以进一步展开为等价形式：

$$
\mathcal{L}_{ce}(w) = -\frac{1}{N} \sum_{n=1}^N \left[ f(x_n;w)_{y_n} - \log\sum_z \exp(f(x_n;w)_z) \right]
$$

这个形式在数值计算上更稳定，因为它避免了先计算 softmax（可能产生接近 0 的值）再取对数（可能导致 $-\infty$）的问题。**这个等价形式正是下面代码实现所采用的形式**，通过 `log_softmax` 函数直接计算 $f(x)_{y} - \log\sum_z \exp(f(x)_z)$。

**Burn 源代码**：`crates/burn-nn/src/loss/cross_entropy.rs`

**log_softmax 实现**（位于 `crates/burn-tensor/src/tensor/activation/base.rs`）：

```rust
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

**交叉熵损失函数实现**：

```rust
fn forward_default(&self, logits: Tensor<B, 2>, targets: Tensor<B, 1, Int>) -> Tensor<B, 1> {
    let [batch_size] = targets.dims();

    // 1. 计算填充掩码（忽略填充令牌）
    let mask = self.padding_mask(&targets);
    
    // 2. 计算 log softmax: log(exp(x_y) / sum_z(exp(x_z)))
    // 对应公式中的 log(exp(f(x;w)_y) / sum_z exp(f(x;w)_z))
    let tensor = log_softmax(logits, 1);
    
    // 3. 提取真实类别的对数概率（log probability）
    // tensor.gather(1, targets.reshape([batch_size, 1])) 选择每个样本真实类别的对数概率
    let tensor = tensor.gather(1, targets.clone().reshape([batch_size, 1]));
    
    // 4. 根据是否使用权重计算损失
    match &self.weights {
        Some(weights) => {
            // 加权交叉熵：先计算 weight * log(p)，取负在后续 .neg() 完成
            let weights = weights.clone().gather(0, targets);
            let tensor = tensor.reshape([batch_size]) * weights.clone();
            let tensor = Self::apply_mask_1d(tensor, mask);
            // 对应加权平均：-sum(weight * log(p)) / sum(weights)
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

#### 专家：数值稳定性与实现细节

**代码实现与等价形式的关系**：
前面的等价形式： 

$$
\mathcal{L}_{ce}(w) = -\frac{1}{N} \sum_{n=1}^N \left[ f(x_n;w)_{y_n} - \log\sum_z \exp(f(x_n;w)_z) \right]
$$

正是代码中 `log_softmax` 函数的数学表达。具体对应关系如下：

1. **`log_softmax` 输出**：对于输入 logits $f(x)$，`log_softmax` 的第 $i$ 个输出为 $f(x)_i - \log\sum_z \exp(f(x)_z)$
2. **提取真实类别**：`.gather(1, targets)` 选择真实类别 $y_n$ 对应的值，得到 $f(x)_{y_n} - \log\sum_z \exp(f(x)_z)$
3. **平均取负**：`.mean().neg()` 计算 $-\frac{1}{N}\sum_{n=1}^N$

**数值稳定性机制**：
`log_softmax` 的实现（见上文）包含两个关键技巧：
1. **减去最大值**：$x_i' = x_i - \max(x)$，确保所有指数运算的输入 *$\leq 0$*，防止 `exp` 溢出至 `inf`
2. **`detach()`**：防止梯度流向 `max_dim` 计算，因为减去最大值只是为了数值稳定，不改变数学结果

**为何这种形式更优**：
对比两种计算路径：
- **路径 A（先 softmax 再 log）**：$\log(\text{softmax}(x)) = \log\left(\frac{\exp(x_i)}{\sum_j \exp(x_j)}\right)$
  - 问题：如果 softmax 输出接近 0，取对数会得到 $-\infty$
- **路径 B（log_softmax）**：$\log\text{softmax}(x_i) = x_i - \log\sum_j \exp(x_j)$
  - 优势：避免中间的小概率值，直接使用 log-sum-exp 技巧，数值稳定且计算高效

**与 1.1 节的联系**：
这里使用的数值稳定性技巧与 1.1 节 `softmax` 的实现一致，体现了 Burn 框架对数值稳定性的系统化处理。

---

### 1.3 对比损失（Contrastive Loss）

对比损失用于度量学习，使相似样本在嵌入空间中更接近，不相似样本更远离。

#### 入门：API 用法

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

#### 进阶：数学公式与源码对应

**数学公式**（对比损失变种）：
对于三元组 $(x_a, x_b, x_c)$ 满足 $y_a = y_b \neq y_c$：
$$
\mathcal{L}_{\text{contrastive}} = \max(0, 1 - f(x_a, x_c; w) + f(x_a, x_b; w))
$$

**符号解释**：
- *$\mathcal{L}_{\text{contrastive}}$*：对比损失
- *$x_a$*：锚点（anchor）样本
- *$x_b$*：正样本（positive），与锚点属于同一类别（$y_a = y_b$）
- *$x_c$*：负样本（negative），与锚点属于不同类别（$y_a \neq y_c$）
- *$y_a, y_b, y_c$*：样本 $x_a, x_b, x_c$ 的类别标签
- *$f(x_a, x_b; w)$*：模型$f$ 在参数 $w$ 下计算的$x_a$ 与$x_b$ 之间的相似度
- *$f(x_a, x_c; w)$*：模型$f$ 在参数 $w$ 下计算的$x_a$ 与$x_c$ 之间的相似度
- *$w$*：模型参数

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
    let cos_sim = cosine_similarity(input1, input2, 1, None);
    let cos_sim: Tensor<B, 1> = cos_sim.squeeze_dim(1);

    let mut loss = cos_sim.zeros_like();

    // 相似样本对（target == 1）：损失 = 1 - cos_sim
    let similar_mask = target.clone().equal_elem(1);
    let similar_loss = cos_sim.clone().neg().add_scalar(1);
    loss = loss.mask_where(similar_mask, similar_loss);

    // 不相似样本对（target == -1）：损失 = max(0, cos_sim - margin)
    let dissimilar_mask = target.equal_elem(-1);
    let dissimilar_loss = relu(cos_sim.clone().sub_scalar(self.margin));
    loss = loss.mask_where(dissimilar_mask, dissimilar_loss);

    loss
}
```

**代码与公式对应关系**：
| 公式部分 | 代码实现 | 说明 |
|---------|---------|------|
| *$f(x_a, x_b; w)$* | `cosine_similarity(input1, input2, 1, None)` | 余弦相似度作为距离函数 |
| *$1 - f(x_a, x_b; w)$* | `cos_sim.clone().neg().add_scalar(1)` | 相似样本损失 |
| *$\max(0, f(x_a, x_c; w) - \text{margin})$* | `relu(cos_sim.clone().sub_scalar(self.margin))` | 不相似样本损失（带边界） |
| 样本类型选择 | `.mask_where(mask, loss)` | 根据 target 值选择相应损失 |

#### 专家：三元组形式与余弦相似度计算细节

**原始三元组损失公式**：
$$
\mathcal{L}_{\text{triplet}} = \max(0, d(x_a, x_p) - d(x_a, x_n) + \alpha)
$$

其中$d$ 是距离函数，$\alpha$ 是边界（margin）。

**公式转换**：
- 相似样本：$1 - \cos(x_1, x_2)$，最小化使 $\cos(x_1, x_2) \to 1$
- 不相似样本：$\max(0, \cos(x_1, x_2) - \text{margin})$，最小化使$\cos(x_1, x_2) < \text{margin}$

**余弦相似度的完整实现**（`crates/burn-tensor/src/tensor/linalg/cosine_similarity.rs`）：

```rust
pub fn cosine_similarity<B: Backend, const D: usize>(
    x1: Tensor<B, D>,
    x2: Tensor<B, D>,
    dim: i32,
    eps: Option<B::FloatElem>,
) -> Tensor<B, D> {
    let eps = eps.unwrap_or_else(|| B::FloatElem::from_elem(DEFAULT_EPSILON));
    
    let dim_idx = if dim < 0 { D as i32 + dim } else { dim } as usize;
    
    // 计算点积：∑(x1_i * x2_i)
    let dot_product = (x1.clone() * x2.clone()).sum_dim(dim_idx);
    
    // 计算L2范数：‖x1‖ 和 ‖x2‖
    let norm_x1 = l2_norm(x1, dim_idx);
    let norm_x2 = l2_norm(x2, dim_idx);
    
    // 分母加上epsilon防止除零
    let denominator = norm_x1.clamp_min(eps) * norm_x2.clamp_min(eps);
    
    dot_product / denominator
}
```

---

### 1.4 其他损失函数

Burn 还提供了多种常见损失函数：

- **均方误差（MSE）**：`burn_nn::loss::MseLoss`
- **平均绝对误差（MAE/L1）**：`burn_nn::loss::LpLoss`
- **Huber 损失**：`burn_nn::loss::HuberLoss`
- **KL 散度**：`burn_nn::loss::KlDivLoss`
- **连接主义时间分类（CTC）**：`burn_nn::loss::CtcLoss`

---

## 2. 自回归模型（Autoregressive Models）

自回归模型通过链式法则分解序列数据的联合概率，常用于语言建模和序列生成。

#### 入门：因果模型与分词器

**因果模型（Causal Models）**

因果模型确保每个时间步的预测仅依赖于之前的输入，不泄露未来信息。

**Burn 实现**：Burn 没有 `with_causal_mask` 参数，因果掩码通过 `generate_autoregressive_mask` 生成后传入 `TransformerEncoderInput`。

**Burn 源代码**：`crates/burn-nn/src/modules/attention/mask.rs`

```rust
// 生成自回归掩码：下三角矩阵，确保位置 t 只能看到 t' <= t
pub fn generate_autoregressive_mask<B: Backend>(
    batch_size: usize,
    seq_length: usize,
    device: &B::Device,
) -> Tensor<B, 3, Bool> {
    // tril_mask 生成 [seq_length, seq_length] 下三角布尔矩阵
    let mask = Tensor::<B, 2, Bool>::tril_mask([seq_length, seq_length], 0, device);
    mask.expand([batch_size, seq_length, seq_length])
}
```

**训练期间（完整序列，一次前向）**：

```rust
use burn::nn::{
    transformer::{TransformerEncoderConfig, TransformerEncoderInput},
    attention::generate_autoregressive_mask,
};
use burn::prelude::*;

// TransformerEncoderConfig::new(d_model, d_ff, n_heads, n_layers)
let transformer = TransformerEncoderConfig::new(512, 2048, 8, 6)
    .init::<Backend>(&device);

// tokens: [batch_size, seq_length, d_model]
let mask_attn = generate_autoregressive_mask::<Backend>(batch_size, seq_length, &device);
let input = TransformerEncoderInput::new(tokens).mask_attn(mask_attn);
let output = transformer.forward(input); // [batch_size, seq_length, d_model]
```

**推理期间（逐步生成，复用 KV 缓存）**：

```rust
// new_autoregressive_cache() 缓存历史 K/V，避免重复计算
let mut cache = transformer.new_autoregressive_cache();

for step in 0..max_seq_len {
    let mask = generate_autoregressive_mask::<Backend>(batch_size, step + 1, &device);
    let input = TransformerEncoderInput::new(current_token).mask_attn(mask);
    let logits = transformer.forward_autoregressive_inference(input, &mut cache);
    let next_token = logits.argmax(2); // 贪婪解码
    // ...拼接 next_token 到已生成序列
}
```

**分词器（Tokenizers）**

Burn 本身不包含分词器实现，但可以与外部分词器库（如 `tokenizers`）结合使用。

```rust
use tokenizers::Tokenizer;

let tokenizer = Tokenizer::from_pretrained("bert-base-uncased").unwrap();
let encoding = tokenizer.encode("Hello, world!", false).unwrap();
let token_ids = encoding.get_ids();
```

---

## 3. 梯度下降（Gradient Descent）

### 3.1 SGD 优化器

#### 入门：API 用法

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

#### 进阶：梯度下降更新公式

**数学公式**：

$$
w_{n+1} = w_n - \eta \nabla\mathcal{L}|_w(w_n)
$$

**符号解释**：
- *$w_n$*：第 $n$ 次迭代时的模型参数
- *$w_{n+1}$*：更新后的模型参数
- *$\eta$*：学习率（learning rate），控制参数更新步长
- *$\nabla\mathcal{L}|_w(w_n)$*：损失函数 $\mathcal{L}$ 在参数 $w_n$ 处的梯度

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
| 动量项 | `momentum.transform(grad, state_momentum)` | 动量加速 |

#### 专家：权重衰减与 L2 正则化

**理论公式**：
总损失：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \frac{\lambda}{2} \|w\|^2$$

梯度：

$$\nabla_w \mathcal{L}_{\text{total}} = \nabla_w \mathcal{L}_{\text{data}} + \lambda w$$

**符号解释**：
- *$\mathcal{L}_{\text{total}}$*：包含正则化的总损失
- *$\mathcal{L}_{\text{data}}$*：仅数据部分的损失（不含正则化）
- *$\lambda$*：权重衰减系数（正则化强度）
- *$w$*：模型参数
- *$\|w\|^2$*：参数 $w$ 的 L2 范数平方
- *$\nabla_w \mathcal{L}_{\text{total}}$*：总损失对参数 $w$ 的梯度
- *$\nabla_w \mathcal{L}_{\text{data}}$*：数据损失对参数 $w$ 的梯度

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

---

### 3.2 动量（Momentum）

#### 进阶：动量公式

**数学公式**：
$$
\begin{aligned}
v_{t+1} &= \mu v_t + (1 - \tau) g_t \\
w_{t+1} &= w_t - \eta v_{t+1}
\end{aligned}
$$

**符号解释**：
- *$v_t$*：第 $t$ 次迭代时的动量（速度）
- *$v_{t+1}$*：更新后的动量
- *$\mu$*：动量因子，控制历史动量的保留比例
- *$\tau$*：阻尼因子
- *$g_t$*：第 $t$ 次迭代时的梯度
- *$w_t$*：第 $t$ 次迭代时的模型参数
- *$w_{t+1}$*：更新后的模型参数
- *$\eta$*：学习率

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

#### 专家：标准动量与 Nesterov 动量对比

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

**Burn 实现中的参数对应关系**：
- `self.momentum` = $\mu$（动量因子）
- `self.dampening` = $\tau$（阻尼因子）
- `self.nesterov` = 是否使用 Nesterov 动量

Nesterov 动量的优势：在 Burn 实现中，`nesterov=true` 时将当前梯度与动量结合 `grad.add(velocity.mul_scalar(self.momentum))`，等效于在"预看"位置计算梯度，收敛速度通常更快。

---

### 3.3 Adam 优化器

#### 入门：API 用法

**Burn 实现**：`burn_optim::Adam` 和 `burn_optim::AdamW`

**Burn 源代码**：`crates/burn-optim/src/optim/adamw.rs`

```rust
use burn::optim::AdamWConfig;

// Config derive 宏根据字段名 beta_1/beta_2/epsilon/weight_decay 自动生成 with_* setter
let config = AdamWConfig::new()           // 默认：β1=0.9, β2=0.999, ε=1e-8, wd=0.01
    .with_beta_1(0.9f32)
    .with_beta_2(0.999f32)
    .with_epsilon(1e-8f32)
    .with_weight_decay(0.01f32)
    .with_cautious_weight_decay(true);    // 谨慎权重衰减变体

// init() 返回优化器实例（与 Learner 配合使用，见 §6.3）
let optimizer = config.init();
```

> `AdamWConfig` 字段名直接对应论文符号：`beta_1`（$\beta_1$）、`beta_2`（$\beta_2$）、`epsilon`（$\epsilon$）。`#[derive(Config)]` 宏为每个字段自动生成对应的 `with_*` setter 方法。

---

### 3.4 学习率调度（Learning Rate Scheduling）

#### 入门：API 用法

**Burn 实现**：`burn_optim::lr_scheduler`

**Burn 源代码**：`crates/burn-optim/src/lr_scheduler/`

```rust
use burn::lr_scheduler::{
    composed::ComposedLrSchedulerConfig,
    cosine::CosineAnnealingLrSchedulerConfig,
    linear::LinearLrSchedulerConfig,
};

// CosineAnnealingLrSchedulerConfig::new(initial_lr, num_iters)
// LinearLrSchedulerConfig::new(initial_lr, final_lr, num_iters)
//
// ComposedLrScheduler 默认将所有子调度器输出相乘（Prod）
// → 预热（0→1）× 余弦退火（1→0）= 先升后降的学习率曲线
let lr_scheduler = ComposedLrSchedulerConfig::new()
    .cosine(CosineAnnealingLrSchedulerConfig::new(1.0, 10_000))   // 余弦退火
    .linear(LinearLrSchedulerConfig::new(1e-8, 1.0, 1_000))       // 预热 1000 步
    .linear(LinearLrSchedulerConfig::new(1e-2, 1e-6, 10_000))     // 衰减至最小
    .init()
    .unwrap();
// LrScheduler::step() 在 Learner 内部自动调用，无需手动调用
```

> `ComposedLrScheduler` 将各子调度器的 `step()` 输出逐元素相乘。典型用法：预热阶段（$0 \to 1$）与余弦退火阶段（$1 \to 0$）相乘，天然得到 Warmup + Cosine Decay 曲线。

#### 进阶：余弦退火公式

**数学公式**（Cosine Annealing）：
$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{T_{\text{cur}}}{T_{\max}}\pi\right)\right)
$$

**符号解释**：
- *$\eta_t$*：第 $t$ 步时的学习率
- *$\eta_{\min}$*：最小学习率
- *$\eta_{\max}$*：最大学习率
- *$T_{\text{cur}}$*：当前训练步数
- *$T_{\max}$*：总训练步数
- *$\pi$*：圆周率

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

#### 专家：余弦退火与线性预热的实现细节

**余弦退火**：当 `progress` 从 0 到 1，$\cos(\pi \cdot \text{progress})$ 从 1 到 -1，学习率从 $\eta_{\max}$ 平滑下降至 $\eta_{\min}$。

**线性预热公式**：
$$
\eta_t = \eta_{\text{start}} + (\eta_{\text{end}} - \eta_{\text{start}}) \times \frac{t}{T_{\text{warmup}}}
$$

**Burn 实现**（`crates/burn-optim/src/lr_scheduler/linear.rs`）：

```rust
fn step(&mut self, step: usize) -> LearningRate {
    if step >= self.num_steps {
        return self.end_learning_rate;
    }
    
    let ratio = step as f64 / self.num_steps as f64;
    self.start_learning_rate + (self.end_learning_rate - self.start_learning_rate) * ratio
}
```

**组合使用**：预热阶段用线性调度（学习率从低到高），之后用余弦退火（学习率从高平滑降至低），是现代大模型训练的标准做法。

---

## 4. 反向传播（Backpropagation）

### 4.1 自动微分（Autodiff）

#### 入门：API 用法

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

#### 进阶：链式法则

Burn 通过 `Autodiff` 后端自动实现反向传播的链式法则。

**数学原理**：
对于复合函数 $f = f^{(D)} \circ f^{(D-1)} \circ \cdots \circ f^{(1)}$，链式法则为：
$$
\frac{\partial \ell}{\partial x^{(d-1)}} = \frac{\partial \ell}{\partial x^{(d)}} \cdot \frac{\partial f^{(d)}}{\partial x^{(d-1)}}
$$

**Burn 使用示例**：

```rust
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

#### 专家：计算图构建与链式法则的具体应用

**计算图节点示意**：

```rust
type Backend = Autodiff<Wgpu>;

// 以下每个操作都生成计算图中的一个节点
let z = x.matmul(y);      // 节点1：矩阵乘法
let w = z.relu();         // 节点2：ReLU激活  
let loss = w.mean();      // 节点3：平均值

// 反向传播（自动应用链式法则）
let grads = loss.backward();
```

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
对于 $z = xy$，有：
$$
\frac{\partial z}{\partial x} = y^T, \quad \frac{\partial z}{\partial y} = x^T
$$

---

### 4.2 梯度裁剪（Gradient Clipping）

防止梯度爆炸的常用技术。

#### 入门：API 用法

```rust
use burn_optim::grad_clipping::GradientClipping;

let clipping = GradientClipping::Norm(1.0); // 梯度范数裁剪到 1.0
optimizer = optimizer.with_grad_clipping(clipping);
```

#### 进阶：范数裁剪公式

**数学公式**：
$$
g_{\text{clipped}} = \begin{cases}
g & \text{if } \|g\| \leq \text{threshold} \\
\text{threshold} \cdot \frac{g}{\|g\|} & \text{otherwise}
\end{cases}
$$

**符号解释**：
- *$g_{\text{clipped}}$*：裁剪后的梯度
- *$g$*：原始梯度
- *$\|g\|$*：梯度 $g$ 的范数（通常指 L2 范数）
- *$\text{threshold}$*：裁剪阈值，当梯度范数超过该值时进行裁剪

**Burn 源代码**：`crates/burn-optim/src/grad_clipping.rs`

```rust
pub fn clip_gradient<B: Backend, const D: usize>(
    &self,
    gradient: Tensor<B, D>,
) -> Tensor<B, D> {
    match self {
        GradientClipping::Norm(max_norm) => {
            let norm = gradient.clone().powf_scalar(2.0).sum().sqrt();
            let scale = max_norm / (norm.to_f64() + 1e-6);
            
            if scale < 1.0 {
                gradient.mul_scalar(scale)
            } else {
                gradient
            }
        }
        // ...
    }
}
```

#### 专家：按值裁剪与按范数裁剪的实现细节

##### 按值裁剪

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

##### 按范数裁剪

**公式**：
$$
g_{\text{clipped}} = \frac{\text{threshold}}{\max(\|g\|, \text{threshold})} \cdot g
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

---

## 5. 深度的价值（The Value of Depth）

深度神经网络通过堆叠多个层来学习层次化表示。

#### 入门：模块化层组合

**Burn 实现**：Burn 没有 `Sequential` 类型，深度模型通过 `#[derive(Module)]` 结构体实现层组合。

**Burn 源代码**：`crates/burn-core/src/module/base.rs`（Module trait），`crates/burn-nn/src/modules/`

```rust
use burn::{
    nn::{conv::Conv2dConfig, norm::BatchNormConfig, Linear, LinearConfig},
    prelude::*,
    tensor::activation::relu,
};

// Burn 的核心范式：#[derive(Module)] 使结构体成为可训练模块
// 宏自动生成参数遍历（供优化器使用）、设备迁移、梯度挂钩等
#[derive(Module, Debug)]
struct ConvNet<B: Backend> {
    conv1: burn::nn::conv::Conv2d<B>,
    bn1:   burn::nn::norm::BatchNorm<B, 2>,
    conv2: burn::nn::conv::Conv2d<B>,
    bn2:   burn::nn::norm::BatchNorm<B, 2>,
    fc:    Linear<B>,
}

impl<B: Backend> ConvNet<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            conv1: Conv2dConfig::new([1, 32], [3, 3]).init(device),
            bn1:   BatchNormConfig::new(32).init(device),
            conv2: Conv2dConfig::new([32, 64], [3, 3]).init(device),
            bn2:   BatchNormConfig::new(64).init(device),
            fc:    LinearConfig::new(64 * 6 * 6, 10).init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        // 逐层手动调用，直观对应 §3.5 中的 f = f^(D) ∘ ... ∘ f^(1)
        let x = relu(self.bn1.forward(self.conv1.forward(x)));
        let x = relu(self.bn2.forward(self.conv2.forward(x)));
        let [b, c, h, w] = x.dims();
        self.fc.forward(x.reshape([b, c * h * w]))
    }
}
```

> `#[derive(Module)]` 是 Burn 的核心设计：结构体字段自动成为可训练参数，`Module` trait 提供 `.to_device()`、`.no_grad()`、`.fork()` 等方法。层数越多，`forward` 中链式调用越长，直观映射了 §3.5 中"$D$ 个映射的组合"。

---

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

Burn 提供了 `SupervisedTraining` Builder 作为高级训练入口，`Learner` 封装模型 + 优化器 + 调度器。

**Burn 源代码**：`crates/burn-train/src/learner/base.rs`，`examples/mnist/src/training.rs`

```rust
use burn::{
    record::CompactRecorder,
    train::{
        Learner, SupervisedTraining,
        MetricEarlyStoppingStrategy, StoppingCondition,
        metric::{
            AccuracyMetric, LossMetric,
            store::{Aggregate, Direction, Split},
        },
    },
};

static ARTIFACT_DIR: &str = "/tmp/burn-model";

// SupervisedTraining 是训练流程的 Builder
let training = SupervisedTraining::new(ARTIFACT_DIR, dataloader_train, dataloader_valid)
    .metrics((AccuracyMetric::new(), LossMetric::new()))  // 追踪指标
    .early_stopping(MetricEarlyStoppingStrategy::new(     // 早停策略
        &LossMetric::<B>::new(),
        Aggregate::Mean,
        Direction::Lowest,
        Split::Valid,
        StoppingCondition::NoImprovementSince { n_epochs: 5 },
    ))
    .with_file_checkpointer(CompactRecorder::new())        // 保存检查点
    .num_epochs(10)
    .summary();

// Learner::new(model, optimizer, lr_scheduler)
let result = training.launch(Learner::new(
    model,
    AdamWConfig::new().init(),
    lr_scheduler,
));

let trained_model = result.model;  // 训练后的模型
```

### 6.4 早停（Early Stopping）

`MetricEarlyStoppingStrategy`（`crates/burn-train/src/learner/early_stopping.rs`）实现 `EarlyStoppingStrategy` trait，通过 `.early_stopping()` 传入 `SupervisedTraining`：

```rust
// EarlyStoppingStrategy trait 定义
pub trait EarlyStoppingStrategy: Send {
    fn should_stop(&mut self, epoch: usize, store: &EventStoreClient) -> bool;
}

// 构造早停策略：监控指标 + 聚合方式 + 方向 + 数据集 + 停止条件
let early_stopping = MetricEarlyStoppingStrategy::new(
    &LossMetric::<B>::new(),
    Aggregate::Mean,        // 对 batch 取均值
    Direction::Lowest,      // 越低越好（分类准确率用 Direction::Highest）
    Split::Valid,           // 监控验证集（不是训练集）
    StoppingCondition::NoImprovementSince { n_epochs: 5 },
);
// 通过 .early_stopping(early_stopping) 挂入 SupervisedTraining（见上方 §6.3）
```

---

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

---

## 8. 基础数学运算与实现模式

本节汇总适用于多个知识点的基础运算和通用实现模式。

### 8.1 向量运算

| 数学运算 | Burn 代码 | 对应公式 |
|---------|-----------|----------|
| 向量点积 | `(x1 * x2).sum_dim(dim)` | $\sum_i x_{1,i} x_{2,i}$ |
| L2 范数 | `x.square().sum().sqrt()` | $\|x\|_2 = \sqrt{\sum_i x_i^2}$ |
| 余弦相似度 | `dot_product / (norm_x1 * norm_x2)` | $\frac{x_1 \cdot x_2}{\|x_1\| \|x_2\|}$ |

### 8.2 数值稳定性模式

| 问题 | 解决方案 | Burn 代码示例 |
|------|----------|---------------|
| 指数溢出 | 减去最大值 | `x - x.detach().max_dim(dim)` |
| 除零错误 | 添加 epsilon | `x / (y + 1e-6)` |
| 对数负数 | 钳制最小值 | `x.clamp_min(1e-8).log()` |

### 8.3 并行计算模式

| 运算类型 | Burn 实现 | 对应数学 |
|----------|-----------|----------|
| 逐元素运算 | `x.exp()`, `x.log()` | $e^x$, $\log x$ |
| 归约运算 | `x.sum_dim(dim)`, `x.mean()` | $\sum_i x_i$, $\frac{1}{N}\sum_i x_i$ |
| 广播运算 | `x.reshape(shape).repeat_dim(dim, n)` | 维度扩展 |

### 8.4 梯度计算模式

| 操作类型 | 前向传播 | 反向传播 |
|----------|----------|----------|
| 线性变换 | `x.matmul(y)` | $\frac{\partial}{\partial x} = y^T$, $\frac{\partial}{\partial y} = x^T$ |
| 激活函数 | `x.relu()` | `mask = x > 0; grad * mask` |
| 损失函数 | `cross_entropy(logits, targets)` | $\frac{\partial\mathcal{L}}{\partial\text{logits}} = \text{softmax} - \text{one\_hot}$ |

---

## 9. 实践指南

### 9.1 实现新损失函数的步骤

1. **写出数学公式**：明确输入输出和计算过程
2. **考虑数值稳定性**：识别可能溢出或不稳定的操作
3. **转换为张量操作**：将求和、乘积等转换为 Burn API 调用
4. **实现前向传播**：实现 `forward` 方法
5. **验证梯度**：使用自动微分验证梯度正确性

### 9.2 实现新优化器的步骤

1. **定义更新公式**：明确参数更新规则
2. **实现状态管理**：定义优化器状态结构
3. **实现 `step` 方法**：应用更新公式
4. **添加配置支持**：实现 `Config` trait
5. **测试收敛性**：在简单问题上测试优化器

### 9.3 调试技巧

1. **梯度检查**：比较自动微分与数值梯度
2. **中间值检查**：在关键点打印张量值
3. **数值范围检查**：检查指数、对数运算的输入范围
4. **形状检查**：验证所有张量操作的形状匹配

---

## 总结与学习建议

本文档按知识点组织，每个主题内部从入门到专家循序渐进，覆盖《深度学习小书》第三章"训练"的核心概念在 Burn 框架中的完整实现。

### 学习路径建议

- **初学者**：只阅读各节中标注入门的部分，通过高级 API 示例快速上手
- **中级学习者**：在掌握 API 后，深入进阶部分，理解数学公式到源码的转换
- **高级学习者/框架开发者**：研究专家部分，掌握数值稳定性和底层实现细节

### 关键收获

1. **理论与实践的结合**：深度学习理论（损失函数、优化算法、反向传播）与 Burn 框架实现的对应关系
2. **数值稳定性的重要性**：实际代码中如何处理指数溢出、除零错误等数值计算问题
3. **框架设计模式**：Burn 的模块化设计、自动微分机制和多后端支持
4. **代码可读性与效率**：Rust 语言特性如何用于构建高效且安全的深度学习框架

### 扩展学习

1. **深入 Burn 源代码**：基于本文档的分析，进一步探索 Burn 的其他模块
2. **对比其他框架**：将 Burn 的实现与 PyTorch、TensorFlow 等框架进行对比
3. **实现自定义组件**：基于本文档中的模式，尝试实现自定义的损失函数或优化器
4. **性能优化研究**：分析 Burn 在不同后端（CPU、GPU）上的性能表现

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
