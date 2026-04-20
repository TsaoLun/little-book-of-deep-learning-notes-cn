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
- **KL 散度**：`burn_nn::loss::KLDivLoss`
- **连接主义时间分类（CTC）**：`burn_nn::loss::CTCLoss`

---

## 2. 自回归模型（Autoregressive Models）

自回归模型是处理离散序列（如自然语言、代码）的核心方法。其基本思想是利用概率论中的**链式法则**，将序列的联合概率分解为逐步的条件概率之积：

$$
P(X_1 = x_1, X_2 = x_2, \dots, X_\tau = x_\tau) = \prod_{t=1}^{\tau} P(X_t = x_t \mid X_1 = x_1, \dots, X_{t-1} = x_{t-1})
$$

模型 *$f$* 在给定已知令牌 *$(x_1, \dots, x_{t-1})$* 的条件下，输出 $K$ 个 logits 组成的向量 $l_t$，表示下一个令牌的条件概率 $\hat{P}(X_t \mid X_1 = x_1, \dots, X_{t-1} = x_{t-1})$。通过逐个采样 $\tau$ 个令牌，链式法则确保生成的序列遵循联合分布——这就是自回归生成。

训练自回归模型通过最小化所有时间步上的交叉熵之和来完成：

$$
\mathcal{L} = \sum_{t=1}^{\tau} \mathcal{L}_{ce}(f(x_1, \dots, x_{t-1}; w),\; x_t)
$$

传统上监测的值不是交叉熵本身，而是**困惑度（perplexity）**，定义为交叉熵的指数。它对应于具有相同熵的均匀分布的值数量，通常更具可解释性。

### 2.1 因果模型（Causal Models）

上述训练过程如果朴素实现，需要对每个时间步 $t$ 分别做一次前向传播，而 *$t < t'$* 的大量计算会在 $t'$ 时被重复，效率极低（$\tau$ 通常为数百至数千）。

解决此问题的标准策略是设计**因果模型**：模型一次性接受完整序列 $(x_1, \dots, x_\tau)$，同时输出所有 logits $(l_1, \dots, l_\tau)$，但其计算结构保证每个位置 $t$ 的输出 $l_t$ 仅依赖于 *$x_1, \dots, x_{t-1}$*，不泄露未来信息。这对应于"不让未来影响过去"的因果约束。

在 Transformer 架构中，这种因果约束通过**注意力掩码（attention mask）**实现——一个下三角矩阵，使位置 $t$ 只能关注（attend to）位置 $t' \leq t$ 的信息。一些框架（如 PyTorch）通过类似 `is_causal=True` 的参数在注意力层内部生成掩码；Burn 采用显式生成掩码的方式，通过 `generate_autoregressive_mask` 创建掩码后传入 `TransformerEncoderInput`，语义更清晰。

#### 入门：API 用法

**Burn 源代码**：`crates/burn-nn/src/modules/attention/mask.rs`

```rust
// 生成自回归掩码：上三角为 true（遮蔽未来位置），下三角为 false（可见）
pub fn generate_autoregressive_mask<B: Backend>(
    batch_size: usize,
    seq_length: usize,
    device: &B::Device,
) -> Tensor<B, 3, Bool> {
    // tril_mask: 下三角(含对角线)=false(可见), 上三角=true(遮蔽)
    // 例如 3×3: [[false, true, true], [false, false, true], [false, false, false]]
    let mask = Tensor::<B, 2, Bool>::tril_mask([seq_length, seq_length], 0, device);
    mask.expand([batch_size, seq_length, seq_length])
}
```

**训练期间（完整序列，一次前向）**：

因果掩码使模型可以在一次前向传播中并行计算所有位置的预测，同时保证每个位置的输出等价于仅接收该位置之前输入时的输出。这极大提升了训练效率。

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

推理时需要逐个生成令牌。Burn 通过 KV 缓存（`new_autoregressive_cache`）避免重复计算历史位置的 Key/Value，每步只需处理新生成的令牌。

```rust
let mut cache = transformer.new_autoregressive_cache();

for step in 0..max_seq_len {
    let mask = generate_autoregressive_mask::<Backend>(batch_size, step + 1, &device);
    let input = TransformerEncoderInput::new(current_token).mask_attn(mask);
    let logits = transformer.forward_autoregressive_inference(input, &mut cache);
    let next_token = logits.argmax(2); // 贪婪解码
    // ...拼接 next_token 到已生成序列
}
```

### 2.2 分词器（Tokenizers）

处理自然语言时，需要将文本转换为来自有限词汇表 $\{1, \dots, K\}$ 的令牌序列。令牌的粒度可以从单个字符到整个单词不等，这种文本与令牌之间的转换由**分词器（tokenizer）**完成。

标准方法是**字节对编码（BPE）**[Sennrich et al., 2015]，它通过分层合并字符组来构建令牌表，尝试获得频率相似但长度各异的词片段——将令牌分配给长而频繁的片段以及稀有的单个符号。

Burn 本身不包含分词器实现，但可以与 Hugging Face 的 `tokenizers` 库（Rust 原生实现）结合使用：

```rust
use tokenizers::Tokenizer;

let tokenizer = Tokenizer::from_pretrained("bert-base-uncased").unwrap();
let encoding = tokenizer.encode("Hello, world!", false).unwrap();
let token_ids = encoding.get_ids(); // 得到令牌 ID 序列，可转换为 Burn 张量
```

---

## 3. 梯度下降（Gradient Descent）

除了线性回归等特定情况外，最优参数 w* 通常没有闭式解。梯度下降是最小化损失函数的通用方法：用随机 *$w_0$* 初始化参数，然后反复沿损失下降最快的方向（即梯度的反方向）更新参数。

### 3.1 随机梯度下降（SGD）

实践中使用的损失都可以表示为各样本损失的平均：$\mathcal{L}(w) = \frac{1}{N}\sum_{n=1}^N \ell_n(w)$。精确计算完整梯度计算量很大，但在样本充分洗牌的假设下，任何子集的梯度都是完整梯度的无偏估计。因此标准做法是将数据集拆分为**小批次（mini-batch）**，基于每个批次的梯度估计更新参数——这就是**小批量随机梯度下降（SGD）**。由于数据的冗余性，这种"更多步、更嘈杂梯度"的策略反而比精确梯度更高效。

#### 入门：API 用法

**Burn 实现**：`burn_optim::Sgd`

```rust
use burn_optim::{SgdConfig, Optimizer, momentum::MomentumConfig, decay::WeightDecayConfig, GradientsParams};

let config = SgdConfig::new()
    .with_momentum(Some(MomentumConfig {
        momentum: 0.9,   // 动量因子
        dampening: 0.1,  // 阻尼因子
        nesterov: true,  // Nesterov 动量
    }))
    .with_weight_decay(Some(WeightDecayConfig { penalty: 0.0001 })); // 权重衰减系数

let mut optimizer = config.init();

// 训练循环中的优化步骤（简化示例）
let learning_rate = 0.01;
let grads = loss.backward();
let grads = GradientsParams::from_grads(grads, &model);
model = optimizer.step(learning_rate, model, grads);
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

SGD 的梯度估计是嘈杂的（基于小批次而非完整数据集），这可能导致参数更新在损失景观中振荡。动量通过累积历史梯度的指数移动平均来平滑更新方向，使优化在一致的方向上加速，在振荡方向上抑制。

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
pub fn transform<const D: usize>(
    &self,
    grad: Tensor<B, D>,
    state: Option<MomentumState<B, D>>,
) -> (Tensor<B, D>, MomentumState<B, D>) {
    // 计算速度：首次迭代 v_0 = g_0，之后 v_t = μ*v_{t-1} + (1-τ)*g_t
    let velocity = if let Some(state) = state {
        grad.clone()
            .mul_scalar(1.0 - self.dampening)
            .add(state.velocity.mul_scalar(self.momentum))
    } else {
        grad.clone()
    };

    // 标准动量直接使用 v_t；Nesterov 动量使用 g_t + μ*v_t（"预看"）
    let grad = match self.nesterov {
        true => velocity.clone().mul_scalar(self.momentum).add(grad),
        false => velocity.clone(),
    };

    (grad, MomentumState::new(velocity))
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

Adam [Kingma and Ba, 2014] 是 SGD 最流行的变体。它为每个参数维护梯度的一阶矩（均值）和二阶矩（未中心化方差）的运行估计，并对更新进行自动归一化，从而避免了不同参数之间的缩放问题和训练速度差异。AdamW 是 Adam 的改进版本，将权重衰减从梯度更新中解耦，直接作用于参数本身。

#### 入门：API 用法

**Burn 实现**：`burn_optim::AdamW`（推荐）和 `burn_optim::Adam`

**Burn 源代码**：`crates/burn-optim/src/optim/adamw.rs`

```rust
use burn::optim::AdamWConfig;

let config = AdamWConfig::new()           // 默认：β1=0.9, β2=0.999, ε=1e-5, wd=1e-4
    .with_beta_1(0.9f32)
    .with_beta_2(0.999f32)
    .with_epsilon(1e-5f32)
    .with_weight_decay(1e-4f32)
    .with_cautious_weight_decay(true)     // 谨慎权重衰减变体
    .with_amsgrad(false);                 // AMSGrad 变体

// init() 返回优化器实例（与 Learner 配合使用，见 [§6.3](burn_foundations.md#63-训练循环trainstep-inferencestep)）
let optimizer = config.init();
```

> `AdamWConfig` 字段名直接对应论文符号：`beta_1`（$\beta_1$）、`beta_2`（$\beta_2$）、`epsilon`（$\epsilon$）。`#[derive(Config)]` 宏为每个字段自动生成对应的 `with_*` setter 方法。

#### 进阶：Adam 更新公式与源码对应

**数学公式**：

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad &\text{（一阶矩估计）} \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad &\text{（二阶矩估计）} \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t} \quad &\text{（偏差校正）} \\
\hat{v}_t &= \frac{v_t}{1 - \beta_2^t} \quad &\text{（偏差校正）} \\
\Delta w &= \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} \quad &\text{（归一化更新量）}
\end{aligned}
$$

**符号解释**：
- *$m_t$*：梯度的一阶矩（指数移动平均）
- *$v_t$*：梯度平方的二阶矩（指数移动平均）
- *$\beta_1, \beta_2$*：矩估计的衰减率
- *$\hat{m}_t, \hat{v}_t$*：偏差校正后的矩估计（补偿初始化为零导致的偏差）
- *$\epsilon$*：防止除零的小常数

**AdamW 的区别**：权重衰减直接作用于参数 $w_{t+1} = w_t(1 - \eta\lambda) - \eta \Delta w$，而非添加到梯度中。

**Burn 源代码**（自适应矩估计部分，位于 `crates/burn-optim/src/optim/adamw.rs`）：

```rust
pub fn transform<B: Backend, const D: usize>(
    &self,
    grad: Tensor<B, D>,
    state: Option<AdaptiveMomentumState<B, D>>,
) -> (Tensor<B, D>, AdaptiveMomentumState<B, D>) {
    let factor_1 = 1.0 - self.beta_1;
    let factor_2 = 1.0 - self.beta_2;

    let state = if let Some(mut state) = state {
        // m_t = β1 * m_{t-1} + (1 - β1) * g_t
        state.moment_1 = state.moment_1.mul_scalar(self.beta_1)
            .add(grad.clone().mul_scalar(factor_1));
        // v_t = β2 * v_{t-1} + (1 - β2) * g_t^2
        state.moment_2 = state.moment_2.mul_scalar(self.beta_2)
            .add(grad.square().mul_scalar(factor_2));
        state.time += 1;
        state
    } else {
        // 初始化：m_0 = (1 - β1) * g_0，v_0 = (1 - β2) * g_0^2
        let moment_1 = grad.clone().mul_scalar(factor_1);
        let moment_2 = grad.square().mul_scalar(factor_2);
        AdaptiveMomentumState { time: 1, moment_1, moment_2, max_moment_2: None }
    };

    let time: i32 = state.time as i32;
    // 偏差校正：m̂_t = m_t / (1 - β1^t)
    let moment_1_corrected = state.moment_1.clone()
        .div_scalar(1f32 - self.beta_1.powi(time));
    // 偏差校正：v̂_t = v_t / (1 - β2^t)
    let moment_2_corrected = state.moment_2.clone()
        .div_scalar(1f32 - self.beta_2.powi(time));
    // Δw = m̂_t / (√v̂_t + ε)
    let update_delta = moment_1_corrected
        .div(moment_2_corrected.sqrt().add_scalar(self.epsilon));

    (update_delta, state)
}
```

**AdamW 参数更新部分**（权重衰减解耦）：

```rust
fn step<const D: usize>(
    &self, lr: LearningRate, tensor: Tensor<B, D>,
    grad: Tensor<B, D>, state: Option<Self::State<D>>,
) -> (Tensor<B, D>, Option<Self::State<D>>) {
    let (raw_delta, momentum_state) = self.momentum.transform(grad, state.map(|s| s.momentum));
    let decay_rate = lr * (self.weight_decay as f64);

    // AdamW：权重衰减直接作用于参数，而非添加到梯度
    // w = w * (1 - η * λ)，而非 grad = grad + λ * w
    let decayed_tensor = if decay_rate == 0.0 {
        tensor.clone()
    } else {
        tensor.clone().mul_scalar(1.0 - decay_rate)
    };

    // w_{t+1} = w_t * (1 - η*λ) - η * Δw
    let tensor_updated = decayed_tensor - raw_delta.mul_scalar(lr);
    (tensor_updated, Some(AdamWState { momentum: momentum_state }))
}
```

---

### 3.4 学习率调度（Learning Rate Scheduling）

学习率 $\eta$ 是梯度下降中最关键的超参数。太小则收敛缓慢且容易陷入局部最小值；太大则会在好的最小值附近反弹，无法收敛。一般策略是：训练初期使用较大学习率以跳出不良局部最小值，后期逐步减小以精确收敛到损失景观的狭窄谷底。

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
fn step(&mut self) -> LearningRate {
    self.current_iter = self.current_iter.wrapping_add(1) % (self.num_iters + 1);
    // η_t = η_min + 0.5 * (η_max - η_min) * (1 + cos(π * t / T_max))
    self.min_lr
        + 0.5 * (self.max_lr - self.min_lr)
            * (1.0 + (self.current_iter as f64 / self.num_iters as f64
                * std::f64::consts::PI)
                .cos())
}
```

#### 专家：余弦退火与线性预热的实现细节

**余弦退火**：当 `current_iter / num_iters` 从 0 到 1，$\cos(\pi \cdot \text{progress})$ 从 1 到 -1，学习率从 $\eta_{\max}$ 平滑下降至 *$\eta_{\min}$*。注意 Burn 实现中使用 `wrapping_add` 和取模实现周期性重启（warm restart）。

**线性预热公式**：
$$
\eta_t = \eta_{\text{start}} + (\eta_{\text{end}} - \eta_{\text{start}}) \times \frac{t}{T_{\text{warmup}}}
$$

**Burn 实现**（`crates/burn-optim/src/lr_scheduler/linear.rs`）：

```rust
fn step(&mut self) -> LearningRate {
    // 倒计时方式：remaining_iters 递减，线性插值到 final_lr
    self.remaining_iters -= (self.remaining_iters != 0) as usize;
    self.final_lr - self.step_size * self.remaining_iters as f64
}
```

**`SchedulerReduction`：子调度器输出的合并策略**

`ComposedLrScheduler` 通过 `SchedulerReduction` 枚举控制多个子调度器学习率的合并方式，源码定义于 `crates/burn-optim/src/lr_scheduler/composed.rs`：

```rust
/// Defines how the learning rates generated by the schedulers are combined.
#[derive(Config, Debug, Copy)]
pub enum SchedulerReduction {
    /// All learning rates are averaged.
    Avg,
    /// All learning rates are summed.
    Sum,
    /// All learning rates are multiplied.
    Prod,
}
```

设各子调度器在第 $t$ 步输出的学习率为 $\eta_t^{(1)}, \eta_t^{(2)}, \dots, \eta_t^{(K)}$，三种合并策略的数学形式为：

$$
\eta_t = \begin{cases}
\dfrac{1}{K}\sum_{k=1}^{K} \eta_t^{(k)} & \text{Avg} \\[6pt]
\sum_{k=1}^{K} \eta_t^{(k)} & \text{Sum} \\[6pt]
\prod_{k=1}^{K} \eta_t^{(k)} & \text{Prod（默认）}
\end{cases}
$$

**Burn 实现**（`step()` 方法核心逻辑）：

```rust
fn step(&mut self) -> LearningRate {
    let mut step = match self.reduction {
        SchedulerReduction::Avg  => 0.0,
        SchedulerReduction::Sum  => 0.0,
        SchedulerReduction::Prod => 1.0,   // 乘法初始值为 1
    };
    let num_scheduler = self.schedulers.len() as f64;

    for lr in self.schedulers.iter_mut().map(|s| /* 每个子调度器的 step() */) {
        step = match self.reduction {
            SchedulerReduction::Avg  => step + (lr / num_scheduler), // 累加后除以数量
            SchedulerReduction::Sum  => step + lr,
            SchedulerReduction::Prod => step * lr,
        }
    }
    step
}
```

**三种策略的适用场景**：

| 策略 | 数学语义 | 典型用法 |
|------|---------|---------|
| `Prod`（默认） | 各调度器作为**乘法因子**独立调节 | 预热因子 $(0 \to 1)$ × 余弦退火因子 $(1 \to 0)$，两个调度器各自独立地将学习率缩放到 $[0,1]$ 范围，相乘后自然得到先升后降曲线 |
| `Avg` | 各调度器贡献等权重**平均学习率** | 多个同类调度器实验，取集成平均；或不同阶段调度器平滑过渡 |
| `Sum` | 各调度器贡献**叠加学习率** | 基础衰减调度 + 周期性脉冲调度（如 cyclical LR），两者相加 |

> 注意：默认值 `SchedulerReduction::Prod` 通过 `#[config(default = "SchedulerReduction::Prod")]` 设置。这意味着调用 `ComposedLrSchedulerConfig::new()` 时无需手动指定合并策略，预热 × 余弦退火的组合开箱即用。若需修改，可调用 `.with_reduction(SchedulerReduction::Avg)` 切换策略。

**组合使用**：预热阶段用线性调度（学习率从低到高），之后用余弦退火（学习率从高平滑降至低），是现代大模型训练的标准做法。非常大的模型训练可能在数千个 GPU 上花费数月，成本高达数百万美元，训练期间通常需要根据损失演变动态手动调整学习率。

---

## 4. 反向传播（Backpropagation）

梯度下降需要计算损失对所有参数的梯度 *$\nabla\ell|_w(w)$*。由于模型 *$f$* 和损失 $L$ 都是标准张量操作的组合，微分学中的链式法则允许我们递归地计算这些梯度。

考虑映射组合 $f = f^{(D)} \circ \cdots \circ f^{(1)}$，前向传播从 $x^{(0)} = x$ 开始逐层计算 $x^{(d)} = f^{(d)}(x^{(d-1)}; w_d)$；反向传播则从输出端开始，利用链式法则递归计算损失对每层输出的梯度。其中单个标量激活值的名称来源于神经元激活的类比，$D$ 是模型深度，各 $f^{(d)}$ 称为“层”。

关于计算成本：大部分计算用于线性操作，前向传播需要一次矩阵乘积，反向传播需要两次（乘以雅可比矩阵），因此反向传播成本约为前向传播的两倍。内存方面，推理时只需存储最苛刻的单层需求，但训练时反向传播需保留前向传播的所有中间激活以计算雅可比矩阵，内存使用与模型深度成比例增长。

### 4.1 自动微分（Autodiff）

#### 入门：API 用法

Burn 通过 `Autodiff` 后端装饰器提供自动微分功能，实现了 Autograd [Baydin et al., 2015] 算法——它跟踪张量操作并即时构建梯度算子的组合，使任意命令式张量代码都可以自动计算梯度。

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

`Autodiff<B>` 是一个泛型装饰器，可包装任意后端 `B`（如 `Wgpu`、`Cuda`、`NdArray` 等）。其内部实现：

```rust
// Autodiff 装饰器定义（crates/burn-autodiff/src/backend.rs）
#[derive(Clone, Copy, Debug, Default)]
pub struct Autodiff<B, C = NoCheckpointing> {
    _b: PhantomData<B>,
    _checkpoint_strategy: PhantomData<C>,
}

impl<B: Backend, C: CheckpointStrategy> AutodiffBackend for Autodiff<B, C> {
    type InnerBackend = B;
    type Gradients = Gradients;
    
    fn backward(tensor: AutodiffTensor<B>) -> Gradients {
        tensor.backward()
    }
}
```

注意泛型参数 `C: CheckpointStrategy`——这对应于 [§3.4](burn_foundations.md#34-学习率调度learning-rate-scheduling) 原文中提到的**检查点技术**：反向传播需保留前向传播的所有中间激活，内存与深度成比例增长。检查点技术通过仅存储部分层的激活、在反向传播时重新计算其余层来交换计算时间与内存 [Chen et al., 2016]。Burn 通过 `CheckpointStrategy` 泛型在编译期选择策略，默认 `NoCheckpointing` 保留所有激活。

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

正如原文所述，前向传播和反向传播的实现细节对程序员是隐藏的——深度学习框架能够自动构建计算梯度的操作序列。在 Burn 中，用户只需编写前向传播代码，`Autodiff` 自动处理一切。

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

// ReLU 的反向传播（框架内部自动生成，用户无需手动编写）
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

**资源使用**：关于计算成本，线性操作前向传播需要一次矩阵乘积，反向传播需要两次（分别乘以 $\frac{\partial}{\partial x}$ 和 $\frac{\partial}{\partial y}$ 的雅可比矩阵），使得反向传播成本约为前向传播的两倍。存在通过可逆层 [Gomez et al., 2017] 或检查点技术（Burn 的 `CheckpointStrategy` 泛型）来交换内存和计算的方法。

**视角转变**：深度学习成功的一个关键因素是不再试图改进通用优化方法，而是转向**设计模型本身以使其可优化**——这正是第 4-5 章中各种层设计（残差连接、归一化等）的核心动机。

---

### 4.2 梯度裁剪（Gradient Clipping）

当梯度通过多层反向传播时，可能被逐层乘以一个因子而呈指数级增大（**梯度爆炸**）或减小（**梯度消失**）。梯度消失会使训练无法进行，或导致模型不同部分以不同速度更新，降低它们的共同适应 [Glorot and Bengio, 2010]。防止梯度爆炸的标准方法是**梯度范数裁剪** [Pascanu et al., 2013]：当梯度范数超过阈值时，将其重新缩放到该阈值。

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

**Burn 源代码**：`crates/burn-optim/src/grad_clipping/base.rs`

```rust
pub fn clip_gradient<B: Backend, const D: usize>(
    &self,
    grad: Tensor<B, D>,
) -> Tensor<B, D> {
    match self {
        GradientClipping::Value(threshold) => self.clip_by_value(grad, *threshold),
        GradientClipping::Norm(max_norm) => self.clip_by_norm(grad, *max_norm),
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
    // 使用浮点类型的最小正数代替硬编码 epsilon
    let min_positive = grad.dtype().finfo()
        .unwrap_or(burn::tensor::FloatDType::F32.finfo())
        .min_positive;
    // 计算裁剪系数：threshold / (‖g‖ + ε)
    let clip_coef = threshold / norm.add_scalar(min_positive);
    // 系数限制在 [0, 1]，只有范数超过阈值时才裁剪
    let clip_coef_clamped = clip_coef.clamp_max(1.0);
    // 应用裁剪：g * min(1, threshold/‖g‖)
    grad.mul(clip_coef_clamped.unsqueeze())
}
```

---

## 5. 深度的价值（The Value of Depth）

正如“深度学习”一词所示，实用的模型通常是长系列映射的组合 $f = f^{(D)} \circ \cdots \circ f^{(1)}$。用梯度下降训练会使各层产生复杂的共同适应——尽管优化过程是渐进且局部的，但深度模型能够逐层变形输入空间的表示，直到数据变得线性可分。理论结果表明，对于固定的计算预算或参数数量，增加深度会导致模型能表示的映射复杂度更高 [Telgarsky, 2016]。当前最先进的性能需要数十层的模型，如残差网络（见 [§5.2 卷积神经网络](burn_deepmodels.md#52-卷积神经网络卷积神经网络-cnn)）或 Transformer（见 [§5.3 Transformer 架构](burn_deepmodels.md#53-transformer-架构)）。

#### 入门：模块化层组合

与 PyTorch 的 `nn.Sequential` 不同，Burn 通过 `#[derive(Module)]` 结构体显式声明各层并在 `forward` 中手动组合——层的调用链直接对应数学上的映射组合 $f^{(D)} \circ \cdots \circ f^{(1)}$，语义更清晰。

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
        // 逐层手动调用，直观对应 [§3.5](burn_deepmodels.md#41-层的概念) 中的 f = f^(D) ∘ ... ∘ f^(1)
        let x = relu(self.bn1.forward(self.conv1.forward(x)));
        let x = relu(self.bn2.forward(self.conv2.forward(x)));
        let [b, c, h, w] = x.dims();
        self.fc.forward(x.reshape([b, c * h * w]))
    }
}
```

> `#[derive(Module)]` 是 Burn 的核心设计：结构体字段自动成为可训练参数，`Module` trait 提供 `.to_device()`、`.no_grad()`、`.fork()` 等方法。层数越多，`forward` 中链式调用越长，直观映射了 [§3.5](burn_deepmodels.md#41-层的概念) 中"$D$ 个映射的组合"。

#### 进阶：Module trait 的核心接口

`#[derive(Module)]` 宏自动为结构体实现 `Module<B>` trait，该 trait 定义了深度学习模块的通用接口：

**Burn 源代码**：`crates/burn-core/src/module/base.rs`

```rust
pub trait Module<B: Backend>: Clone + Send + Debug {
    type Record: Record<B>;
    
    fn collect_devices(&self, devices: Devices<B>) -> Devices<B>; // 收集所有子模块所在设备
    fn devices(&self) -> Devices<B>;                         // 获取模块所在设备列表
    fn fork(self, device: &B::Device) -> Self;               // 克隆到指定设备（独立梯度图）
    fn to_device(self, device: &B::Device) -> Self;          // 迁移到指定设备
    fn no_grad(self) -> Self;                                // 冻结参数（不参与梯度计算）
    fn num_params(&self) -> usize;                           // 可训练参数总数
    fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V);   // 只读遍历参数树
    fn map<M: ModuleMapper<B>>(self, mapper: &mut M) -> Self;// 变换参数树
    fn into_record(self) -> Self::Record;                    // 转换为可序列化记录
    fn load_record(self, record: Self::Record) -> Self;      // 从记录恢复
}
```

> `visit()` 和 `map()` 是模块系统的基石——优化器通过 `map()` 对每个参数张量应用梯度更新，`num_params()` 内部通过 `visit()` 遍历并累加参数元素数量。

训练时还需要 `AutodiffModule<B>` trait，提供训练/推理模式切换：

```rust
pub trait AutodiffModule<B: AutodiffBackend>: Module<B> + Send + Debug {
    type InnerModule: Module<B::InnerBackend>;
    
    fn valid(&self) -> Self::InnerModule;   // 转为推理模式（去除 Autodiff 包装）
}
```

**`#[derive(Module)]` 宏的工作原理**：
- 结构体中实现了 `Module` 的字段自动被识别为子模块/参数
- 自动生成 `visit()`/`map()` 的递归遍历逻辑（供优化器收集梯度使用）
- 使用 `#[module(skip)]` 标注可跳过非参数字段（如配置、缓存等）

#### 专家：Visitor/Mapper 模式与参数包装

`Module` trait 通过访问者模式（Visitor Pattern）实现参数的统一遍历与变换。宏生成的代码为每个字段递归调用 `visit`/`map`：

**Burn 源代码**：`crates/burn-core/src/module/base.rs`

```rust
// 只读访问器：遍历模块树中的所有参数张量
pub trait ModuleVisitor<B: Backend> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {}
    fn visit_int<const D: usize>(&mut self, param: &Param<Tensor<B, D, Int>>) {}
    fn visit_bool<const D: usize>(&mut self, param: &Param<Tensor<B, D, Bool>>) {}
    fn enter_module(&mut self, name: &str, container_type: &str) {}
    fn exit_module(&mut self, name: &str, container_type: &str) {}
}

// 可变映射器：变换模块树中的所有参数张量（优化器通过此应用梯度）
pub trait ModuleMapper<B: Backend> {
    fn map_float<const D: usize>(
        &mut self, param: Param<Tensor<B, D>>,
    ) -> Param<Tensor<B, D>> { /* 默认返回原值 */ }
    fn map_int<const D: usize>(
        &mut self, param: Param<Tensor<B, D, Int>>,
    ) -> Param<Tensor<B, D, Int>> { /* 默认返回原值 */ }
    // ...
}
```

参数张量被包装在 `Param<T>` 中，携带唯一 ID 和梯度开关：

**Burn 源代码**：`crates/burn-core/src/module/param/base.rs`

```rust
pub struct Param<T: Parameter> {
    pub id: ParamId,                       // 唯一标识符（优化器用此匹配梯度）
    state: SyncOnceCell<T>,                // 延迟初始化的张量值
    initialization: Option<...>,           // 惰性初始化配置
    param_mapper: ParamMapper<T>,          // load/save 时的变换
    pub(crate) require_grad: bool,         // 是否需要梯度
}
```

宏生成的 `visit` 实现示例（展开后的等价代码）：

```rust
// #[derive(Module)] 为 ConvNet 生成的 visit 方法（简化）
fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V) {
    visitor.enter_module("conv1", "Conv2d");
    self.conv1.visit(visitor);        // 递归进入子模块
    visitor.exit_module("conv1", "Conv2d");
    visitor.enter_module("bn1", "BatchNorm");
    self.bn1.visit(visitor);
    visitor.exit_module("bn1", "BatchNorm");
    // ... 对每个字段递归
}
```

这一设计使得 `num_params()` 只需要一个简单的 visitor 实现即可统计全模型参数量，而优化器通过 `map()` 可以一次性对所有参数应用梯度更新——无需手动列举参数。

---

## 6. 训练协议（Training Protocols）

训练深度网络需要一整套协议，以充分利用计算和数据，确保模型在新数据上表现良好。正如 [§3.6](burn_foundations.md#61-数据集与分割) 所述，这至少需要三组互不相交的数据：**训练集**（优化参数）、**验证集**（调整超参数、监控过拟合）、**测试集**（最终评估）。完整训练分为多个 epoch，每个 epoch 遍历一次所有训练样本。损失的典型动态是：训练损失持续下降，而验证损失可能在若干 epoch 后到达最小值后回升——这反映了过拟合。

### 6.1 数据集与分割

#### 入门：Dataset trait 与数据分割

**Burn 源代码**：`crates/burn-dataset/src/dataset/base.rs`

Burn 的数据集系统基于一个简洁的 trait：

```rust
pub trait Dataset<I>: Send + Sync {
    fn get(&self, index: usize) -> Option<I>;   // 按索引获取样本
    fn len(&self) -> usize;                      // 数据集大小
    fn is_empty(&self) -> bool { self.len() == 0 }
    fn iter(&self) -> DatasetIterator<'_, I>;    // 迭代器
}
```

使用 `PartialDataset` 实现训练/验证分割：

```rust
use burn::data::dataset::{
    Dataset,
    transform::PartialDataset,
    vision::MnistDataset,
};
use std::sync::Arc;

let dataset = Arc::new(MnistDataset::train());     // 60000 张手写数字图片
let train_set = PartialDataset::new(dataset.clone(), 0, 50000);     // 前 50000 训练
let valid_set = PartialDataset::new(dataset.clone(), 50000, 60000); // 后 10000 验证
let test_set  = MnistDataset::test();              // 独立测试集（10000 张）
```

#### 进阶：数据集变换与数据源

Burn 提供了丰富的数据集变换组合器（`crates/burn-dataset/src/transform/`），采用装饰器模式层层包装：

| 变换 | 用途 | 示例 |
|------|------|------|
| `MapperDataset<D, M, I>` | 逐样本变换（归一化、增强） | 实现 `Mapper<I, O>` trait |
| `ShuffledDataset<D, I>` | 随机打乱顺序 | `ShuffledDataset::with_seed(dataset, 42)` |
| `SamplerDataset<D, I>` | 有/无放回采样 | 过采样少数类、控制 epoch 大小 |
| `PartialDataset<D, I>` | 切片取子集 | 训练/验证分割 |

```rust
use burn::data::dataset::transform::{MapperDataset, Mapper};

// 自定义数据变换：实现 Mapper trait
struct NormalizeMapper;
impl Mapper<RawItem, ProcessedItem> for NormalizeMapper {
    fn map(&self, item: &RawItem) -> ProcessedItem {
        ProcessedItem {
            image: item.image.iter().map(|&v| v as f32 / 255.0).collect(),
            label: item.label,
        }
    }
}

let normalized = MapperDataset::new(dataset, NormalizeMapper);
```

**数据源**：除了内存数据集 `InMemDataset`（支持 JSON/CSV 加载），Burn 还提供：
- `SqliteDataset`：基于 SQLite 的大规模数据集，支持懒加载
- `HuggingfaceDatasetLoader`：直接从 Hugging Face Hub 下载数据集

```rust
use burn::data::dataset::source::huggingface::HuggingfaceDatasetLoader;

let dataset: SqliteDataset<TextItem> = HuggingfaceDatasetLoader::new("ag_news")
    .dataset("train")
    .unwrap();
```

### 6.2 数据加载器

DataLoader 负责将数据集样本组装为 mini-batch 并送入模型。[§3.3](burn_foundations.md#31-随机梯度下降sgd) 中提到 SGD 的核心思想是每次只用一部分数据估计梯度——DataLoader 正是实现这一机制的组件。

```rust
use burn::data::dataloader::DataLoaderBuilder;

let dataloader_train = DataLoaderBuilder::new(batcher)
    .batch_size(64)           // mini-batch 大小
    .shuffle(42)              // 每个 epoch 随机打乱
    .num_workers(4)           // 多线程预加载
    .build(train_set);

let dataloader_valid = DataLoaderBuilder::new(batcher)
    .batch_size(64)
    .build(valid_set);        // 验证集不需要 shuffle
```

> `Batcher` trait 定义了如何将单个样本组装为 batch tensor。通常需要实现 `Batcher<B, Item, Batch>` 对原始数据进行 padding、归一化等预处理。

### 6.3 训练循环（TrainStep / InferenceStep）

#### 入门：定义训练和验证步骤

模型需要实现 `TrainStep`（训练步）和 `InferenceStep`（验证/推理步）两个 trait：

**Burn 源代码**：`crates/burn-train/src/learner/train_val.rs`

```rust
pub trait TrainStep {
    type Input: Send + 'static;
    type Output: ItemLazy + 'static;
    
    fn step(&self, item: Self::Input) -> TrainOutput<Self::Output>;
}

pub trait InferenceStep {
    type Input: Send + 'static;
    type Output: ItemLazy + 'static;
    
    fn step(&self, item: Self::Input) -> Self::Output;
}
```

实现示例（以 MNIST 分类为例）：

```rust
impl<B: AutodiffBackend> TrainStep for Model<B> {
    type Input = MnistBatch<B>;
    type Output = ClassificationOutput<B>;
    
    fn step(&self, batch: MnistBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let output = self.forward(batch.images, batch.targets.clone());
        let loss = output.loss.clone();
        // backward() 计算梯度，返回 TrainOutput 供优化器使用
        TrainOutput::new(self, loss.backward(), output)
    }
}

impl<B: Backend> InferenceStep for Model<B> {
    type Input = MnistBatch<B>;
    type Output = ClassificationOutput<B>;
    
    fn step(&self, batch: MnistBatch<B>) -> ClassificationOutput<B> {
        self.forward(batch.images, batch.targets)  // 推理模式：不计算梯度
    }
}
```

#### 进阶：SupervisedTraining 与 Learner

`SupervisedTraining` 是完整训练流程的 Builder，`Learner` 封装模型 + 优化器 + 学习率调度器。

**Burn 源代码**：`crates/burn-train/src/learner/supervised/paradigm.rs`，`crates/burn-train/src/learner/base.rs`

```rust
// Learner 封装训练的三个核心组件
pub struct Learner<LC: LearningComponentsTypes> {
    model: LC::TrainingModel,       // 模型
    optim: LC::Optimizer,           // 优化器
    lr_scheduler: LC::LrScheduler,  // 学习率调度器
    lr: f64,                        // 当前学习率
}
```

完整训练流程示例：

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

// SupervisedTraining Builder：配置整个训练过程
let training = SupervisedTraining::new(
        ARTIFACT_DIR,
        dataloader_train,       // 训练集 DataLoader
        dataloader_valid,       // 验证集 DataLoader
    )
    .metrics((AccuracyMetric::new(), LossMetric::new()))   // 追踪准确率和损失
    .early_stopping(MetricEarlyStoppingStrategy::new(      // 早停策略（见 [§6.4](burn_foundations.md#64-指标系统metrics)）
        &LossMetric::<B>::new(),
        Aggregate::Mean,
        Direction::Lowest,
        Split::Valid,
        StoppingCondition::NoImprovementSince { n_epochs: 5 },
    ))
    .with_file_checkpointer(CompactRecorder::new())        // 保存检查点
    .num_epochs(10)
    .summary();

// launch() 启动训练循环，内部每个 epoch：
//   1. 遍历训练 DataLoader → model.step(batch) → backward → optimizer.step
//   2. 遍历验证 DataLoader → model.valid().step(batch) → 计算指标
//   3. 检查早停条件、保存检查点、更新学习率
let result = training.launch(Learner::new(
    model,
    AdamWConfig::new().init(),
    lr_scheduler,
));

let trained_model = result.model;
```

### 6.4 指标系统（Metrics）

Burn 的指标系统通过 `Metric` trait 和 `Adaptor` trait 实现解耦——模型输出只需实现 `Adaptor` 即可对接任意指标。

**Burn 源代码**：`crates/burn-train/src/metric/base.rs`

```rust
pub trait Metric: Send + Sync + Clone {
    type Input;
    fn name(&self) -> MetricName;
    fn update(&mut self, item: &Self::Input, metadata: &MetricMetadata) -> SerializedEntry;
    fn clear(&mut self);
}

// Adaptor 将模型输出转换为指标输入，实现解耦
pub trait Adaptor<T> {
    fn adapt(&self) -> T;
}
```

内置指标包括 `AccuracyMetric`、`LossMetric`、`TopKAccuracyMetric`、`CpuUse`、`CpuMemory`、`CudaMetric` 等。`MetricMetadata` 还提供当前学习率（`lr`）和训练进度信息。

### 6.5 早停（Early Stopping）

[§3.6](burn_foundations.md#65-早停early-stopping) 指出验证损失可能在若干 epoch 后开始回升（过拟合），早停策略可以在这种情况下自动终止训练。

**Burn 源代码**：`crates/burn-train/src/learner/early_stopping.rs`

```rust
pub trait EarlyStoppingStrategy: Send {
    fn should_stop(&mut self, epoch: usize, store: &EventStoreClient) -> bool;
}

// 基于指标的早停：监控指标 + 聚合方式 + 方向 + 数据集 + 停止条件
let early_stopping = MetricEarlyStoppingStrategy::new(
    &LossMetric::<B>::new(),
    Aggregate::Mean,        // 对 batch 取均值
    Direction::Lowest,      // 损失越低越好（准确率用 Direction::Highest）
    Split::Valid,           // 监控验证集（而非训练集）
    StoppingCondition::NoImprovementSince { n_epochs: 5 },  // 5个 epoch 无改善即停止
);
```

### 6.6 微调（Fine-tuning）

[§3.6](burn_foundations.md#66-微调fine-tuning) 强调微调是将预训练模型适配到下游任务的核心手段——特别是当下游任务数据有限时，预训练模型编码的统计结构提供了良好的归纳偏置。在 Burn 中通过 `load_record` 加载预训练权重，然后用 `no_grad()` 选择性冻结部分层：

```rust
// 1. 加载预训练模型权重
let pretrained_record = CompactRecorder::new()
    .load::<ModelRecord<B>>(pretrained_path, &device)
    .unwrap();
let model = ModelConfig::new().init(&device).load_record(pretrained_record);

// 2. 冻结特征提取层，只训练分类头（迁移学习）
// no_grad() 使该子模块的参数不参与梯度计算
let model = Model {
    backbone: model.backbone.no_grad(),   // 冻结
    classifier: model.classifier,          // 可训练
};

// 3. 使用较小的学习率微调
let lr_scheduler = CosineAnnealingLrSchedulerConfig::new(1e-5, num_iters).init();
```

---

## 7. 规模的好处（The Benefit of Scale）

§3.7 指出性能随数据量和模型规模的增加而提高，遵循显著的缩放定律 [Kaplan et al., 2020]。受益于这些缩放定律部分归功于深度模型可以通过增加层数或特征维度任意扩展，也归功于 SGD 一次只需要一部分数据——可以处理远超设备内存的数据集。典型视觉模型有 $10^7 - 10^8$ 个参数（$10^{18} - 10^{19}$ FLOPs），大型语言模型有 $10^8 - 10^{11}$ 个参数（$10^{20} - 10^{23}$ FLOPs），后者需要多 GPU 集群。

### 7.1 多后端支持

#### 入门：选择计算后端

Burn 的核心设计是**后端无关**——模型代码不绑定任何特定硬件。通过泛型参数 `B: Backend` 在编译期选择后端，同一份代码可以在 CPU、GPU、浏览器中运行。

```rust
use burn::backend::{Autodiff, NdArray, Wgpu};

// 开发调试：纯 Rust CPU 后端（无外部依赖）
type DevBackend = NdArray;
type DevTrainBackend = Autodiff<DevBackend>;

// 生产训练：GPU 后端
type ProdBackend = Wgpu;
type ProdTrainBackend = Autodiff<ProdBackend>;

// 同一个函数签名，不同后端直接切换
fn train<B: AutodiffBackend>(device: &B::Device) { /* ... */ }

train::<DevTrainBackend>(&NdArrayDevice::Cpu);
train::<ProdTrainBackend>(&WgpuDevice::default());
```

#### 进阶：Backend trait 架构

**Burn 源代码**：`crates/burn-backend/src/backend/base.rs`

所有后端实现统一的 `Backend` trait，定义了张量类型和操作接口：

```rust
pub trait Backend:
    FloatTensorOps<Self>     // 浮点张量操作（matmul, exp, log 等）
    + IntTensorOps<Self>     // 整数张量操作
    + BoolTensorOps<Self>    // 布尔张量操作
    + ModuleOps<Self>        // 模块操作（conv, pool, embedding 等）
    + ActivationOps<Self>    // 激活函数（relu, gelu, sigmoid 等）
    + QTensorOps<Self>       // 量化张量操作
    + TransactionOps<Self>   // 批量操作事务
    + Clone + Default + Send + Sync + 'static
{
    type Device: DeviceOps;                              // 设备类型
    type FloatTensorPrimitive: TensorMetadata + 'static; // 浮点张量原语
    type FloatElem: Element;                             // 浮点元素类型
    type IntTensorPrimitive: TensorMetadata + 'static;   // 整数张量原语
    type IntElem: Element;                               // 整数元素类型
    // ...
}
```

**可用后端一览**：

| Crate | 类型 | 适用场景 | 特点 |
|-------|------|----------|------|
| `burn-ndarray` | CPU | 开发调试、嵌入式部署 | 纯 Rust，零外部依赖 |
| `burn-wgpu` | GPU | 跨平台 GPU 训练/推理 | WebGPU（Vulkan/Metal/DX12） |
| `burn-cuda` | GPU | NVIDIA GPU 训练 | CUDA 原生，最高性能 |
| `burn-rocm` | GPU | AMD GPU 训练 | ROCm 支持 |
| `burn-tch` | GPU | 需要 LibTorch 生态 | PyTorch C++ 后端绑定 |
| `burn-candle` | GPU | Hugging Face 生态 | Candle 后端绑定 |

**装饰器后端**（包装其他后端添加功能）：

| Crate | 功能 | 用法 |
|-------|------|------|
| `burn-autodiff` | 自动微分 | `Autodiff<B>` / `Autodiff<B, BalancedCheckpointing>` |
| `burn-fusion` | 算子融合优化 | 自动合并连续操作减少内存传输 |

### 7.2 Autodiff 装饰器的类型系统

`Autodiff<B, C>` 的精妙之处在于它本身也实现了 `Backend`——可以作为泛型参数传入任何接受 `B: Backend` 的函数，同时额外实现 `AutodiffBackend` 提供梯度计算能力：

**Burn 源代码**：`crates/burn-autodiff/src/backend.rs`

```rust
#[derive(Clone, Copy, Debug, Default)]
pub struct Autodiff<B, C = NoCheckpointing> {
    _b: PhantomData<B>,
    _checkpoint_strategy: PhantomData<C>,
}

// Autodiff<B> 本身就是 Backend——可以无缝传入任何泛型函数
impl<B: Backend, C: CheckpointStrategy> Backend for Autodiff<B, C> {
    type Device = B::Device;                        // 设备类型透传
    type FloatTensorPrimitive = AutodiffTensor<B>;  // 浮点张量包装为计算图节点
    type IntTensorPrimitive = B::IntTensorPrimitive;// 整数/布尔张量无需梯度，直接透传
    // ...
}

impl<B: Backend, C: CheckpointStrategy> AutodiffBackend for Autodiff<B, C> {
    type InnerBackend = B;         // 解包后回到原始后端
    type Gradients = Gradients;
    
    fn backward(tensor: AutodiffTensor<B>) -> Gradients;         // 反向传播
    fn grad(tensor: &AutodiffTensor<B>, grads: &Gradients)       // 读取梯度
        -> Option<B::FloatTensorPrimitive>;
    fn inner(tensor: AutodiffTensor<B>) -> B::FloatTensorPrimitive; // 解包获取内部张量
}
```

> 这种设计意味着 `Autodiff<Wgpu>` 的 `Device` 就是 `WgpuDevice`，整数和布尔张量直接透传给内部后端——只有浮点张量被包装进计算图以追踪梯度。推理时通过 `model.valid()` 去除 `Autodiff` 包装，回到纯 `Wgpu` 后端，无任何额外开销。

### 7.3 分布式训练

大规模训练需要多设备并行。Burn 通过 `burn-collective` crate 提供集合通信原语，支持梯度同步：

**Burn 源代码**：`crates/burn-collective/src/api.rs`

```rust
use burn_collective::{register, all_reduce, ReduceOperation, PeerId, CollectiveConfig};

// 1. 注册当前进程到通信组
register::<B>(peer_id, device, config)?;

// 2. 全局梯度聚合（All-Reduce）
// 每个设备计算局部梯度 → all_reduce 同步求和/平均 → 所有设备获得一致的全局梯度
let synced_grads = all_reduce::<B>(peer_id, local_grads, ReduceOperation::Sum)?;

// 3. 广播（从一个节点向所有节点发送）
let weights = broadcast::<B>(peer_id, if is_root { Some(tensor) } else { None })?;
```

`burn-communication` crate 提供底层通信协议抽象（`Protocol` trait），目前支持 WebSocket 通信通道，使得分布式训练不依赖 NCCL 等 GPU 专属库。

#### 专家：规模化训练的实践考量

**梯度累积**：当单 GPU 内存不足以容纳大 batch 时，可通过 `SupervisedTraining` 的 `.grad_accumulation(n)` 方法模拟更大的 batch size——累积 $n$ 个 mini-batch 的梯度后再执行一次参数更新。

**模型参数量估算**：`model.num_params()` 通过 Visitor 模式遍历所有 `Param<Tensor>` 的元素数量求和。对于 f32 模型，内存占用约为 $\text{params} \times 4$ 字节；训练时需额外存储梯度和优化器状态，总内存约为推理的 3-4 倍。

**混合精度**：通过后端的 `supports_dtype()` 和 `dtype_usage()` 方法查询设备对不同精度的支持情况，在性能和精度之间取得平衡。

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
