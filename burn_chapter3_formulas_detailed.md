# Burn 框架中深度学习公式的详细源代码实现

本文档详细分析《深度学习小书》第三章中核心数学公式在 Burn 框架中的具体源代码实现，重点关注公式到代码的一一对应关系。

## 1. 基础数学运算的代码映射

### 1.1 向量运算

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

### 1.2 Softmax 与数值稳定性

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

## 2. 损失函数的公式实现

### 2.1 交叉熵损失的完整推导

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

### 2.2 对比损失的三元组形式

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

## 3. 优化算法的公式实现

### 3.1 权重衰减（L2正则化）

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

### 3.2 动量优化器

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

### 3.3 梯度裁剪

#### 3.3.1 按值裁剪

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

#### 3.3.2 按范数裁剪

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

## 4. 自动微分的链式法则实现

### 4.1 计算图构建

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

### 4.2 链式法则的具体应用

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

## 5. 学习率调度的数学实现

### 5.1 余弦退火调度

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

### 5.2 线性预热

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

## 6. 关键实现模式总结

### 6.1 数值稳定性模式

| 问题 | 解决方案 | Burn 代码示例 |
|------|----------|---------------|
| 指数溢出 | 减去最大值 | `x - x.detach().max_dim(dim)` |
| 除零错误 | 添加 epsilon | `x / (y + 1e-6)` |
| 对数负数 | 钳制最小值 | `x.clamp_min(1e-8).log()` |

### 6.2 并行计算模式

| 运算类型 | Burn 实现 | 对应数学 |
|----------|-----------|----------|
| 逐元素运算 | `x.exp()`, `x.log()` | $e^x$, $\log x$ |
| 归约运算 | `x.sum_dim(dim)`, `x.mean()` | $\sum_i x_i$, $\frac{1}{N}\sum_i x_i$ |
| 广播运算 | `x.reshape(shape).repeat_dim(dim, n)` | 维度扩展 |

### 6.3 梯度计算模式

| 操作类型 | 前向传播 | 反向传播 |
|----------|----------|----------|
| 线性变换 | `x.matmul(y)` | $\frac{\partial}{\partial x} = y^T$, $\frac{\partial}{\partial y} = x^T$ |
| 激活函数 | `x.relu()` | `mask = x > 0; grad * mask` |
| 损失函数 | `cross_entropy(logits, targets)` | $\frac{\partial\mathcal{L}}{\partial\text{logits}} = \text{softmax} - \text{one\_hot}$ |

## 7. 从公式到代码的实践指南

### 7.1 实现新损失函数的步骤

1. **写出数学公式**：明确输入输出和计算过程
2. **考虑数值稳定性**：识别可能溢出或不稳定的操作
3. **转换为张量操作**：将求和、乘积等转换为 Burn API 调用
4. **实现前向传播**：实现 `forward` 方法
5. **验证梯度**：使用自动微分验证梯度正确性

### 7.2 实现新优化器的步骤

1. **定义更新公式**：明确参数更新规则
2. **实现状态管理**：定义优化器状态结构
3. **实现 `step` 方法**：应用更新公式
4. **添加配置支持**：实现 `Config` trait
5. **测试收敛性**：在简单问题上测试优化器

### 7.3 调试技巧

1. **梯度检查**：比较自动微分与数值梯度
2. **中间值检查**：在关键点打印张量值
3. **数值范围检查**：检查指数、对数运算的输入范围
4. **形状检查**：验证所有张量操作的形状匹配

## 结论

Burn 框架的源代码清晰地反映了深度学习数学公式的实现。通过分析这些实现，我们可以：

1. **理解公式的实际计算**：看到数学公式如何转化为高效、稳定的代码
2. **学习数值稳定性技巧**：掌握防止数值问题的最佳实践
3. **掌握优化技术**：理解各种优化算法的具体实现细节
4. **扩展框架功能**：基于现有模式实现新的损失函数或优化器

这种公式到代码的映射不仅有助于理解 Burn 框架，也为在其他框架中实现类似功能提供了参考。

---

**相关文件**：
- [burn_chapter3_mapping.md](burn_chapter3_mapping.md)：第三章知识点到 Burn 代码的高级映射
- [3_foundations.md](../3_foundations.md)：《深度学习小书》第三章原文