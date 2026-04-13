# 《深度学习小书》第三章公式在 Burn 框架中的源代码实现

本文档详细展示《深度学习小书》第三章中核心数学公式在 Rust 深度学习框架 [Burn](https://github.com/tracel-ai/burn) 中的具体源代码实现。每个公式都配有对应的 Burn 代码片段和详细解释。

## 1. 损失函数（Loss Functions）

### 1.1 Softmax / Logits 转换

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

### 1.2 交叉熵损失（Cross-Entropy Loss）

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

### 1.3 对比损失（Contrastive Loss / Cosine Embedding Loss）

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

## 2. 梯度下降（Gradient Descent）

### 2.1 梯度下降更新公式

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

### 2.2 动量（Momentum）实现

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

## 3. 反向传播与自动微分

### 3.1 链式法则实现

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

## 4. 训练协议相关公式

### 4.1 学习率调度：余弦退火

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

### 4.2 梯度裁剪：范数裁剪

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

## 总结：公式到代码的映射模式

通过分析 Burn 框架的源代码，我们可以看到深度学习数学公式到 Rust 代码的几种典型映射模式：

### 1. **逐元素运算的直接映射**
- 指数运算：`x.exp()` → $\exp(x)$
- 对数运算：`x.log()` → $\log(x)$
- 加法/减法：`x.add(y)` → $x + y$，`x.sub(y)` → $x - y$
- 乘法/除法：`x.mul(y)` → $x \times y$，`x.div(y)` → $x / y$

### 2. **归约运算的批量处理**
- 求和：`x.sum()` → $\sum_i x_i$
- 平均值：`x.mean()` → $\frac{1}{N}\sum_i x_i$
- 最大值：`x.max_dim(dim)` → $\max_i x_i$

### 3. **数值稳定性技巧**
- Softmax 中的减去最大值：`x - x.detach().max_dim(dim)`
- 对数运算中的小常数：`log(x + eps)`
- 除法中的防零：`x / (y + 1e-6)`

### 4. **条件逻辑实现**
- ReLU：`x.maximum(0)` → $\max(0, x)$
- 对比损失中的条件选择：`.mask_where(mask, value)`
- 梯度裁剪中的条件缩放：`if norm > threshold { gradient * (threshold/norm) }`

### 5. **自动微分抽象**
- 梯度计算：`loss.backward()` → 自动计算 $\nabla_w \mathcal{L}$
- 梯度累积：`grads.accumulate()` → 小批量梯度累积
- 参数更新：`optimizer.step()` → 自动应用优化器更新规则

## 学习建议

1. **对照学习**：阅读论文或教材中的公式时，同时在 Burn 代码库中搜索对应实现
2. **调试理解**：使用 Rust 调试器单步跟踪梯度计算过程，观察中间变量
3. **自定义实现**：尝试自己实现简单的损失函数或优化器，与 Burn 实现对比
4. **性能分析**：使用性能分析工具理解不同实现的效率差异

## 参考资料

1. Burn 源代码：https://github.com/tracel-ai/burn
2. 《深度学习小书》第三章：训练
3. PyTorch 对应实现对比：https://pytorch.org/docs/stable/nn.html
4. 自动微分原理：Baydin et al., "Automatic Differentiation in Machine Learning: a Survey" (2015)