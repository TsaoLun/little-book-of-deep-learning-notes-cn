fn main() {}

#[cfg(test)]
mod tests {
    use burn::backend::Flex;
    use burn::nn::attention::generate_autoregressive_mask;
    // use burn::backend::ndarray::NdArray;
    use burn::nn::loss::{CosineEmbeddingLossConfig, CrossEntropyLossConfig};
    use burn::prelude::*;
    use burn::tensor::{Distribution, activation};

    // type Backend = NdArray<f32>;
    type Backend = Flex<f32>;

    /// 1.1 Softmax：将 logits 转换为概率分布
    #[test]
    fn test_softmax() {
        let device = Default::default();
        let batch_size = 4;
        let num_classes = 3;

        // 将 logits 转换为概率分布
        let logits = Tensor::<Backend, 2>::random(
            [batch_size, num_classes],
            Distribution::Normal(0., 1.),
            &device,
        );
        println!("Logits 输出: {}\n", logits);
        let probabilities = activation::softmax(logits, 1); // 沿类别维度（dim=1）计算 softmax
        println!("Softmax 概率分布: {}\n", probabilities);

        // 验证：每行概率之和应为 1
        let row_sums = probabilities.clone().sum_dim(1);
        println!("每行概率之和: {}\n", row_sums);
        let expected = Tensor::<Backend, 2>::ones([batch_size, 1], &device);
        row_sums
            .to_data()
            .assert_approx_eq::<f32>(&expected.to_data(), burn::tensor::Tolerance::absolute(1e-5));

        // 验证：所有概率值应在 [0, 1] 范围内
        let min_val: f32 = probabilities.clone().min().into_scalar().elem::<f32>();
        let max_val: f32 = probabilities.max().into_scalar().elem::<f32>();
        assert!(
            min_val >= 0.0,
            "Softmax 输出最小值应 >= 0，实际为 {min_val}"
        );
        assert!(
            max_val <= 1.0,
            "Softmax 输出最大值应 <= 1，实际为 {max_val}"
        );
    }

    /// 1.2 交叉熵损失：衡量预测概率分布与真实分布的差异
    #[test]
    fn test_cross_entropy_loss() {
        let device = Default::default();
        let batch_size = 3;
        let num_classes = 3;

        // 创建交叉熵损失函数（带标签平滑和类别权重）
        let loss_fn = CrossEntropyLossConfig::new()
            .with_smoothing(Some(0.1))
            .with_weights(Some(vec![1.0, 2.0, 3.0]))
            .init::<Backend>(&device);

        // 使用示例
        let logits = Tensor::<Backend, 2>::random(
            [batch_size, num_classes],
            Distribution::Normal(0., 1.),
            &device,
        );
        println!("Logits 输入: {}\n", logits);
        let targets = Tensor::<Backend, 1, Int>::from_ints([0, 2, 1], &device);
        println!("目标类别索引: {}\n", targets);
        let loss_value = loss_fn.forward(logits, targets);
        println!("交叉熵损失值: {}\n", loss_value);

        // 验证：损失值应为标量（形状 [1]），且为正数
        assert_eq!(loss_value.dims(), [1]);
        let scalar: f32 = loss_value.into_scalar().elem();
        assert!(scalar > 0.0, "交叉熵损失应 > 0，实际为 {scalar}");
    }

    /// 1.2 交叉熵损失：完美预测应产生较低损失
    #[test]
    fn test_cross_entropy_loss_perfect_prediction() {
        let device = Default::default();

        let loss_fn = CrossEntropyLossConfig::new().init::<Backend>(&device);

        // 构造一个接近完美的预测：真实类别上的 logit 远大于其他类别
        let logits = Tensor::<Backend, 2>::from_floats([[10.0, -10.0, -10.0]], &device);
        let targets = Tensor::<Backend, 1, Int>::from_ints([0], &device);
        let loss_perfect = loss_fn.forward(logits, targets);

        // 构造一个错误的预测：真实类别上的 logit 远小于其他类别
        let logits_wrong = Tensor::<Backend, 2>::from_floats([[-10.0, 10.0, -10.0]], &device);
        let targets_wrong = Tensor::<Backend, 1, Int>::from_ints([0], &device);
        let loss_wrong = loss_fn.forward(logits_wrong, targets_wrong);

        let logits_medium = Tensor::<Backend, 2>::from_floats([[10.0, 8.0, -10.0]], &device);
        let targets_medium = Tensor::<Backend, 1, Int>::from_ints([0], &device);
        let loss_medium = loss_fn.forward(logits_medium, targets_medium);

        let perfect: f32 = loss_perfect.into_scalar().elem();
        let wrong: f32 = loss_wrong.into_scalar().elem();
        let medium: f32 = loss_medium.into_scalar().elem();
        println!("完美预测损失: {perfect}, 错误预测损失: {wrong}, 中等预测损失: {medium}");

        // 验证：完美预测的损失应远小于错误预测
        assert!(
            perfect < wrong,
            "完美预测损失 ({perfect}) 应小于错误预测损失 ({wrong})"
        );
    }

    #[test]
    fn test_contrastive_loss() {
        let loss = CosineEmbeddingLossConfig::new()
            .with_margin(0.5) // 边界值
            .init();
        let device = Default::default();

        // 三个样本对，设计为可手算的固定向量：
        //
        // 样本 0：target=1（相似），input1 == input2 → cosine=1.0 → loss = 1-1.0 = 0.0
        // 样本 1：target=-1（不相似），input1 == input2 → cosine=1.0 → loss = max(0, 1.0-0.5) = 0.5  ← 惩罚最大
        // 样本 2：target=1（相似），两向量正交 → cosine=0.0 → loss = 1-0.0 = 1.0                   ← 损失最大
        //
        // 预期平均损失 = (0.0 + 0.5 + 1.0) / 3 ≈ 0.5
        let input1 = Tensor::<Backend, 2>::from_floats(
            [
                [1.0, 0.0, 0.0, 0.0], // 样本 0
                [1.0, 0.0, 0.0, 0.0], // 样本 1
                [1.0, 0.0, 0.0, 0.0], // 样本 2
            ],
            &device,
        );
        let input2 = Tensor::<Backend, 2>::from_floats(
            [
                [1.0, 0.0, 0.0, 0.0], // 样本 0：与 input1 完全相同
                [1.0, 0.0, 0.0, 0.0], // 样本 1：与 input1 完全相同（但标注为不相似，产生惩罚）
                [0.0, 1.0, 0.0, 0.0], // 样本 2：与 input1 正交（但标注为相似，损失为 1）
            ],
            &device,
        );
        let target = Tensor::from_ints([1, -1, 1], &device); // 1 表示相似，-1 表示不相似

        let loss_value = loss.forward(input1, input2, target);
        let scalar: f32 = loss_value.clone().into_scalar().elem();
        println!("Cosine Embedding Loss: {scalar}");
        // 验证预期平均损失
        assert!(
            (scalar - 0.5).abs() < 1e-5,
            "预期损失 ≈ 0.5，实际为 {scalar}"
        );
    }

    #[test]
    fn test_generate_autoregressive_mask() {
        let device = Default::default();

        let mask = generate_autoregressive_mask::<Backend>(2, 4, &device);
        println!("Autoregressive Mask: {}\n", mask);
        mask.into_data().assert_eq(
            &TensorData::from([
                [
                    [false, true, true, true],
                    [false, false, true, true],
                    [false, false, false, true],
                    [false, false, false, false],
                ],
                [
                    [false, true, true, true],
                    [false, false, true, true],
                    [false, false, false, true],
                    [false, false, false, false],
                ],
            ]),
            false,
        );
    }

    /// 3.4 学习率调度：预热 + 余弦退火（Warmup + Cosine Decay）
    ///
    /// ComposedLrScheduler 将各子调度器输出相乘（Prod）：
    ///   线性预热（0→1 倍率）× 余弦退火（base_lr→0）= 先升后降的学习率曲线
    ///
    /// 这是现代大模型训练的标准做法：
    ///   - 预热阶段（warmup）：线性调度使学习率从低升至目标值，避免训练初期梯度爆炸
    ///   - 余弦退火阶段：学习率从高平滑降至接近 0，精确收敛到损失景观的狭窄谷底
    #[test]
    fn test_warmup_cosine_lr_scheduler() {
        use burn::lr_scheduler::{
            LrScheduler, composed::ComposedLrSchedulerConfig,
            cosine::CosineAnnealingLrSchedulerConfig, linear::LinearLrSchedulerConfig,
        };

        let warmup_steps = 5_usize;
        let total_steps = 20_usize;
        let base_lr = 1e-3_f64;

        // 线性预热倍率：1e-6 → 1.0（warmup_steps 步后保持 1.0）
        // 余弦退火：base_lr → 0（total_steps 步内平滑衰减）
        // 乘积 = 先升后降的 Warmup + Cosine Decay 曲线
        // 注：LinearLrScheduler 要求 initial_lr > 0，故用极小正数 1e-6 近似 0
        let mut scheduler = ComposedLrSchedulerConfig::new()
            .cosine(CosineAnnealingLrSchedulerConfig::new(base_lr, total_steps))
            .linear(LinearLrSchedulerConfig::new(1e-6, 1.0, warmup_steps))
            .init()
            .unwrap();

        let lrs: Vec<f64> = (0..total_steps).map(|_| scheduler.step()).collect();

        println!("预热 + 余弦退火学习率曲线（共 {total_steps} 步，前 {warmup_steps} 步预热）:");
        for (i, lr) in lrs.iter().enumerate() {
            let tag = if i < warmup_steps {
                "↑预热(线形为主+余弦退火弱->强)"
            } else {
                "↓退火(余弦)"
            };
            println!("  步 {:2} [{}]: {:.6}", i + 1, tag, lr);
        }

        // 验证预热阶段：学习率逐步上升
        assert!(
            lrs[1] > lrs[0],
            "预热阶段第 2 步 ({:.6}) 应 > 第 1 步 ({:.6})",
            lrs[1],
            lrs[0]
        );
        assert!(
            lrs[warmup_steps - 1] > lrs[0],
            "预热结束时 ({:.6}) 应 > 起点 ({:.6})",
            lrs[warmup_steps - 1],
            lrs[0]
        );

        // 验证余弦退火阶段：预热结束后学习率持续下降
        let post_warmup = &lrs[warmup_steps..];
        for i in 1..post_warmup.len() {
            assert!(
                post_warmup[i] <= post_warmup[i - 1],
                "余弦退火阶段步 {} ({:.6}) 应 ≤ 步 {} ({:.6})",
                warmup_steps + i + 1,
                post_warmup[i],
                warmup_steps + i,
                post_warmup[i - 1]
            );
        }

        // 验证最终学习率趋近 0（余弦退火在 total_steps 步时输出 0）
        assert!(
            *lrs.last().unwrap() < base_lr / 10.0,
            "最终学习率 ({:.6}) 应远小于基础学习率 ({:.6})",
            lrs.last().unwrap(),
            base_lr
        );
    }

    /// 4.1 自动微分（Autodiff）：三步模式与链式法则验证
    ///
    /// 本测试渐进式地验证 Burn 自动微分的三个核心机制：
    ///
    /// 【Phase 1】最简梯度 — 标量乘法 c = a * b
    ///   验证 ∂c/∂b = a，确认 require_grad / backward / grad 的基本流程。
    ///
    /// 【Phase 2】链式法则 — loss = mean(x @ y)
    ///   用 2×2 固定矩阵手算每一步反向传播，验证框架输出与手算完全一致：
    ///     mean backward: ∂loss/∂z = 1/N * ones
    ///     matmul backward: ∂loss/∂y = xᵀ @ (∂loss/∂z)
    ///
    /// 【Phase 3】require_grad 语义
    ///   未标记的张量调用 grad() 返回 None；标记的张量梯度可正常提取。
    #[test]
    fn test_autodiff() {
        use burn::backend::Autodiff;
        use burn::tensor::Tensor;

        // ─── 第一步：用 Autodiff 包装后端 ───
        type AutoBackend = Autodiff<Flex<f32>>;
        let device = Default::default();

        // ══════════════════════════════════════════════════════════════
        // Phase 1: 最简梯度 — 标量乘法
        // ══════════════════════════════════════════════════════════════
        //
        // 数学：c = a * b，∂c/∂b = a = 2.0
        // 目的：验证三步流程 require_grad → backward → grad
        println!("═══ Phase 1: 标量乘法 c = a * b ═══");

        let a: Tensor<AutoBackend, 1> = Tensor::from_floats([2.0_f32], &device);
        let b: Tensor<AutoBackend, 1> =
            Tensor::from_floats([3.0_f32], &device).require_grad(); // 第二步：标记

        let c = a.clone().mul(b.clone()); // 前向：c = 2 * 3 = 6
        println!("c = a * b = {}", c);

        let grads = c.backward(); // 第三步：反向传播
        let grad_b: f32 = b
            .grad(&grads)
            .expect("b 已标记 require_grad，梯度应存在")
            .into_scalar()
            .elem::<f32>();
        println!("∂c/∂b = {grad_b}  (预期 = a = 2.0)\n");
        assert!(
            (grad_b - 2.0).abs() < 1e-5,
            "∂c/∂b 应为 2.0，实际为 {grad_b}"
        );

        // ══════════════════════════════════════════════════════════════
        // Phase 2: 链式法则 — loss = mean(x @ y)
        // ══════════════════════════════════════════════════════════════
        //
        // 前向传播：
        //   x = [[1,0],[0,2]]  y = [[3,4],[5,6]]
        //   z = x @ y = [[3,4],[10,12]]
        //   loss = mean(z) = (3+4+10+12)/4 = 7.25
        //
        // 反向传播（链式法则，从输出到输入）：
        //   节点2 mean: loss = (1/N) * Σz_ij，N = 2×2 = 4
        //     ∂loss/∂z_ij = 1/N = 1/4 = 0.25（每个元素贡献相等）
        //     → ∂loss/∂z = 0.25 * ones = [[0.25,0.25],[0.25,0.25]]
        //   节点1 matmul: ∂loss/∂y = xᵀ @ (∂loss/∂z)
        //                          = [[1,0],[0,2]]ᵀ @ [[0.25,0.25],[0.25,0.25]]
        //                          = [[0.25,0.25],[0.5,0.5]]
        println!("═══ Phase 2: 链式法则 loss = mean(x @ y) ═══");

        let x: Tensor<AutoBackend, 2> =
            Tensor::from_floats([[1.0_f32, 0.0], [0.0, 2.0]], &device);
        let y: Tensor<AutoBackend, 2> =
            Tensor::from_floats([[3.0_f32, 4.0], [5.0, 6.0]], &device).require_grad();

        let z = x.clone().matmul(y.clone());
        println!("z = x @ y = {}", z);
        let loss = z.mean();
        println!("loss = mean(z) = {}", loss);

        let grads = loss.backward();
        let grad_y = y.grad(&grads).expect("y 已标记 require_grad");
        println!("∂loss/∂y = {}", grad_y);

        // 验证：与手算结果逐元素比对
        let expected = Tensor::<Flex<f32>, 2>::from_floats(
            [[0.25_f32, 0.25], [0.5, 0.5]],
            &device,
        );
        grad_y.to_data().assert_approx_eq::<f32>(
            &expected.to_data(),
            burn::tensor::Tolerance::absolute(1e-5),
        );
        println!("✓ 与手算结果一致: [[0.25, 0.25], [0.5, 0.5]]\n");

        // ══════════════════════════════════════════════════════════════
        // Phase 3: require_grad 语义
        // ══════════════════════════════════════════════════════════════
        //
        // 未标记 require_grad 的张量 → grad() 返回 None
        // 已标记 require_grad 的张量 → grad() 返回 Some(梯度)
        println!("═══ Phase 3: require_grad 语义验证 ═══");

        let p: Tensor<AutoBackend, 1> = Tensor::from_floats([5.0_f32], &device); // 无 require_grad
        let q: Tensor<AutoBackend, 1> =
            Tensor::from_floats([7.0_f32], &device).require_grad();
        let r = p.clone().mul(q.clone());
        let grads = r.backward();

        assert!(
            p.grad(&grads).is_none(),
            "未标记 require_grad 的张量不应有梯度"
        );
        println!("p.grad() = None  ✓ (p 未标记 require_grad)");

        let grad_q: f32 = q.grad(&grads).unwrap().into_scalar().elem::<f32>();
        assert!(
            (grad_q - 5.0).abs() < 1e-5,
            "∂(p*q)/∂q 应为 p = 5.0，实际为 {grad_q}"
        );
        println!("q.grad() = {grad_q}  ✓ (∂(p*q)/∂q = p = 5.0)");
    }
}
