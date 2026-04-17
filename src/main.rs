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
}
