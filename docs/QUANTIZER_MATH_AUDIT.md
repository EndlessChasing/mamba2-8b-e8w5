# E8/W5 8B PPL 退化：量化数学与方法迁移审计

本轮为只读诊断。没有改动冻结的 codec、runtime、校准或量化文件，没有重新量化模型，也没有发布或上传。GPU 组件恢复实验另行记录，本报告不把尚未完成的实验当作原因证据。

## 当前结论

1. **未发现 E8 旋转、Hessian 变换、逆变换、block LDL 或 buffered LDLQ 的移植错误。** 独立 CPU 检查通过，详见下文及 [数学检查 JSON](../reports/quantizer_math_audit_v1.json)。
2. **旧 2.7B 方法从未达到 FP16 等质量。** 它的选型参照是 W4。旧抽样 WT2 中 FP16→E8/W5 的 PPL 增幅为 **30.0992%**；旧完整 WT2 为 **27.9157%**。
3. 8B 当前完整 WT2 从 **7.244528283258169→8.708513788389588，+20.2082%**。这没有达到预先设定的质量目标，但不能仅凭这个增幅认定出现新的实现错误。两个模型的 tokenizer、执行方式及评测范围不同，**不能据跨模型增幅证明 8B 更好或更差**。
4. 现阶段应优先用组件恢复实验区分 E8 投影与两个 W5 词表的贡献。局部 proxy error、Hessian 对角范围和权重误差都不能直接归因 PPL。

## 旧结果的精确来源与对照

以下路径均位于历史工作区 `/Users/Kun/Documents/ChatGPT/LowDram ASIC Design/`；它们不是本次重新运行的结果。输入文件 SHA 已记录在 [迁移审计 JSON](../reports/quantizer_transfer_metadata_v1.json)。

### 原 16 个固定测试窗口，base 分支

| 权重 | PPL | 原始 JSON |
|---|---:|---|
| FP16 | 8.324625765838759 | `artifacts/ppl_wt2_2048_16win_fp16_v0.json` → `arms.base.ppl` |
| W4 | 11.143332174370787 | `artifacts/ppl_wt2_2048_16win_w4_v0.json` → `arms.base.ppl` |
| 最终混合校准 E8/W5 | 10.830269135913916 | `artifacts/ppl_wt2_2048_16win_e8_mixed2048_cd2_embed5_v1.json` → `arms.base.ppl` |

计算：`(10.830269135913916 / 8.324625765838759 - 1) × 100 = 30.099171308786122%`。
相对 W4 则为 `-2.809420320224354%`。这两个比较的基线不同。

范围：GPT-NeoX tokenizer，16 个分散且互不重叠的 2048-token WT2 test 窗口，共32,752个 next-token target；每窗重置 state，FP32 CE、pooled-token PPL。base 分支关闭读出值，但沿用当时的研究执行路径。这些测试窗口此前已被研究使用，不能称为新的盲测。

### 旧完整评测，2048 上下文，broadN＋FP32 state

原始来源：`artifacts/full_quality_v1/summary.json` → `ppl`；选择 `seqlen=2048`。

| 数据范围 | FP16 | W4 | E8/W5 | E8/W5 相对 FP16 |
|---|---:|---:|---:|---:|
| 完整 WT2 test | 9.098635280150152 | 12.18879772937847 | 11.638584752076923 | +27.915719% |
| 固定 C4 validation 切片 | 12.903335915184412 | 18.698365277784422 | 18.41337689063518 | +42.702453% |
| 完整 PTB test | 13.923628107122255 | 19.80378854855087 | 19.399614336938733 | +39.328731% |

WT2为288,729个target、142窗；C4是固定切片而非完整数据集。该轮带历史读出、严格逐 token state 路径。当前8B使用官方 native tokenizer、公共 Mamba 实现、完整 WT2的300,963个target及147窗 prefill；**上述 PPL 不能横向视为相同评测条件。**

## 迁移时保留与改变了什么

| 项目 | 2.7B 最终 E8/W5 | 8B v1 |
|---|---|---|
| E8 码值 | 16 bit / 8 权重 | 相同 |
| scale / damping / tuning | 0.9 / 0.01 / 2轮 | 相同 |
| B/C/dt 高精度保护 | 无 | 无 |
| 校准数据 | 16×2048 prose＋32条加权 recall，共96,661 token | 32×2048 prose，共65,536 token |
| 校准读出 | 历史 broadN 读出启用 | 原始公共模型，无该读出 |
| recall 权重 | query32、needle4，每样本归一化 | 无 recall 样本 |
| 词表 | 50,288×2560，输入/输出绑定，W5一份 | 256,000×4096，输入/输出独立，W5两份 |
| 词表独立参数占比 | 4.7635% | 25.4601% |
| in_proj 的 B/C/dt 行 | 336/10576，3.1770% | 2176/18560，11.7241% |

历史校准精确来源：`artifacts/hessian_train_mixed2048_manifest_v1.json`；历史量化参数来自 `artifacts/e8_mixed2048_cd2_embed5_manifest_v1.json`。当前源为 [quantization_v1.json](../reports/quantization_v1.json)。旧方法没有完整移植其私有校准流程；8B采用独立公开协议。

AST 函数体比较确认 `vector_quantize`、`rotate_last`、`block_ldl`、`decode_e8` 与旧实现相同。`rotation_factors` 的改动是先用 FP64 生成 DCT 系数，再转FP32执行，以降低非二次幂轴的数值误差；它保持同一数学变换。

## 独立数学检查

令 `B=diag(balance)`、`U/V`为输入/输出随机符号矩阵，`Rn/Rm`为代码定义的正交右乘变换。实际表达式为：

```text
Hreg = H / mean(diag(H)) + damping * I
Wr   = Rm.T @ V @ W @ B @ U @ Rn
Hr   = Rn.T @ U @ inv(B) @ Hreg @ inv(B) @ U @ Rn
Ŵ    = V @ Rm @ (scale * Q) @ Rn.T @ U @ inv(B)
```

该变换保持线性层代理目标 `tr((Ŵ-W) Hreg (Ŵ-W).T)`。输入均衡在权重上乘、在 Hessian 两侧除，是一致的；输出正交旋转保持目标的 Frobenius 范数。逆旋转在进入 Mamba 非线性之前完成，ngroups=8 不要求这些旋转与 grouped RMSNorm 或 SSM 更新交换顺序。

新脚本 [audit_quantizer_math.py](../scripts/audit_quantizer_math.py) 独立检查：

| 检查 | 实测结果 |
|---|---|
| 权重旋转 vs 显式 Kronecker 矩阵 | 最大相对差4.49e-7 |
| Hessian 旋转 vs 显式表达式 | 最大相对差5.48e-7 |
| 未量化时完整正逆变换 | 最大相对差3.26e-7 |
| 变换前后加权误差目标 | 最大相对差1.72e-7 |
| block LDL 重建 | 最大相对差3.05e-7 |
| 实际18560维输出轴 DCT145×H128 正逆变换 | 4.22e-7 |
| buffered LDLQ vs 上游非 buffered LDLQ | 0轮、2轮均 values/indices 完全一致 |
| E8快速量化 vs FP64枚举65,536向量最近邻 | 4种幅度×32向量全部匹配最近距离；最大差1.42e-14 |

这些是 CPU 合成算例与实际轴尺寸检查，不是完整8B误差上界，也不能证明默认 scale、damping 或 LDLQ 是最优。现有全模型507张量解码 SHA 审计另见 [decoded_weight_audit.json](../reports/decoded_weight_audit.json)。

## 当前量化误差分布

下面是每层误差指标的中位数；proxy error 指相对于原校准输入的线性输出误差范数，不是PPL误差。

| 模型/矩阵 | 相对权重误差中位数 | 相对校准输出误差中位数 |
|---|---:|---:|
| 2.7B in_proj | 36.3433% | 17.9318% |
| 8B in_proj | 37.4813% | 14.6360% |
| 2.7B out_proj | 39.7434% | 17.4692% |
| 8B out_proj | 42.5597% | 19.9640% |

8B out_proj 的 proxy error 最大为 layer18 **22.9938%**，其次 layer23 **22.9399%**、layer17 **22.8473%**。in_proj 最大为 layer25 **16.3348%**、layer26 **16.2126%**、layer18 **16.0730%**。

layer55.out_proj 的权重误差最高 **48.5613%**，其校准输出误差却是该族最低 **4.6757%**。这说明按权重MSE直接挑“最坏层”会误导：原激活协方差的方向性很重要。两个W5词表的权重误差分别为 input **5.5476%**、head **6.6201%**，无法从百分比直接推算其logit/PPL贡献。

## 阻尼诊断的实际边界

先看全部元数据，out_proj的`max(diag H)/min(diag H)`中位数约2.99e8。但只看极值容易高估问题。CPU mmap读取三份原Hessian，并核对其校准SHA，得到：

| 层 | `Hii/mean(Hii)<0.01`通道 | 这些通道占原trace | 中位通道的阻尼/原对角值 |
|---|---:|---:|---:|
| out18 | 9/8192，0.1099% | 0.0000369% | 1.2714% |
| out43 | 5/8192，0.0610% | 0.00000126% | 1.5776% |
| out55 | 473/8192，5.7739% | 0.0312902% | 17.5519% |

因此，极大对角动态范围在前两例主要由少量近零通道造成，尚不足以支持“阻尼广泛保护了死通道，导致主要PPL损失”。layer55存在更明显的非均匀性，但其校准输出误差最低，也不能仅据此优先归因。没有做阻尼扫描、特征值分解或新的GPU实验；对角比值也不是矩阵条件数。

## 当前待验证原因，按诊断优先级排列

1. **E8投影和两个W5词表的正常有损误差及其相互作用。** 先完成FP16/W5词表与FP16/E8投影的交叉恢复。历史2.7B同一初版E8投影中，仅词表FP16→W5即使PPL从10.380478437456313变为11.136981018858211，增加7.2877%；这个历史实例支持检查词表，但不能替代8B实验。
2. **out_proj与in_proj中特定功能行的敏感度。** 当前out_proj代理输出误差更大；8B B/C/dt行占比增至11.72%，而当前目标对输出行没有按后续状态更新敏感度加权。需要分别恢复in/out及z/x/B/C/dt，不能直接宣布其中一组是主因。
3. **教师输入校准与完整量化网络运行输入的差异。** 当前所有Hessian来自原模型，未使用前层量化后逐步漂移的输入；局部二次线性目标也不包含SSM递归和后续非线性的完整损失。是否有必要做顺序校准或恢复训练，要看组件实验与可接受的额外容量。
4. **从旧模型直接沿用 scale0.9、damping0.01、2轮。** 它们没有在8B上证明最优；旧混合校准与当前prose校准也不同。只有在明确主要误差来源后，才值得做小范围开发集扫描，避免将试验数和算力投入到无关部件。

旋转方向、均衡逆向、Hessian转置、LDLQ缓冲索引和序列化损坏目前均无支持证据。任何后续修复的质量判断仍需相同8B协议下的PPL/MK；已经观察过的test结果应继续如实标记。

## 复现本轮 CPU 数学检查

```sh
python scripts/audit_quantizer_math.py \
  --report reports/quantizer_math_audit_v1.json \
  --hessian-dir calibration/v1
```

默认只读取out18、out43、out55三份Hessian；不复制大文件，也不使用GPU。整个本轮CPU运行13.68秒。
