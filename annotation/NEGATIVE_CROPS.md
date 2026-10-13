# Negative bbox / point 标注

先按 [运行指南](README.md) 配置环境和 API，然后使用以下两个入口分别标注 random 和 nearby negative crop：

```bash
bash annotation/label_random_negative.sh \
  --dataset /path/to/train.parquet \
  --output-dir results/random-negative-preview --limit 10 --workers 4

bash annotation/label_nearby_negative.sh \
  --dataset /path/to/train.parquet \
  --output-dir results/nearby-negative-preview --limit 10 --workers 4
```

先激活所需Python环境，或设置 `PYTHON_BIN`。依赖Pillow、PyArrow、OpenAI、tqdm。API配置沿用正crop的 `ARK_MODEL`、`ARK_BASE_URL`、`ARK_API_KEY_ENV`；可用 `--model`、`--base-url`、`--api-key-env` 覆盖。脚本读取 `.project.env` 和 `.seed.env`，密钥不写入输出。正式全量运行去掉 `--limit`，使用新输出目录。

每条请求同时发送两张图片：**第一张原图，第二张 `teacher_images` 中的positive crop**，并附问题。模型直接标注负bbox和point，请求不包含参考答案。完整prompt在 `annotation/label/negative_grounding.py` 的 `build_prompt` 中。

- random：任选与问题无关、无答案证据的区域，不刻意寻找正目标的邻域。这里是模型选择的任意无关区域，**不是均匀随机抽样**。
- nearby：先在原图定位positive crop，再选择紧邻其外侧的局部环境，排除目标和答案证据，不选远处区域。

模型输出与正crop一致：

```text
<bbox>100 200 300 400</bbox>
<point>200 300</point>
```

坐标是原图的0–999归一化整数。框的位置由模型直接决定，宽高必须与positive crop的实际像素尺寸完全一致。prompt提供原图尺寸、positive尺寸和对应归一化宽高；明显不同尺寸的提案拒绝，仅校正0–999坐标量化导致的舍入误差，绝不缩放图片。最终裁剪以bbox_pixel为准，原始模型输出保存在raw_response及proposed_bbox_pixel中。point由模型输出，必须位于框内。归一化和像素坐标均保存。无法找到合法区域时返回零面积框，脚本记为失败；非法范围、反向框、框外point不静默修正。如果输入 `extra_info.bbox_pixel` 存在，额外检查与正框不重叠；缺失时该几何检查无法进行，语义位置依赖两图标注及人工抽查。

输出：`grounding.jsonl`（包含原始回答、prompt、bbox/point、crop_path及错误）、`crops/*.png`、`summary.json`、`manifest.json`。默认在全部选中行成功后导出 `train.parquet`，仅替换 `teacher_neg_images`，逐列验证其他字段保持不变。负bbox/point留在独立标注记录中，不覆盖正框元数据。

相同命令可断点续跑；成功且crop存在的记录跳过，失败重试，`--limit`在跳过成功行之前应用。配置或输入变化需要新输出目录。每个入口只接受一个Parquet文件，不支持 `--random` 输入行打乱。`--no-write-train-parquet` 可仅标坐标，`--save-crops` 可单独保存crop。模型标注不能保证语义全部正确，正式训练前应抽查。
