# PVDataset：无人机光伏组件实例分割数据集与基准

[English](README.md) | [简体中文](README_zh-CN.md)

PVDataset 提供面向**无人机影像中单块光伏组件实例分割**的数据处理工具与基准评估代码，支持标注准备、监督模型与文本提示 zero-shot 模型评估，以及大幅影像推理。

数据集覆盖中国宁波、绍兴和深圳的 22 座光伏电站。无人机倾斜影像保留了组件尺度、朝向、安装高度、遮挡、光照及密集排列等变化。每个实例对应一块独立物理光伏组件，而不是内部电池片、组件内部分区或整排阵列。

## 数据集

| 地区 | 训练电站数 | 测试电站数 | 评估用途 |
|---|---:|---:|---|
| 宁波 | 12 | 6 | 城市内测试 |
| 绍兴 | 0 | 3 | 跨城市泛化测试 |
| 深圳 | 0 | 1 | 跨城市泛化测试 |

数据以电站为单位划分。计划发布每座电站的去畸变影像及 COCO 格式的 `instances.json`。**数据下载链接和 DOI 暂未发布。** 本代码仓库不包含影像、三维模型、相机参数或模型权重。

标注通过三维组件轮廓绘制、逐图重投影和人工检查生成。部分遮挡组件保留投影轮廓，完全遮挡组件删除。COCO 标注采用左上角为原点、x 向右、y 向下的像素坐标。

## 基准评估

- **监督实例分割：**基于 **MMDetection 3.2.0**，包含 Mask R-CNN、关闭语义分支的 HTC、SOLOv2 和 Mask2Former。
- **文本提示 zero-shot 实例分割：**采用 **SAM 3 官方实现**，不训练或微调模型。固定提示词为 `solar panel`，通过 10 张训练影像的初步比较选定，测试时保持不变。
- **大图评估：**支持切片推理、边缘实例合并、重复预测抑制、COCO 检测框与掩码 AP 评估，并输出电站级及总体结果。
- **像素级评估：**对预测实例掩码取并集，计算前景与边界指标，不另行训练语义分割模型。

代码默认值为作者实际使用的基准参数；如通过命令行覆盖参数，应在报告结果时记录。SAM 3 的具体上游 commit 和服务器锁定环境暂未记录，监督模型最终权重的发布方式尚未确定。

## 仓库结构

```text
preprocess/                  重投影、标注转换、影像与标签裁切
mmdetection/
  configs/pv-detect/         四个监督模型的配置
  configs/_base_/           所需基础配置
  tools/                    训练、推理、合并、导出与评估
sam3/                       文本提示 zero-shot 推理与评估
tests/                      基于合成数据的预处理检查
docs/                       安装与使用说明
third_party/                第三方许可证
provenance/                 文件清单、来源映射与验证记录
```

`mmdetection/` 和 `sam3/` 仅包含本研究使用的工具与配置，**不是对应框架的完整源码**。使用前需单独安装官方框架。

## 开始使用

1. 按[安装与路径设置](docs/SETUP.md)准备环境，建议将预处理、MMDetection 和 SAM 3 分开安装。
2. 按[数据预处理说明](docs/PREPROCESSING.md)准备影像与 COCO 标注。人工检查流程为 **SHP → ISAT → 人工检查 → COCO**，另提供影像与标注裁切脚本。
3. 修改模型配置中的数据路径，包括 HTC 自身的 `data_root` 设置。
4. 按[基准评估说明](docs/BENCHMARK.md)执行训练、大图推理、结果汇总及像素级评估。

评估工具使用的数据结构示例：

```text
data/
  train-crop/009/{images/, instances.json}
  test-crop/013/{images/, instances.json}
  test/
    NingBo/013/{images/, instances.json}
    ShaoXing/004/{images/, instances.json}
    ShenZhen/031/{images/, instances.json}
```

以上仅为目录示例，不是完整划分清单。当前验证配置使用 `test-crop/013`；与最终实验对应的权重选择流程仍需核对，详见[安装说明](docs/SETUP.md)。

### 使用注意事项

- ISAT 转 COCO 时，一个标注对象对应一个实例，不按相同 group 编号合并。
- 重投影需要另行提供三维标注及匹配的相机参数，使用二维数据的用户无需重复执行此步骤。当前仓库尚未包含去畸变脚本。
- 检查用 GeoJSON 采用 y 取反的像素坐标，不是地理经纬度。
- 单图 GeoJSON 工具默认使用掩码 NMS，大图评估工具默认使用检测框 NMS；比较结果时应明确并统一设置。
- SAM 3 推理入口的内存保护机制依赖 Linux cgroup/systemd。
- 修改权重、数据、提示词或后处理参数后，应使用新的输出目录，避免复用不兼容的缓存结果。

安装预处理依赖后，可在仓库根目录执行检查：

```bash
python -m unittest discover -s tests -v
```

`provenance/validation.json` 区分当前检查与历史检查记录，不代表已经完成完整 GPU 实验复现。

## 文件完整性

`SHA256SUMS.txt` 提供文件校验值，用于识别内容变动或损坏，不参与程序运行。`provenance/release_inventory.json` 为文件清单，`source_manifest.json` 记录脚本来源。校验范围不包含校验文件自身、Git 元数据和 Python 缓存。

## 许可证与引用

**作者自编代码的许可证尚未确定。** 公开可见不等于已经授予开源使用许可。第三方内容遵循各自的许可证，参见[第三方声明](THIRD_PARTY_NOTICES.md)及 `third_party/`。

数据许可证将随数据发布说明，论文信息和推荐引用格式将在确定后补充。

## 问题反馈

欢迎通过 [GitHub Issues](https://github.com/plzhang1998/PVDataset/issues) 反馈问题或讨论数据与工具。报告技术问题时，请附上运行命令、框架版本和报错信息。
