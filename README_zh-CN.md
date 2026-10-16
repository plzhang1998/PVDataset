# 无人机光伏组件实例分割：开源候选代码

## 这是什么

这是从现有项目中提取的**候选版**，不是原项目的完整拷贝，未上传GitHub。
数据、权重和第三方框架另行获取。自编代码的许可证尚待作者确定。

保留内容：

- `preprocess/`：三维多边形重投影、SHP→ISAT→COCO、SHP直接转COCO，以及影像和COCO标签裁切。去畸变脚本后续补充。
- `mmdetection/configs/pv-detect/`：作者指定的四个监督模型配置，以及它们实际依赖的基础配置。
- `mmdetection/tools/`：训练/测试、大图拼接评估、单图GeoJSON推理和像素精度评估。
- `sam3/`：自编单文本提示zero-shot大图评估入口。
- `tests/`：仅使用合成数据的回归测试，不读取真实数据集。
- `provenance/`：来源相对路径、文件SHA256、选取理由及验证记录。

不包含MMSegmentation、模型权重、图像、训练日志、缓存、备份和完整MMDetection/SAM3源码。
后两者需要在对应环境中单独安装；这不是可独立运行的框架安装包。

## 使用顺序

1. 按 [SETUP](docs/SETUP.md) 准备独立的GIS、MMDetection、SAM3环境。
2. 将实际数据路径填入数据集基配置；HTC内重复的data_root也保持一致。
3. 按 [PREPROCESSING](docs/PREPROCESSING.md) 裁切和转换数据，先跑小样本。
4. 按 [BENCHMARK](docs/BENCHMARK.md) 运行训练和评估。每次改参数使用新的输出目录。
5. 发布前完成 [检查清单](docs/RELEASE_CHECKLIST.md)，再建立GitHub仓库并选择代码许可证。

注意：本次不改变训练参数、拼接算法、阈值或标注规则。本地配置中013参与验证及best权重选择的事实保留，
需要与服务器最终实验核对；不能因为打包完成就宣称不存在测试信息参与模型选择。

## 快速检查

在具备预处理依赖的环境、候选仓库根目录运行：

```bash
python -m unittest discover -s tests -v
```

真实GPU训练和完整城市推理不属于这次打包检查。详细验证结果见 `provenance/validation.json`。

## 特别说明

- `instances.json`是最终标注文件名。
- ISAT转COCO保持“一个object一个实例”，不按group合并。
- 单图GeoJSON工具默认mask NMS，大图评估默认bbox NMS；需要一致时显式指定参数。
- SAM3 runner保留Linux systemd/cgroup内存保护，不能直接当Windows版程序运行。
- 三维投影脚本需要未公开的相机与三维标签输入；二维数据使用者无需执行这些构建步骤。

## 文件完整性校验

`SHA256SUMS.txt`记录发布文件的SHA-256校验值，不参与程序运行。文件被修改或下载损坏后，重新计算的校验值会发生变化。该清单不包含自身、Git元数据和Python缓存；最终修改完成后需要重新生成。校验一致不代表算法正确，也不单独证明文件来源可信。

`provenance/release_inventory.json`列出当前版本的文件及大小；`source_manifest.json`保留脚本来源与当前路径的对应关系。验证记录会区分当前检查和历史检查，不能将旧版本的测试通过记录视为新版本已经通过。
