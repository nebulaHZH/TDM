# MAT文件格式转换说明

由于.mat文件格式不便于直接查看，项目已进行以下修改，将所有保存格式改为PNG图片和JSON日志格式。

## 修改内容

### 1. main.py 修改
- **原来**: 训练损失保存为 `training_history.mat`
- **现在**: 
  - 保存损失曲线图为 `training_loss_history.png`
  - 保存训练日志为 `training_history.json`

### 2. Diffusion_model.py 修改
- **原来**: 
  - 训练样本保存为 `train_example_epoch{epoch}.mat`
  - 测试结果保存为 `test_example_epoch{epoch}.mat`
- **现在**:
  - 训练样本保存为多个PNG图片: `train_example_epoch{epoch}_batch{i}_sample{idx}.png`
  - 测试结果保存为多个PNG图片: `test_example_epoch{epoch}_sample{sample_idx}_batch{batch_idx}.png`
  - 测试结果日志保存为: `test_results_epoch{epoch}.json`

## 新增功能

### 1. 损失曲线可视化
训练损失现在会自动生成可视化图表，包含：
- 损失变化趋势
- 网格线显示
- 高分辨率PNG格式（300 DPI）

### 2. 训练日志JSON格式
包含以下信息：
```json
{
  "train_loss_history": [损失值数组],
  "total_epochs": 总训练轮数,
  "final_loss": 最终损失值,
  "best_loss": 最佳损失值,
  "training_completed": 训练完成状态
}
```

### 3. 图像样本保存
- 自动将训练和测试图像转换为PNG格式
- 数据范围自动归一化到[0,1]
- 使用灰度colormap显示医学图像
- 每个图像包含标题和epoch信息

## 转换工具

### convert_mat_to_png.py
提供转换现有.mat文件的工具：

**使用方法**:
```bash
# 转换单个文件
python convert_mat_to_png.py path/to/file.mat

# 转换目录中所有.mat文件
python convert_mat_to_png.py path/to/directory/
```

**功能特点**:
- 自动识别数据类型（损失数据或图像数据）
- 损失数据生成曲线图和JSON日志
- 图像数据生成PNG图片
- 支持多维数据批量处理
- 自动数据范围归一化

## 优势

### 1. 易于查看
- PNG图片可以直接在任何图片查看器中打开
- 不需要MATLAB或特殊软件
- 支持在网页浏览器中查看

### 2. 跨平台兼容
- 所有操作系统都支持PNG格式
- JSON格式是标准的数据交换格式
- 便于后续数据分析和处理

### 3. 存储效率
- PNG格式通常比.mat文件更小
- JSON格式便于程序读取和处理
- 支持更好的压缩比

### 4. 便于分享
- 可以直接在文档中插入PNG图片
- JSON数据可以轻松导入其他分析工具
- 便于在论文和报告中使用

## 注意事项

1. **数据精度**: PNG图片是8位格式，如果需要保持原始精度用于进一步分析，建议同时保存numpy数组或使用其他无损格式。

2. **文件大小**: 对于大量图像，PNG文件可能比.mat文件占用更多空间，可以考虑使用压缩。

3. **兼容性**: 现有的读取.mat文件的代码需要相应修改来读取新格式。

4. **批量转换**: 使用提供的转换工具可以将现有的.mat文件批量转换为新格式。