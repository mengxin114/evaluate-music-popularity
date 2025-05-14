# evaluate-music-popularity

本项目旨在通过机器学习方法预测歌曲的流行度，并分析影响流行度的关键特征。

## 目录结构

```
├── main.py                      # 主程序入口
├── README.md                    # 项目说明文档
├── dataset/                     # 数据集文件夹
│   ├── high_popularity_spotify_data.csv
│   └── low_popularity_spotify_data.csv
├── models/                      # 训练好的模型文件
├── notebooks/                   # 数据探索与分析的Jupyter笔记本
├── reports/                     # 结果报告与图表
├── src/                         # 源代码文件夹
```

## 依赖环境

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- shap
- joblib

安装依赖：
```bash
pip install -r requirements.txt
```

## 快速开始

1. **准备数据**  
   将数据集放入 `dataset/` 文件夹。

2. **运行主程序**  
   在命令行中执行：
   ```bash
   python main.py
   ```

3. **输出结果**  
   - 训练好的模型保存在 `models/` 文件夹。
   - 评估报告和特征分析图表保存在 `reports/` 文件夹。

## 主要功能

- 数据预处理与特征工程
- 训练分类模型预测歌曲流行度
- 评估模型性能（分类报告、ROC曲线等）
- 分析特征对流行度的影响（特征重要性、SHAP分析）

## 结果解读

- `reports/feature_importance.png`：展示各特征对流行度预测的贡献度。
- `reports/shap_summary.png`：展示SHAP方法下特征对预测的影响。
- `reports/roc_curve.png`：模型的ROC曲线。
- `reports/classification_report.csv`：分类评估详细报告。