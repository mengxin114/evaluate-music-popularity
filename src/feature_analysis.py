import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import numpy as np
import shap


def analyze_features(X_train):
    # 定义特征类型
    numeric = ["energy", "tempo", "danceability", "loudness"]
    categorical = ["playlist_genre", "playlist_subgenre", "mode"]

    # 数值特征分析
    plt.figure(figsize=(12, 8))
    for i, col in enumerate(numeric):
        plt.subplot(2, 2, i + 1)
        sns.histplot(X_train[col], kde=True)
    plt.tight_layout()
    plt.savefig("reports/numeric_features.png")

    # 分类特征分析
    plt.figure(figsize=(12, 6))
    for i, col in enumerate(categorical):
        plt.subplot(1, 3, i + 1)
        X_train[col].value_counts().plot(kind="bar")
    plt.tight_layout()
    plt.savefig("reports/categorical_features.png")

    # 创建预处理对象
    preprocessor = ColumnTransformer(
        [("num", StandardScaler(), numeric), ("cat", OneHotEncoder(), categorical)]
    )

    return preprocessor


def analyze_feature_importance(model_path, X_train, y_train):
    # 加载模型
    pipeline = joblib.load(model_path)

    # 提取 Pipeline 中的预处理器和最终模型
    preprocessor = pipeline.named_steps[
        "preprocessor"
    ]  # 假设预处理器命名为 "preprocessor"
    model = pipeline.steps[-1][1]  # 获取 Pipeline 的最后一步模型

    # 获取经过预处理后的特征名称
    if hasattr(preprocessor, "transformers_"):
        feature_names = []
        for name, transformer, columns in preprocessor.transformers_:
            if hasattr(transformer, "get_feature_names_out"):
                feature_names.extend(transformer.get_feature_names_out(columns))
            else:
                feature_names.extend(columns)
    else:
        feature_names = X_train.columns

    # 检查模型是否支持特征重要性
    if hasattr(model, "feature_importances_"):
        # 基于特征重要性绘图
        importance = model.feature_importances_
        sorted_idx = np.argsort(importance)[::-1]

        # 只显示前20个最重要特征
        top_n = 20
        top_idx = sorted_idx[:top_n]
        plt.figure(figsize=(12, 7))
        bars = plt.barh(
            range(len(top_idx)),
            importance[top_idx][::-1],
            color=plt.cm.viridis(np.linspace(0, 1, len(top_idx))),
        )
        plt.yticks(
            range(len(top_idx)), np.array(feature_names)[top_idx][::-1], fontsize=12
        )
        plt.xlabel("Importance", fontsize=14)
        plt.title("Top Feature Importances", fontsize=16)
        plt.gca().invert_yaxis()
        # 在条形图上显示数值
        for i, bar in enumerate(bars):
            plt.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.3f}",
                va="center",
                fontsize=10,
            )
        plt.tight_layout()
        plt.savefig("reports/feature_importance.png", dpi=200)
        plt.close()
        print("Feature importance saved to reports/feature_importance.png")
    else:
        # 使用 SHAP 分析
        X_transformed = preprocessor.transform(X_train)
        # 稀疏矩阵转为稠密，并转为float类型
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()
        X_transformed = X_transformed.astype(float)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)

        shap.summary_plot(
            shap_values, X_transformed, feature_names=feature_names, show=False
        )
        plt.savefig("reports/shap_summary.png")
        print("SHAP summary plot saved to reports/shap_summary.png")
