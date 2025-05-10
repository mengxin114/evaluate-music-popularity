import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


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
