import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_preprocess():
    # 加载数据
    low = pd.read_csv("dataset/low_popularity_spotify_data.csv")
    high = pd.read_csv("dataset/high_popularity_spotify_data.csv")

    # 添加标签并合并
    low["popularity_label"] = 0
    high["popularity_label"] = 1
    df = pd.concat([low, high]).sample(frac=1, random_state=42)
    df = df.dropna()

    # 划分训练测试集
    X = df.drop(
        ["popularity_label", "track_id", "track_href", "uri"], axis=1, errors="ignore"
    )
    y = df["popularity_label"]

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess()
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
