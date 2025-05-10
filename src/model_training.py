from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib
from datetime import datetime
from .feature_analysis import analyze_features


def train_model(X_train, y_train):
    # 获取预处理对象
    preprocessor = analyze_features(X_train)

    # 创建模型管道
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    # 训练模型
    model.fit(X_train, y_train)

    # 保存模型
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/model_{timestamp}.pkl"
    joblib.dump(model, model_path)

    return model_path
