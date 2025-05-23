from src.data_preprocessing import load_and_preprocess
from src.model_training import train_model
from src.model_evaluation import evaluate_model
from src.feature_analysis import analyze_feature_importance  # 新增导入


def main():
    # 1. 数据预处理
    X_train, X_test, y_train, y_test = load_and_preprocess()

    # 2. 模型训练
    model_path = train_model(X_train, y_train)
    print(f"Model saved at: {model_path}")

    # 3. 模型评估
    report, roc_auc = evaluate_model(model_path, X_test, y_test)
    print(f"\nClassification Report:\n{report}")
    print(f"\nROC AUC Score: {roc_auc:.4f}")

    # 4. 特征重要性分析
    analyze_feature_importance(model_path, X_train, y_train)  # 新增调用


if __name__ == "__main__":
    main()
