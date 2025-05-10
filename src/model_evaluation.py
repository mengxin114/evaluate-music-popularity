from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd
import joblib


def evaluate_model(model_path, X_test, y_test):
    # 加载模型
    model = joblib.load(model_path)

    # 预测
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # 评估
    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_proba)

    # 保存评估结果
    pd.DataFrame(report).transpose().to_csv("reports/classification_report.csv")
    with open("reports/roc_auc.txt", "w") as f:
        f.write(str(roc_auc))

    return report, roc_auc
