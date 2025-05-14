from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import pandas as pd
import joblib
import matplotlib.pyplot as plt


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

    # 生成并保存ROC曲线
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("reports/roc_curve.png")
    plt.close()
    print("ROC curve saved to reports/roc_curve.png")
    return report, roc_auc
