import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from dataset_loader import load_data

X_train, X_test, y_train, y_test = load_data()

def feature_selection_objective(position):
    """
    تابع هدف انتخاب ویژگی‌ها
    ورودی:
        position : بردار پیوسته بین 0 و 1 -> بعداً با آستانه 0.5 دودویی میشه (0 یا 1)
    خروجی:
        مقدار cost (هرچی کمتر = بهتر)
    """

    mask = position >= 0.5   
    if not np.any(mask):
        return 1.0  

    X_train_selected = X_train[:, mask]
    X_test_selected = X_test[:, mask]

    clf = DecisionTreeClassifier()
    clf.fit(X_train_selected, y_train)

    y_pred = clf.predict(X_test_selected)
    acc = accuracy_score(y_test, y_pred)

    return 1 - acc   