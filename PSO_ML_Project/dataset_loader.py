from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def load_data(test_size=0.3, random_state=42):
    """
    دیتاست Iris رو بارگذاری می‌کنه و به train/test تقسیم می‌کنه.
    خروجی:
        X_train, X_test, y_train, y_test
    """
    iris = load_iris()
    X = iris.data        
    y = iris.target      

    return train_test_split(X, y, test_size=test_size, random_state=random_state)