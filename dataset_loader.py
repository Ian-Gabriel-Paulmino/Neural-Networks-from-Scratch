from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_data(test_size=0.2, random_state=42):
    # Load dataset
    data = load_breast_cancer()
    X = data.data      # shape (569, 30)
    y = data.target    # 0 = malignant, 1 = benign
    feature_names = data.feature_names

    # Normalize features (important for NN training)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # reshape y for consistency with NN (column vector)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    # Return metadata too (for visualization)
    return X_train, X_test, y_train, y_test, feature_names
