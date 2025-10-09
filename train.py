# scripts/visualize_data.py
from src.utils.visualize_dataset import visualize_dataset
from dataset_loader import get_data

def main():
    # Load the original dataset (raw, before preprocessing)
    X_train, X_test, y_train, y_test, feature_names = get_data()
    visualize_dataset(X_train, X_test, y_train, y_test, feature_names)


if __name__ == "__main__":
    main()
