import matplotlib.pyplot as plt
import pandas as pd

def visualize_dataset(X_train, X_test, y_train, y_test, feature_names, num_features=5):
    # Combine training and test sets for inspection
    df_train = pd.DataFrame(X_train, columns=feature_names)
    df_train['target'] = y_train.flatten()
    df_train['set'] = 'train'

    df_test = pd.DataFrame(X_test, columns=feature_names)
    df_test['target'] = y_test.flatten()
    df_test['set'] = 'test'

    df = pd.concat([df_train, df_test], ignore_index=True)

    print("Combined dataset shape:", df.shape)
    print("Training samples:", df_train.shape[0])
    print("Test samples:", df_test.shape[0])
    print("\nTarget distribution (train/test):")
    print(df.groupby('set')['target'].value_counts())

    # Plot distributions of first few normalized features
    df.iloc[:, :num_features].hist(figsize=(10, 8))
    plt.suptitle("Feature distributions (normalized)")
    plt.show()

    # Plot target distribution by split
    df.groupby('set')['target'].value_counts().unstack().plot(
        kind='bar', title="Target distribution per set (0=malignant, 1=benign)"
    )
    plt.show()
