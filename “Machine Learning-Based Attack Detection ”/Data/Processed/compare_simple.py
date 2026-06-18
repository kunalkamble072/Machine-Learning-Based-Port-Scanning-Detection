import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_and_evaluate(path):
    df = pd.read_csv(path)

    df.columns = df.columns.str.strip()

    # detect label
    label_col = [col for col in df.columns if col.lower() in ["label", "attack type"]][0]

    df.rename(columns={label_col: "Label"}, inplace=True)

    X = df.drop("Label", axis=1)
    y = df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return accuracy_score(y_test, y_pred)


# 🔥 Compare both datasets
full_acc = train_and_evaluate("Data/Processed/balanced_40k.csv")
selected_acc = train_and_evaluate("Data/Processed/selected_features_dataset.csv")

print("\n===== RESULTS =====")
print(f"Full Dataset Accuracy: {full_acc:.4f}")
print(f"Selected Features Accuracy: {selected_acc:.4f}")