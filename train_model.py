import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from data_preprocessing import load_data, encode_and_scale, generate_risk_label

def train_credit_model(data_path="german_credit_data.csv", model_path="credit_risk_model.pkl"):
    df = load_data(data_path)
    df, _, _ = encode_and_scale(df)
    df = generate_risk_label(df)

    X = df.drop(columns=["Risk"])
    y = df["Risk"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                        test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    train_credit_model()
