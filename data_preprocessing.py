import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(path: str = "german_credit_data.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
    return df

def encode_and_scale(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Saving accounts'].fillna('unknown', inplace=True)
    df['Checking account'].fillna('unknown', inplace=True)

    categorical_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    scaler = StandardScaler()
    numerical_cols = ['Age', 'Job','Credit amount', 'Duration']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df, label_encoders, scaler

def generate_risk_label(df: pd.DataFrame) -> pd.DataFrame:
    credit_threshold = df['Credit amount'].median()
    duration_threshold = df['Duration'].median()
    df['Risk'] = ((df['Credit amount'] > credit_threshold) & 
                  (df['Duration'] < duration_threshold)).astype(int)
    return df
