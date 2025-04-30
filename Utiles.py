import pandas as pd
import numpy as np

def get_data(file_path):
    """
    Reads a CSV file and returns a DataFrame.
    
    Parameters:
    file_path (str): The path to the CSV file.
    
    Returns:
    pd.DataFrame: The DataFrame containing the data from the CSV file.
    """
    df = pd.read_csv(file_path)
    return df

def data_info(df):
    """
    Prints the information of the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to be analyzed.
    """
    print("DataFrame Information:")
    print(df.info())
    print("\nDataFrame Description:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nduplicated Values:")
    print(df.duplicated().sum())


def preproccessing_data(df):
    """
    Removes rows with missing values and duplicates from the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to be cleaned.
    
    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """
    df = df.dropna()
    df = df.drop_duplicates()
    for value in  df["BloodPressure"] :
        if value == 0:
            remove = df[df["BloodPressure"] == 0].index
            df.drop(remove, inplace=True)
    for value in  df["BMI"] :
        if value == 0:
            remove = df[df["BMI"] == 0].index
            df.drop(remove, inplace=True)
    for value in df["SkinThickness"]:
        if value == 0:
            df.replace({"SkinThickness": 0}, np.random.uniform(1.55, 3), inplace=True)
    for value in df["Glucose"]:
        if value == 0:
            df.replace({"Glucose": 0}, np.random.uniform(70, 125), inplace=True)
    
    return df

def removing_outliers(df):
    """
    Removes outliers from the DataFrame using the IQR method.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to be cleaned.
    
    Returns:
    pd.DataFrame: The cleaned DataFrame without outliers.
    """
    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Keep only the rows within bounds
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    return df


def visualize_data(df):
    """
    Visualizes the data using histograms and boxplots.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to be visualized.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set the style of seaborn
    sns.set(style="whitegrid")

    #  Correlation matrix
    plt.figure(figsize=(10,8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

    # Distribution of each feature
    df.hist(bins=20, figsize=(15, 10), color='skyblue', edgecolor='black')
    plt.suptitle('Feature Distributions')
    plt.show()

    # Pairplot to visualize relationships
    sns.pairplot(df, diag_kind='kde')
    plt.suptitle('Pairplot of Features', y=1.02)
    plt.show()

    #  Target variable count plot (assuming 'Outcome' is the target)
    if 'Outcome' in df.columns:
        sns.countplot(x='Outcome', data=df, palette='pastel')
        plt.title('Distribution of Outcome')
        plt.show()

    # Create boxplots for each column
    features = df.columns.drop('Outcome') if 'Outcome' in df.columns else df.columns
    for feature in features:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[feature], color='lightgreen')
        plt.title(f'Boxplot of {feature}')
        plt.show()


def split_data(df, target_col):
    """
    Splits the DataFrame into features and target variable.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to be split.
    target_col (str): The name of the target column.
    
    Returns:
    X (pd.DataFrame): Features DataFrame.
    y (pd.Series): Target variable Series.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def train_test_split (X, y):
    """
    Splits the data into training and testing sets.
    
    Parameters:
    X (pd.DataFrame): Features DataFrame.
    y (pd.Series): Target variable Series.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Random seed for reproducibility.
    
    Returns:
    X_train (pd.DataFrame): Training features DataFrame.
    X_test (pd.DataFrame): Testing features DataFrame.
    y_train (pd.Series): Training target variable Series.
    y_test (pd.Series): Testing target variable Series.
    """
    from sklearn.model_selection import train_test_split as tts
    from sklearn.preprocessing import StandardScaler
    import joblib

    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.3, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    joblib.dump(scaler, 'scaler.joblib')
    print("Scaler saved as scaler.joblib")
   
    return X_train, X_test, y_train, y_test 

def modeling(X_train, X_test, y_train, y_test):
    """
    Trains a Random Forest Classifier and makes predictions.
    
    Parameters:
    X_train (pd.DataFrame): Training features DataFrame.
    X_test (pd.DataFrame): Testing features DataFrame.
    y_train (pd.Series): Training target variable Series.
    y_test (pd.Series): Testing target variable Series.
    
    Returns:
    y_pred (pd.Series): Predicted target variable values.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    import joblib

    model = RandomForestClassifier(max_depth=4, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=5)
    print("Cross-validation accuracy:", scores.mean())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_train_pred = model.predict(X_train)

    joblib.dump(model, 'model.joblib')
    print("Model saved as model.joblib")
    
    return y_train ,y_pred , y_train_pred


def evaluate_model(y_test, y_pred, y_train, y_train_pred):
    """
    Evaluates the model's performance using various metrics.
    
    Parameters:
    y_test (pd.Series): True target variable values.
    y_pred (pd.Series): Predicted target variable values.
    
    Returns:
    None
    """
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
    import matplotlib.pyplot as plt

    print("Confusion Matrix Test:")
    print(confusion_matrix(y_test, y_pred))

    print("Confusion Matrix Train:")
    print(confusion_matrix(y_train, y_train_pred))

    print("\nClassification Report Test:")
    print(classification_report(y_test, y_pred))

    print("\nClassification Report Train:")
    print(classification_report(y_train, y_train_pred))

    print("\nAccuracy Score Test:")
    print(accuracy_score(y_test, y_pred))

    print("\nAccuracy Score Train:")
    print(accuracy_score(y_train, y_train_pred))

    print("\nROC AUC Score Test:")
    print(roc_auc_score(y_test, y_pred))

    print("\nROC AUC Score Train:")
    print(roc_auc_score(y_train, y_train_pred))


    fpr_Test, tpr_Test, thresholds_Test = roc_curve(y_test, y_pred)

    fpr_Train, tpr_Train, thresholds_Train = roc_curve(y_train, y_train_pred)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr_Test, tpr_Test, label='ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve Test')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(fpr_Train, tpr_Train, label='ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve Train')
    plt.legend()
    plt.show()