from Utiles import *

file_path = "diabetes.csv"
df = get_data(file_path)
data_info(df)
df = preproccessing_data(df)
df = removing_outliers(df)
df.to_csv("diabetes_cleaned.csv", index=False)
print("Data preprocessed and saved to diabetes_cleaned.csv")

visualize_data(df)
print("Data visualization completed")

x, y = split_data(df,'Outcome')
print("Data split into features and target variable")

x_train, x_test, y_train, y_test = train_test_split(x, y)
print("Data split into training and testing sets")

y_train , y_pred , y_train_pred = modeling(x_train, x_test, y_train, y_test)
print("Model trained and predictions made")

evaluate_model(y_test, y_pred, y_train, y_train_pred)
print("Model evaluation completed")

