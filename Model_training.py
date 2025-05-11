
# =============================================================================================================================================
'''  Creating train test data splits '''
# ===============================================================================================================================================

# Split data into features (X) and labels (y)
X = df.drop('col_1', axis=1)  # Features
y = df['col_1']  # Target

# Split the data into training and test sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # test_size has approximately 3550 samples.

# sportsbooks odds = implied probabilities
x_test_sportsbooks_odds = x_test['col_39']

x_train_sportsbooks_odds = x_train['col_39']

x_test = x_test.drop('col_39', axis = 1)

x_train = x_train.drop('col_39', axis = 1)

# Train the model
my_classifier_model.fit(x_train, y_train)

# Make predictions on the test set
y_pred = my_classifier_model.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
