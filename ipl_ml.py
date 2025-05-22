import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import numpy as np
import warnings

warnings.filterwarnings('ignore') # Suppress warnings for cleaner output

# --- 1. Simulate Loading and Preprocessing Data ---
# Since I cannot directly access your files, I'll create dummy dataframes
# resembling the structure of your IPL dataset for demonstration purposes.

# Expanded matches.csv to have more entries and ensure multiple winners
# This time, using a more programmatic way to extend the data
base_matches_data = {
    'id': range(1, 21),
    'season': [2008, 2008, 2008, 2009, 2009, 2009, 2010, 2010, 2010, 2011,
               2011, 2011, 2012, 2012, 2012, 2013, 2013, 2013, 2014, 2014],
    'city': ['Bangalore', 'Chandigarh', 'Delhi', 'Durban', 'Cape Town', 'Johannesburg',
             'Mumbai', 'Kolkata', 'Chennai', 'Delhi', 'Mumbai', 'Chennai',
             'Mumbai', 'Kolkata', 'Chennai', 'Hyderabad', 'Bangalore', 'Delhi',
             'Abu Dhabi', 'Sharjah'],
    'date': ['2008-04-18', '2008-04-19', '2008-04-19', '2009-04-18', '2009-04-18', '2009-04-19',
             '2010-03-12', '2010-03-13', '2010-03-14', '2011-04-09', '2011-04-10', '2011-04-11',
             '2012-04-04', '2012-04-05', '2012-04-06', '2013-04-03', '2013-04-04', '2013-04-06',
             '2014-04-16', '2014-04-17'],
    'team1': ['Royal Challengers Bangalore', 'Kings XI Punjab', 'Delhi Daredevils',
              'Royal Challengers Bangalore', 'Chennai Super Kings', 'Deccan Chargers',
              'Mumbai Indians', 'Kolkata Knight Riders', 'Chennai Super Kings',
              'Delhi Daredevils', 'Mumbai Indians', 'Chennai Super Kings',
              'Mumbai Indians', 'Kolkata Knight Riders', 'Chennai Super Kings',
              'Sunrisers Hyderabad', 'Royal Challengers Bangalore', 'Delhi Daredevils',
              'Kolkata Knight Riders', 'Delhi Daredevils'],
    'team2': ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
              'Delhi Daredevils', 'Rajasthan Royals', 'Kolkata Knight Riders',
              'Delhi Daredevils', 'Deccan Chargers', 'Kolkata Knight Riders',
              'Mumbai Indians', 'Delhi Daredevils', 'Kochi Tuskers Kerala',
              'Chennai Super Kings', 'Delhi Daredevils', 'Royal Challengers Bangalore',
              'Royal Challengers Bangalore', 'Mumbai Indians', 'Kolkata Knight Riders',
              'Mumbai Indians', 'Royal Challengers Bangalore'],
    'toss_winner': ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
                    'Royal Challengers Bangalore', 'Rajasthan Royals', 'Deccan Chargers',
                    'Mumbai Indians', 'Deccan Chargers', 'Chennai Super Kings',
                    'Mumbai Indians', 'Mumbai Indians', 'Kochi Tuskers Kerala',
                    'Mumbai Indians', 'Delhi Daredevils', 'Royal Challengers Bangalore',
                    'Royal Challengers Bangalore', 'Royal Challengers Bangalore', 'Kolkata Knight Riders',
                    'Kolkata Knight Riders', 'Royal Challengers Bangalore'],
    'toss_decision': ['field', 'bat', 'bat', 'field', 'bat', 'bat',
                      'field', 'field', 'bat', 'bat', 'field', 'field',
                      'field', 'field', 'field', 'field', 'field', 'field',
                      'field', 'field'],
    'winner': ['Kolkata Knight Riders', 'Chennai Super Kings', 'Delhi Daredevils',
               'Royal Challengers Bangalore', 'Rajasthan Royals', 'Deccan Chargers',
               'Mumbai Indians', 'Deccan Chargers', 'Chennai Super Kings',
               'Mumbai Indians', 'Mumbai Indians', 'Chennai Super Kings',
               'Mumbai Indians', 'Delhi Daredevils', 'Chennai Super Kings',
               'Royal Challengers Bangalore', 'Mumbai Indians', 'Kolkata Knight Riders',
               'Kolkata Knight Riders', 'Royal Challengers Bangalore'],
    'win_by_runs': [140, 33, 3, 6, 14, 11, 4, 24, 9, 38, 8, 11, 8, 8, 5, 7, 9, 13, 41, 8],
    'win_by_wickets': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}

# Duplicate the base data to create a larger dataset
num_duplicates = 5 # Create 5 times the original data, resulting in 100 entries
full_matches_data = {key: [] for key in base_matches_data}

for i in range(num_duplicates):
    for key, value_list in base_matches_data.items():
        if key == 'id':
            # Ensure unique IDs across duplicates
            full_matches_data[key].extend([id_val + i * len(base_matches_data['id']) for id_val in value_list])
        else:
            full_matches_data[key].extend(value_list)

matches_df = pd.DataFrame(full_matches_data)

# Dummy teams.csv (Can remain the same as the teams appearing in matches_df are covered)
teams_data = {
    'team_id': range(1, 15),
    'team_name': ['Mumbai Indians', 'Chennai Super Kings', 'Kolkata Knight Riders',
                  'Royal Challengers Bangalore', 'Delhi Daredevils', 'Kings XI Punjab',
                  'Rajasthan Royals', 'Sunrisers Hyderabad', 'Deccan Chargers',
                  'Kochi Tuskers Kerala', 'Pune Warriors', 'Gujarat Lions',
                  'Rising Pune Supergiant', 'Lucknow Super Giants']
}
teams_df = pd.DataFrame(teams_data)

print("Original Matches DataFrame (first 5 rows):")
print(matches_df.head())
print("\nOriginal Teams DataFrame (first 5 rows):")
print(teams_df.head())

# Feature Engineering (Basic)
# For simplicity, we'll use team1, team2, toss_winner, toss_decision, and city as features.
# A more robust model would incorporate historical performance, player stats, etc.

# Identify all unique teams
all_teams = pd.concat([matches_df['team1'], matches_df['team2'], matches_df['winner']]).unique()
team_encoder = LabelEncoder()
team_encoder.fit(all_teams)

# Encode categorical features
matches_df['team1_encoded'] = team_encoder.transform(matches_df['team1'])
matches_df['team2_encoded'] = team_encoder.transform(matches_df['team2'])
matches_df['toss_winner_encoded'] = team_encoder.transform(matches_df['toss_winner'])

# Encode 'toss_decision' and 'city'
toss_decision_encoder = LabelEncoder()
matches_df['toss_decision_encoded'] = toss_decision_encoder.fit_transform(matches_df['toss_decision'])

city_encoder = LabelEncoder()
matches_df['city_encoded'] = city_encoder.fit_transform(matches_df['city'])

# Define features (X) and target (y)
features = ['team1_encoded', 'team2_encoded', 'toss_winner_encoded', 'toss_decision_encoded', 'city_encoded']
X = matches_df[features]
y = matches_df['winner'] # The actual team name will be the target

# Encode the target variable (winner)
winner_encoder = LabelEncoder()
y_encoded = winner_encoder.fit_transform(y)


# Split data into training and testing sets
# Test size adjusted to 0.3 for a larger test set with 50 rows
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)

print(f"\nShape of X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Shape of X_test: {X_test.shape}, y_test: {y_test.shape}")
print("\nEncoded features (first 5 rows of X_train):")
print(X_train.head())
print("\nEncoded target (first 5 values of y_train):")
print(y_train[:5])


# --- 2. Train and Evaluate Supervised ML Algorithms ---

def evaluate_model(model, X_test, y_test, model_name, winner_encoder):
    """
    Trains a model and evaluates its performance using precision, recall, F1-score, and accuracy.
    Prints a classification report and confusion matrix.
    """
    y_pred = model.predict(X_test)

    print(f"\n--- {model_name} ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # For precision, recall, f1-score, we need to handle multi-class classification
    # Use 'weighted' average to account for class imbalance if any, or 'macro' for unweighted mean.
    # 'weighted' is generally preferred when classes are imbalanced.
    print(f"Precision (weighted): {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"Recall (weighted): {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"F1-Score (weighted): {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=winner_encoder.classes_, zero_division=0))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    # Optional: Display confusion matrix as a DataFrame for better readability
    # cm_df = pd.DataFrame(cm, index=winner_encoder.classes_, columns=winner_encoder.classes_)
    # print(cm_df)


# --- Logistic Regression ---
print("\n==============================================")
print("             Logistic Regression              ")
print("==============================================")
lr_model = LogisticRegression(max_iter=1000, random_state=42) # Increased max_iter for convergence
lr_model.fit(X_train, y_train)
evaluate_model(lr_model, X_test, y_test, "Logistic Regression", winner_encoder)


# --- Decision Tree Classifier ---
print("\n==============================================")
print("           Decision Tree Classifier           ")
print("==============================================")
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
evaluate_model(dt_model, X_test, y_test, "Decision Tree Classifier", winner_encoder)


# --- Random Forest Classifier ---
print("\n==============================================")
print("           Random Forest Classifier           ")
print("==============================================")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
evaluate_model(rf_model, X_test, y_test, "Random Forest Classifier", winner_encoder)