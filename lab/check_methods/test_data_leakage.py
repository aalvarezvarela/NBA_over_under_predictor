import pandas as pd
from nba_ou.create_training_data.create_df_to_predict import create_df_to_predict
from nba_ou.prediction.clean_data_before_prediction import prepare_training_dataframe
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)


"2025-12-19"

# Create training data up to a specific date
date_to_train_1 = "2025-12-19"

safe_limit = "2025-10-11"  # Optional: specify start date


#set environment variable 'test' to enable test data leakage condition
import os
os.environ['test'] = 'Yes'
df_total_zero = create_df_to_predict(
    recent_limit_to_include=date_to_train_1, older_limit_to_include=safe_limit
)

os.environ['test'] = 'No'

df_total_normal = create_df_to_predict(
    recent_limit_to_include=date_to_train_1, older_limit_to_include=safe_limit
)

print(f"Total games created (with zeros): {len(df_total_zero)}")

df_total_normal[df_total_normal['GAME_DATE'] == "2025-12-19"]
df_total_zero[df_total_zero['GAME_DATE'] == "2025-12-19"]


df_total_zero.drop(columns=["TOTAL_POINTS"]).equals(df_total_normal.drop(columns=["TOTAL_POINTS"]))


import pandas as pd
#for testing drop the TOTAL_POINTS
pd.testing.assert_frame_equal(df_total_zero.drop(columns=["TOTAL_POINTS"]), df_total_normal.drop(columns=["TOTAL_POINTS"])  )

print()


