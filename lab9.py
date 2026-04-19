

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               StackingRegressor)
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
import lime
import lime.lime_tabular
import warnings
warnings.filterwarnings("ignore")


# ==============================================================================
# STEP 1 - Load and Prepare Data
# ==============================================================================

def load_and_prepare_data(filepath):
    """
    Load the dataset and encode categorical columns into numbers.
    Returns X (features), y (target), and the feature names.
    """
    df = pd.read_excel(filepath)

    # Encode categorical columns using LabelEncoder
    le_state  = LabelEncoder()
    le_season = LabelEncoder()
    le_food   = LabelEncoder()

    df['state_enc']  = le_state.fit_transform(df['state'])
    df['season_enc'] = le_season.fit_transform(df['season'])
    df['food_enc']   = le_food.fit_transform(df['food_item'])

    # Extract the starting year as a number (e.g., "2016-17" -> 2016)
    df['year_num'] = df['year'].str[:4].astype(int)

    feature_cols = ['state_enc', 'year_num', 'season_enc', 'food_enc']
    X = df[feature_cols].values
    y = df['production'].values

    return X, y, feature_cols


# ==============================================================================
# A1 - Stacking Regressor
# ==============================================================================

def build_stacking_regressor(meta_model_name='ridge'):
    """
    Build a Stacking Regressor with:
      - Base models: Random Forest, Gradient Boosting, Decision Tree
      - Meta-model (final_estimator): Ridge or Linear Regression (configurable)
    
    meta_model_name options: 'ridge', 'linear'
    """
    # Base models (level-0 learners)
    base_models = [
        ('random_forest',       RandomForestRegressor(n_estimators=50, random_state=42)),
        ('gradient_boosting',   GradientBoostingRegressor(n_estimators=50, random_state=42)),
        ('decision_tree',       DecisionTreeRegressor(max_depth=5, random_state=42)),
    ]

    # Choose meta-model (level-1 learner)
    if meta_model_name == 'ridge':
        meta_model = Ridge()
    else:
        meta_model = LinearRegression()

    stacking_model = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5  # 5-fold cross-validation used to train the meta model
    )

    return stacking_model


def train_and_evaluate_stacking(X_train, X_test, y_train, y_test, meta_model_name='ridge'):
    """
    Train the stacking regressor and return evaluation metrics.
    """
    model = build_stacking_regressor(meta_model_name)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    return model, y_pred, mae, rmse, r2


# ==============================================================================
# A2 - Pipeline (Preprocessing + Model in one flow)
# ==============================================================================

def build_pipeline(model):
    """
    Build a sklearn Pipeline that chains:
      1. StandardScaler  - scales features to mean=0, std=1
      2. Any regressor   - passed as argument
    
    This ensures scaling and prediction happen in one step automatically.
    """
    pipe = Pipeline(steps=[
        ('scaler', StandardScaler()),  # Step 1: Normalize features
        ('model',  model)              # Step 2: Train/predict
    ])
    return pipe


def train_pipeline(X_train, X_test, y_train, y_test):
    """
    Build and train the pipeline using Stacking Regressor as the model.
    Returns the fitted pipeline and evaluation metrics.
    """
    # Use Stacking Regressor inside the pipeline
    stacking_model = build_stacking_regressor(meta_model_name='ridge')
    pipe = build_pipeline(stacking_model)

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    return pipe, y_pred, mae, rmse, r2


# ==============================================================================
# A3 - LIME Explainer
# ==============================================================================

def explain_with_lime(pipeline, X_train, X_test, feature_names, num_samples=3):
    """
    Use LIME (Local Interpretable Model-agnostic Explanations) to explain
    individual predictions made by the pipeline.
    
    LIME works by:
      1. Taking one test sample
      2. Creating slightly modified versions of it
      3. Seeing how the model's output changes
      4. Fitting a simple linear model on those changes to explain importance
    
    Returns a list of (sample_index, explanation_dict) tuples.
    """
    # LIME needs a plain predict function (not pipeline's transform+predict)
    def predict_fn(data):
        return pipeline.predict(data)

    # Create the LIME explainer for tabular data
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        mode='regression'
    )

    explanations = []
    for i in range(num_samples):
        sample = X_test[i]
        exp = explainer.explain_instance(
            data_row=sample,
            predict_fn=predict_fn,
            num_features=len(feature_names)
        )
        # Get feature importance as a dict
        exp_dict = dict(exp.as_list())
        explanations.append((i, exp_dict))

    return explanations


# ==============================================================================
# MAIN PROGRAM - Run everything and print results
# ==============================================================================

if __name__ == "__main__":

    DATASET_PATH = "FINAL_MERGED_DATASET.xlsx"
    FEATURE_NAMES = ['state_enc', 'year_num', 'season_enc', 'food_enc']

    print("=" * 65)
    print("  Lab 09 | Stacking, Pipeline & LIME | 22AIE213")
    print("=" * 65)

    # ---------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------
    print("\n[STEP 1] Loading and preparing dataset...")
    X, y, feature_names = load_and_prepare_data(DATASET_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Total samples  : {len(X)}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples : {len(X_test)}")

    # ---------------------------------------------------------------
    # A1 - Stacking Regressor with two different meta-models
    # ---------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  A1 - Stacking Regressor Results")
    print("=" * 65)

    results = {}

    for meta in ['ridge', 'linear']:
        print(f"\n  Meta-model: {meta.upper()}")
        model, y_pred, mae, rmse, r2 = train_and_evaluate_stacking(
            X_train, X_test, y_train, y_test, meta_model_name=meta
        )
        results[f'Stacking ({meta})'] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
        print(f"    MAE  : {mae:.4f}")
        print(f"    RMSE : {rmse:.4f}")
        print(f"    R²   : {r2:.4f}")

    # Also show individual base model results for comparison
    print("\n  --- Individual Base Model Results (for comparison) ---")
    base_models_list = [
        ('Random Forest',      RandomForestRegressor(n_estimators=50, random_state=42)),
        ('Gradient Boosting',  GradientBoostingRegressor(n_estimators=50, random_state=42)),
        ('Decision Tree',      DecisionTreeRegressor(max_depth=5, random_state=42)),
    ]
    for name, mdl in base_models_list:
        mdl.fit(X_train, y_train)
        preds = mdl.predict(X_test)
        mae_b  = mean_absolute_error(y_test, preds)
        rmse_b = np.sqrt(mean_squared_error(y_test, preds))
        r2_b   = r2_score(y_test, preds)
        results[name] = {'MAE': mae_b, 'RMSE': rmse_b, 'R2': r2_b}
        print(f"\n  {name}:")
        print(f"    MAE  : {mae_b:.4f}")
        print(f"    RMSE : {rmse_b:.4f}")
        print(f"    R²   : {r2_b:.4f}")

    # Summary table
    print("\n  --- Summary Table (A1) ---")
    print(f"  {'Model':<30} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
    print("  " + "-" * 58)
    for model_name, metrics in results.items():
        print(f"  {model_name:<30} {metrics['MAE']:>8.4f} {metrics['RMSE']:>8.4f} {metrics['R2']:>8.4f}")

    # ---------------------------------------------------------------
    # A2 - Pipeline
    # ---------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  A2 - Pipeline (Scaler + Stacking Regressor)")
    print("=" * 65)

    pipe, pipe_preds, pipe_mae, pipe_rmse, pipe_r2 = train_pipeline(
        X_train, X_test, y_train, y_test
    )

    print(f"\n  Pipeline Steps: {[step[0] for step in pipe.steps]}")
    print(f"\n  Pipeline Evaluation:")
    print(f"    MAE  : {pipe_mae:.4f}")
    print(f"    RMSE : {pipe_rmse:.4f}")
    print(f"    R²   : {pipe_r2:.4f}")

    # Show a few sample predictions from pipeline
    print("\n  Sample Predictions (first 5 test samples):")
    print(f"  {'Index':<8} {'Actual':>10} {'Predicted':>12}")
    print("  " + "-" * 32)
    for i in range(5):
        print(f"  {i:<8} {y_test[i]:>10.2f} {pipe_preds[i]:>12.2f}")

    # ---------------------------------------------------------------
    # A3 - LIME Explainer
    # ---------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  A3 - LIME Explanation for Pipeline Predictions")
    print("=" * 65)
    print("\n  Explaining 3 individual predictions using LIME...")
    print("  (LIME tells us which feature pushed the prediction up or down)\n")

    explanations = explain_with_lime(pipe, X_train, X_test, feature_names, num_samples=3)

    for idx, exp_dict in explanations:
        actual    = y_test[idx]
        predicted = pipe.predict(X_test[idx].reshape(1, -1))[0]

        print(f"  --- Sample #{idx} ---")
        print(f"    Actual Production    : {actual:.2f} lakh tonnes")
        print(f"    Predicted Production : {predicted:.2f} lakh tonnes")
        print(f"    LIME Feature Importances:")
        for feat, importance in exp_dict.items():
            direction = "↑ increases" if importance > 0 else "↓ decreases"
            print(f"      {feat:<30} : {importance:+.4f}  ({direction} prediction)")
        print()

   