""" Helper functions for HR analytics """

# Standard library imports
import time
import itertools
import json

# Third-party library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
)
from sklearn.ensemble import RandomForestClassifier
from imblearn.metrics import geometric_mean_score
from statsmodels.stats.outliers_influence import variance_inflation_factor


def plot_data(dataframe):
    """Plot the distribution of each column in the DataFrame"""
    num_columns = len(dataframe.columns)
    num_rows = int(np.ceil(num_columns / 2))

    _, axes = plt.subplots(num_rows, 2, figsize=(15, num_rows * 5))
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    i = -1

    # Loop through the DataFrame columns and create plots
    for i, col in enumerate(dataframe.columns):
        axis = axes[i]
        # Check data type of the column
        if pd.api.types.is_numeric_dtype(dataframe[col]):
            # Additional check for binary data (0/1 or True/False)
            if sorted(dataframe[col].unique()) in [[0, 1], [False, True]]:
                sns.countplot(x=col, data=dataframe, ax=axis)
                axis.set_title(f"Distribution of {col} - Binary")
            else:
                sns.histplot(dataframe[col], kde=True, ax=axis)
                axis.set_title(f"Distribution of {col} - Numeric")
        else:
            sns.countplot(x=col, data=dataframe, ax=axis)
            axis.set_title(f"Distribution of {col} - Categorical")
        plt.setp(axis.get_xticklabels(), rotation=45, horizontalalignment="right")

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()


def display_outlier_percentage(df):
    # Select only the numeric columns for outlier calculation
    df_numeric = df.select_dtypes(include=[np.number])

    # Calculate the IQR for each numeric column
    Q1 = df_numeric.quantile(0.25)
    Q3 = df_numeric.quantile(0.75)
    IQR = Q3 - Q1

    # Calculate the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Calculate the percentage of outliers for each column
    number_of_outliers = ((df_numeric < lower_bound) | (df_numeric > upper_bound)).sum()
    outlier_percentage = (
        (df_numeric < lower_bound) | (df_numeric > upper_bound)
    ).mean() * 100

    # Create a DataFrame to display the results
    result_df = pd.DataFrame(
        {
            "Number of Outliers": number_of_outliers,
            "Percentage": outlier_percentage,
        }
    )

    return result_df


def sumstatsfmt(df):
    """Format the summary statistics DataFrame."""
    df.rename(
        index={
            "satisfaction_level": "Satisfaction Level",
            "last_evaluation": "Last Evaluation",
            "number_project": "Number of Projects",
            "average_monthly_hours": "Average Monthly Hours",
            "tenure": "Tenure (years)",
            0: "Stay",
            1: "Left",
        },
        columns={
            "count": "n",
            "mean": "Mean",
            "std": "SD",
            "min": "Min",
            "25%": "Q1",
            "50%": "Median",
            "75%": "Q3",
            "max": "Max",
        },
        inplace=True,
    )

    formats = {
        "n": "{:,.0f}",
        "Mean": "{:,.3f}",
        "SD": "{:,.3f}",
        "Min": "{:,.0f}",
        "Q1": "{:,.3f}",
        "Median": "{:,.3f}",
        "Q3": "{:,.3f}",
        "Max": "{:,.0f}",
    }
    for col, f in formats.items():
        df[col] = df[col].map(lambda x: f.format(x))
    return df


def map_labels(var, labels_dict):
    """Map the index of a DataFrame according to the provided dictionary."""
    return labels_dict.get(var, var)


def calculate_rates(df_grouped):
    """Calculate retention and attrition rates."""
    retention_rate = df_grouped[0] / (df_grouped[0] + df_grouped[1])
    attrition_rate = df_grouped[1] / (df_grouped[0] + df_grouped[1])
    return retention_rate, attrition_rate


def annotate_bars(ax):
    """Annotate bars with their height as a percentage."""
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        ax.annotate(
            f"{height:.2%}",
            (x + width / 2, y + height / 2),
            ha="center",
            va="center",
            fontsize=10,
        )


def plot_attrition_by_variable(df, var, labels_map, title):
    """Plot a stacked bar chart for retention and attrition rates."""
    # Group by the variable and 'left' column and calculate rates
    df_grouped = df.groupby([var, "left"]).size().unstack()
    df_grouped["Retention Rate"], df_grouped["Attrition Rate"] = calculate_rates(
        df_grouped
    )

    # Update index labels
    df_grouped.index = df_grouped.index.map(lambda x: map_labels(x, labels_map))

    # Plotting the stacked bar chart
    ax = df_grouped[["Retention Rate", "Attrition Rate"]].plot(
        kind="bar", stacked=True, figsize=(15, 3), title=title
    )

    # Adding labels
    ax.set_xlabel(labels_map.get(var, var))
    ax.set_ylabel("Proportion")

    # Rotate x-axis labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    # Annotate bars with percentages
    annotate_bars(ax)

    # Adjusting the legend
    ax.legend(["Stay", "Left"], loc="best", bbox_to_anchor=(1, 1.02))

    # Show plot
    plt.tight_layout()
    plt.show()


def generate_scatterplt(data, x, y, xtitle, ytitle, title, xcoord, ycoord):
    """Generate a scatter plot with a threshold line."""
    _, ax = plt.subplots(figsize=(14, 7))
    sns.scatterplot(data=data, x=x, y=y, hue="left", alpha=0.2)
    ax.axvline(x=176, color="red", linestyle="dotted", label="Threshold")
    ax.text(xcoord, ycoord, "176", color="red")
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(title="", handles=handles, labels=["Threshold", "Left", "Stay"])
    ax.legend(handles=handles, labels=["Stay", "Left", "Overtime Threshold"])

    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.title(title, fontsize="14")
    plt.show()


def prepare_data_for_plotting(df, threshold_value, feature, target):
    """
    Prepares data for plotting by adding an 'overtime' flag based on a threshold
    and splitting the DataFrame into two based on this flag.
    """
    df["overtime"] = np.where(df[feature] >= threshold_value, 1, 0)
    data_no_overtime = df[df["overtime"] == 0]
    data_overtime = df[df["overtime"] == 1]
    return data_no_overtime, data_overtime


def plot_satisfaction_histogram(data, ax, title, hue, palette, labels):
    """
    Plots a histogram of satisfaction levels with the given data.
    """
    sns.histplot(
        data=data,
        x="satisfaction_level",
        hue=hue,
        ax=ax,
        palette=palette,
        element="step",
        common_norm=False,
        alpha=0.3,
    )
    ax.set_title(title)
    ax.set_xlabel("Satisfaction Level")
    ax.legend(title="", labels=labels, loc="upper left", fontsize="small")


# Main function to create histograms based on overtime
def create_overtime_histograms(df, threshold, feature, target, colors):
    # Prepare the data for plotting
    pdf_no_overtime, pdf_overtime = prepare_data_for_plotting(
        df, threshold, feature, target
    )

    # Set up the figure for plotting
    fig, axes = plt.subplots(nrows=2, figsize=(12, 12), constrained_layout=True)

    # Plot histograms
    plot_satisfaction_histogram(
        pdf_no_overtime,
        axes[0],
        "No Working Overtime",
        target,
        colors,
        ["Left", "Stay"],
    )
    plot_satisfaction_histogram(
        pdf_overtime, axes[1], "Working Overtime", target, colors, ["Left", "Stay"]
    )

    plt.show()


def check_outcome_variable(y):
    """Check if the outcome variable is binary or categorical"""
    if y.nunique() <= 2:
        print("The outcome variable is binary or categorical.")
    else:
        print(
            "The outcome variable is not binary or categorical. Logistic regression may not be appropriate."
        )


def check_independence_of_observations(df):
    """Placeholder for checking the independence of observations"""
    # This is more of a study design question and often can't be tested directly.
    # However, for time series data, you can check for autocorrelation.
    print("Ensure that the observations are independent.")


def check_outliers(df):
    """Check for extreme outliers in the numeric feature set using Z-score."""
    # Select only numeric columns for z-score calculation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_df = df[numeric_cols]

    # Compute Z-scores for numeric columns
    z_scores = numeric_df.apply(lambda x: stats.zscore(x.dropna()))
    outliers = np.abs(z_scores) > 3

    # Return only the rows which have outliers
    return df[outliers.any(axis=1)]


def check_linearity_of_logit(df, y):
    """Check for the linearity of logit for each feature"""
    # Assess linearity with the Box-Tidwell test or visually with scatterplots
    for column in df.columns:
        sns.regplot(x=column, y=y, data=df, logistic=True).set_title(
            f"Logit linearity check for {column}"
        )
        plt.show()


def check_sample_size(y):
    """Check if the sample size is sufficiently large"""
    if len(y) >= 10 * y.nunique():  # Rule of thumb: 10 observations per category
        print("The sample size is sufficiently large.")
    else:
        print("The sample size may not be sufficiently large.")


def encode_categorical_variables(df):
    """One-hot encode categorical variables."""
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    encoder = OneHotEncoder(
        drop="first", sparse=False
    )  # Ensure the output is a dense array
    encoded_vars = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(
        encoded_vars, columns=encoder.get_feature_names_out(categorical_cols)
    )

    non_categorical_data = df.drop(columns=categorical_cols)
    reset_non_categorical = non_categorical_data.reset_index(drop=True)
    encoded_df = pd.concat([reset_non_categorical, encoded_df], axis=1)

    return encoded_df


def check_multicollinearity(df):
    """Check for multicollinearity in feature set."""
    # Ensure that all variables are numeric
    df_numeric = encode_categorical_variables(df)

    vif_data = pd.DataFrame()
    vif_data["feature"] = df_numeric.columns
    vif_data["VIF"] = [
        variance_inflation_factor(df_numeric.values, i)
        for i in range(df_numeric.shape[1])
    ]
    return vif_data


def logistic_regression_assumptions(df, target):
    """Main function to check all logistic regression assumptions"""
    y = df[target]
    X = df.drop(columns=[target])

    # Check if the outcome variable is binary/categorical
    check_outcome_variable(y)

    # Check for independence of observations
    check_independence_of_observations(df)

    # Check for multicollinearity
    multicollinearity_results = check_multicollinearity(X)
    print(multicollinearity_results)

    # Check for extreme outliers
    numeric_columns = df.select_dtypes(include=[np.number]).columns.difference(["left"])

    # Run the outlier check
    outliers = check_outliers(df[numeric_columns])
    print(f"Found {len(outliers)} extreme outliers")

    # Check for the linearity of the logit
    # check_linearity_of_logit(X, y)

    # Check for sufficiently large sample size
    check_sample_size(y)


def prepare_data(df, target, lr=True, test_size=0.20, random_state=42):
    """
    Prepares training and test datasets for modeling.
    """
    # Select features and target
    if lr == True:
        X = df[
            [
                "satisfaction_level",
                "last_evaluation",
                "number_project",
                "average_monthly_hours",
                "tenure",
                "work_accident",
                "promotion_last_5years",
                "salary",
                "overtime",
                "department",
            ]
        ]
    else:
        X = df[
            [
                "satisfaction_level",
                "tenure",
                "work_accident",
                "promotion_last_5years",
                "department",
                "salary",
            ]
        ]
    y = df[target]

    # Encode categorical variables
    X_enc = encode_categorical_variables(X)

    # Split data into training and test sets
    X_train_enc, X_test_enc, y_train, y_test = train_test_split(
        X_enc,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=y,
    )

    return X_train_enc, X_test_enc, y_train, y_test


def encode_department(X):
    """
    Encodes categorical variables within the dataset.
    """
    encoder = LabelEncoder()
    X_encoded = X.copy()
    X_encoded["department"] = encoder.fit_transform(X["department"])

    return X_encoded


def print_dataset_summary(X_train, X_test, y_train, y_test):
    """
    Prints the summary of the training and test datasets.
    """
    print("------Summary------")
    print("Training set:")
    print(f"{X_train.shape[0]} entries, {X_train.shape[1]} columns")
    print_percentage_summary(y_train)

    print("\nTest set:")
    print(f"{X_test.shape[0]} entries, {X_test.shape[1]} columns")
    print_percentage_summary(y_test)


def print_percentage_summary(y):
    """
    Prints the percentage summary of the labels in the dataset.
    """
    unique, counts = np.unique(y, return_counts=True)
    pct = dict(zip(unique, counts * 100 / len(y)))
    print(f"Stay = {pct[0]:.4f}%, Left = {pct[1]:.4f}%")


def scale_data(X_train, X_test, scaler, numeric_features):
    if scaler is not None:
        scaler = scaler()
        X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
        X_test[numeric_features] = scaler.transform(X_test[numeric_features])
    return X_train, X_test


# Function to train and evaluate the model
def train_and_evaluate(X_train, y_train, X_test, y_test, scaler_name):
    lr = LogisticRegressionCV(
        random_state=42, class_weight="balanced", n_jobs=-1, max_iter=500
    )
    start_time = time.time()
    lr.fit(X_train, y_train)
    end_time = time.time()

    y_pred = lr.predict(X_test)
    y_pred_proba = lr.predict_proba(X_test)[:, 1]

    scores = {
        "ROC AUC": roc_auc_score(y_test, y_pred_proba),
        "AP": average_precision_score(y_test, y_pred_proba),
        "Balanced accuracy": balanced_accuracy_score(y_test, y_pred),
        "G-mean": geometric_mean_score(y_test, y_pred),
        "Youden's index": balanced_accuracy_score(y_test, y_pred, adjusted=True),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "Training_time": end_time - start_time,
    }

    classification = classification_report(
        y_test, y_pred, target_names=["Stay", "Left"], digits=4, output_dict=True
    )
    confusion = confusion_matrix(y_test, y_pred)

    return scores, classification, confusion


# Main function to orchestrate the scaling, training, and evaluation
def model_assessment_pipeline(
    X_train, y_train, X_test, y_test, scalers, numeric_features
):
    results, creport, cmatrix = {}, {}, {}

    for scaler in scalers:
        scaler_name = "No Scaling" if scaler is None else scaler.__name__
        X_train_scaled, X_test_scaled = scale_data(
            X_train.copy(), X_test.copy(), scaler, numeric_features
        )
        scores, classification, confusion = train_and_evaluate(
            X_train_scaled, y_train, X_test_scaled, y_test, scaler_name
        )

        results[scaler_name] = scores
        # Ensure that classification is a dictionary suitable for DataFrame creation
        if isinstance(classification, dict):
            # Transpose the DataFrame to have the correct orientation
            creport[scaler_name] = pd.DataFrame(classification).transpose()
        else:
            raise TypeError("The classification report did not produce a dictionary.")

        cmatrix[scaler_name] = pd.DataFrame(
            confusion, index=["Stay", "Left"], columns=["Stay", "Left"]
        )

    return results, creport, cmatrix


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix", cmap=plt.cm.Blues):
    """
    Plots a confusion matrix with annotations.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title, fontsize=20)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, fontsize=14)
    plt.yticks(tick_marks, class_names, fontsize=14)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(
            j,
            i,
            f"{cm[i, j]}",
            horizontalalignment="center",
            color=color,
            fontsize=16,
        )

    plt.tight_layout()
    plt.ylabel("True label", fontsize=16)
    plt.xlabel("Predicted label", fontsize=16)
    plt.show()


def generate_classification_report(y_true, y_pred, target_names, digits=4):
    """
    Generates a classification report as a DataFrame.
    """
    report_dict = classification_report(
        y_true, y_pred, target_names=target_names, digits=digits, output_dict=True
    )
    report_df = pd.DataFrame(report_dict).transpose()
    return report_df


def calculate_performance_metrics(model, X_test, y_true, y_pred):
    """
    Calculates performance metrics for a given model and returns them as a DataFrame.
    """
    metrics = {
        "ROC AUC": roc_auc_score(y_true, model.predict_proba(X_test)[:, 1]),
        "AP": average_precision_score(y_true, model.predict_proba(X_test)[:, 1]),
        "Balanced accuracy": balanced_accuracy_score(y_true, y_pred),
        "G-mean": geometric_mean_score(y_true, y_pred),
        "Youden's index": balanced_accuracy_score(y_true, y_pred, adjusted=True),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

    metrics_df = pd.DataFrame(metrics, index=[0])
    return metrics_df


def calculate_metrics(y_true, y_pred_proba, y_pred, average="macro"):
    metrics = {
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
        "ap": average_precision_score(y_true, y_pred_proba),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "gmean": geometric_mean_score(y_true, y_pred, average=average),
        "youden": balanced_accuracy_score(y_true, y_pred, adjusted=True),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    return metrics


def prepare_classification_report(y_true, y_pred, target_names):
    """
    Prepare a classification report based on true labels and predictions.
    """
    report = classification_report(
        y_true, y_pred, target_names=target_names, digits=4, output_dict=True
    )
    return pd.DataFrame.from_dict(report).transpose()


def dump_params(params, file_path):
    with open(file_path, "w") as f:
        json.dump(params, f, indent=4)


def load_params(file_path):
    """Load parameters from a JSON file."""
    try:
        with open(file_path, "r") as file:
            params = json.load(file)
            if params:  # Check if the parameters are not empty
                return params
            else:
                return None
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def create_model_with_params(params):
    """Create a RandomForestClassifier with given parameters."""
    return RandomForestClassifier(**params)
