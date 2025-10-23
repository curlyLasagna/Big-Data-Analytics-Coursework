import marimo

__generated_with = "0.16.5"
app = marimo.App(width="columns")


@app.cell
def _():
    import marimo as mo
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import (
        FunctionTransformer,
        StandardScaler,
        LabelEncoder,
    )
    from sklearn.feature_selection import chi2, SelectKBest
    from sklearn.metrics import roc_curve, auc, accuracy_score, roc_auc_score
    import polars as pl
    import altair as alt
    from category_encoders import BinaryEncoder, TargetEncoder, OneHotEncoder
    import pyarrow
    import pandas as pd
    from sklearn import set_config
    import numpy as np

    set_config(transform_output="pandas")
    return (
        ColumnTransformer,
        FunctionTransformer,
        OneHotEncoder,
        RandomForestClassifier,
        SelectKBest,
        StandardScaler,
        TargetEncoder,
        accuracy_score,
        alt,
        chi2,
        mo,
        pd,
        roc_curve,
        train_test_split,
    )


@app.cell
def _(mo):
    mo.md(r"""## Reading in the dataset""")
    return


@app.cell
def _(pd):
    df = pd.read_csv("./shopping_trends_updated.csv")
    df_categorical = list(
        df.select_dtypes(exclude="number")
        .drop(columns="Subscription Status")
        .columns
    )

    df_categorical = list(
        df.select_dtypes(exclude="number")
        .drop(columns="Subscription Status")
        .columns
    )

    df.groupby("Subscription Status").count()
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""## Exploratory Data Analysis""")
    return


@app.cell
def _(df):
    df.groupby("Location").value_counts()
    return


@app.cell
def _(mo):
    mo.md(r"""## Data Preprocessing""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Test-train split

    We want to split our dataset to test and train before encoding to avoid any data leakage.

    The split will be 70/30 and the dataset stratified by location.
    """
    )
    return


@app.cell
def _(df, train_test_split):
    X_train, X_test, y_train, y_test = train_test_split(
        # Drop customer ID and target variable
        df.drop(columns=["Customer ID"]),
        # Target variable
        df["Subscription Status"],
        # 70/30 test size
        test_size=0.3,
        # Equal representation of locations
        stratify=df["Location"],
    )
    return X_test, X_train, y_test, y_train


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Standard Scale

    Set a standard scale to numerical features
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""### Feature Encoding""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    The following features will be mapped to 0 or 1 as they are binary values

    - Subscription Status
    - Discount Applied
    - Promo Code Used
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### One-Hot Encoding

    For the following features:
    - Item Purchased
    - Category
    - Color
    - Season
    - Shipping Type
    - Payment Method
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""### Target Encoding""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Label Encoding

    Columns:
    - Size
    - Frequency of Purchases
    """
    )
    return


@app.cell
def _(FunctionTransformer):
    label_size_mapping = {"S": 0, "M": 1, "L": 2, "XL": 3}
    label_freq_mapping = {
        "Every 3 Months": 0,
        "Monthly": 1,
        "Bi-Weekly": 2,
        "Quarterly": 3,
        "Annually": 4,
        "Weekly": 5,
        "Fortnightly": 6,
    }

    label_size_func = FunctionTransformer(
        lambda x: x.replace(label_size_mapping).infer_objects(copy=False)
    )

    label_freq_func = FunctionTransformer(
        lambda x: x.replace(label_freq_mapping).infer_objects(copy=False)
    )
    return label_freq_func, label_size_func


@app.cell
def _(
    ColumnTransformer,
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
    TargetEncoder,
    X_test,
    X_train,
    label_freq_func,
    label_size_func,
    y_test,
    y_train,
):
    pre_processing = ColumnTransformer(
        transformers=[
            # Mapping binary values
            (
                "bi_label",
                FunctionTransformer(
                    lambda x: x.replace({"Yes": 1, "No": 0}), validate=False
                ),
                ["Discount Applied", "Promo Code Used", "Subscription Status"],
            ),
            (
                "std",
                StandardScaler(),
                [
                    "Age",
                    "Purchase Amount (USD)",
                    "Review Rating",
                    "Previous Purchases",
                ],
            ),
            (
                "one_hot",
                OneHotEncoder(),
                [
                    "Item Purchased",
                    "Category",
                    "Color",
                    "Season",
                    "Shipping Type",
                    "Payment Method",
                    "Gender",
                ],
            ),
            # Target encoding
            ("target", TargetEncoder(), ["Location"]),
            # Label encoding
            ("label_size", label_size_func, ["Size"]),
            ("label_freq", label_freq_func, ["Frequency of Purchases"]),
        ],
        # Keep other columns as is, but this shouldn't be the case because otherwise, we'll get errors
        remainder="passthrough",
    )

    # Pre process both test and training data
    pre_processed_df_train = pre_processing.fit_transform(
        X=X_train, y=y_train.replace({"Yes": 1, "No": 0})
    )

    pre_processed_df_test = pre_processing.fit_transform(
        X=X_test, y=y_test.replace({"Yes": 1, "No": 0})
    )
    return pre_processed_df_test, pre_processed_df_train


@app.cell
def _(mo):
    mo.md(r"""### Feature Selection""")
    return


@app.cell
def _(SelectKBest, chi2, pd, pre_processed_df_train, y_train):
    feature_scores = (
        SelectKBest(score_func=chi2, k=5).fit(
            pre_processed_df_train.select_dtypes(exclude=[float]), y_train
        )
    ).scores_

    feature_scores_df = pd.Series(
        feature_scores,
        index=pre_processed_df_train.select_dtypes(exclude=[float]).columns,
    ).sort_values(ascending=False)
    return (feature_scores_df,)


@app.cell
def _(
    RandomForestClassifier,
    accuracy_score,
    feature_scores_df,
    pre_processed_df_test,
    pre_processed_df_train,
    y_test,
    y_train,
):
    y_pred = (
        RandomForestClassifier(n_estimators=1)
        .fit(
            pre_processed_df_train[feature_scores_df.nlargest(2).index.tolist()],
            y_train,
        )
        .predict(
            pre_processed_df_test[feature_scores_df.nlargest(2).index.tolist()]
        )
    )



    accuracy_score(y_test, y_pred)
    return (y_pred,)


@app.cell
def _(RandomForestClassifier, pre_processed_df_train, y_train):
    feature_names = pre_processed_df_train.columns

    sorted(list(zip(feature_names, list(RandomForestClassifier().fit(pre_processed_df_train, y_train).feature_importances_))), key=lambda x: x[1], reverse=True)
    return


@app.cell
def _(pd, roc_curve, y_pred, y_test):
    fpr, tpr, thresh = roc_curve(y_test.replace({"Yes": 1, "No": 0}), pd.Series(y_pred).replace({"Yes": 1, "No": 0}), pos_label=1)
    return fpr, tpr


@app.cell
def _(alt, fpr, pd, tpr):
    roc_data= pd.DataFrame({'False Positive Rate': fpr,
        'True Positive Rate': tpr})

    roc_chart = alt.Chart(roc_data).mark_line().encode(
        x=alt.X('False Positive Rate', title='False Positive Rate'),
        y=alt.Y('True Positive Rate', title='True Positive Rate'),
        tooltip=['False Positive Rate', 'True Positive Rate']
    ).properties(
        title='ROC Curve'
    )

    diagonal_line = alt.Chart(pd.DataFrame({'x': [0, 1], 'y': [0, 1]})).mark_line(
        color='gray',
        strokeDash=[5, 5]
    ).encode(x='x', y='y')

    (roc_chart + diagonal_line).interactive()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
