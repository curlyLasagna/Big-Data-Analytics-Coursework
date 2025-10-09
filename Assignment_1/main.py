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
        alt,
        chi2,
        mo,
        pd,
        pl,
        train_test_split,
    )


@app.cell
def _(mo):
    mo.md(r"""## Reading in the dataset""")
    return


@app.cell
def _(pd):
    df = pd.read_csv("./shopping_trends_updated.csv")
    df_categorical = list(df.select_dtypes(exclude='number').drop(columns="Subscription Status").columns)
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
def _(alt, df):
    (
        alt.Chart(df.select("Age"))
        .mark_boxplot()
        .encode(
            alt.Y("Age:Q").scale(zero=False),
        )
    )
    return


@app.cell
def _(df, pl):
    df.select(pl.col("Subscription Status").value_counts())
    return


@app.cell
def _(df, pl):
    df.select(pl.col("Location").value_counts()).unnest("Location")
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
def _(StandardScaler, X_train):
    transformer_std = (
        "std",
        StandardScaler(),
        list(X_train.select_dtypes(include="number").columns),
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
def _(FunctionTransformer, pd):
    def bar(x: pd.DataFrame):
        return x.map({"Yes": 1, "No": 0})


    bi_map_func = FunctionTransformer(bar)
    return (bi_map_func,)


@app.cell
def _(X_train, bi_map_func):
    bi_map_func.fit_transform(X_train["Subscription Status"])
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


app._unparsable_cell(
    r"""
    transformer_one_hot = 
    """,
    name="_"
)


@app.cell
def _(mo):
    mo.md(r"""### Target Encoding""")
    return


app._unparsable_cell(
    r"""
    transformer_target = 
    """,
    name="_"
)


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

    label_size_func = FunctionTransformer(lambda x: x.replace(label_size_mapping))

    label_freq_func = FunctionTransformer(lambda x: x.replace(label_freq_mapping))
    return label_freq_func, label_size_func


@app.cell
def _(X_train):
    X_train
    return


@app.cell
def _(
    ColumnTransformer,
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
    TargetEncoder,
    X_train,
    label_freq_func,
    label_size_func,
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
                    "Gender"
                ],
            ),
            # Target encoding
            ("target", TargetEncoder(), ["Location"]),
            # Label encoding
            ("label_size", label_size_func, ["Size"]),
            ("label_freq", label_freq_func, ["Frequency of Purchases"]),
        ],
        remainder="passthrough",
    )

    pre_processed_df = pre_processing.fit_transform(X=X_train, y=y_train.replace({"Yes": 1, "No": 0}))
    return pre_processed_df, pre_processing


@app.cell
def _(pre_processed_df):
    pre_processed_df
    return


@app.cell
def _(mo):
    mo.md(r"""### Feature Selection""")
    return


@app.cell
def _(SelectKBest, chi2, pre_processed_df, y_train):
    SelectKBest(score_func=chi2, k=10).fit_transform(pre_processed_df.select_dtypes(exclude=[float]), y_train)
    return


@app.cell
def _(mo):
    mo.md(r"""## Training""")
    return


@app.cell
def _():
    return


@app.cell
def _(pre_processed_df):
    pre_processed_df.columns[pre_processed_df.isin(["No"]).any()]
    return


@app.cell
def _(
    RandomForestClassifier,
    X_test,
    pre_processed_df,
    pre_processing,
    y_test,
    y_train,
):
    RandomForestClassifier().fit(pre_processed_df, y_train).predict(pre_processing.fit_transform(X_test, y_test.replace({"Yes": 1, "No": 0})))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
