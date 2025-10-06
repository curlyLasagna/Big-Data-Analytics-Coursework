import marimo

__generated_with = "0.16.5"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import polars as pl
    import altair as alt
    return alt, pl


@app.cell
def _(pl):
    df = pl.read_csv('./shopping_trends_updated.csv')
    df.null_count()
    return (df,)


@app.cell
def _(df):
    df
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
def _():
    return


if __name__ == "__main__":
    app.run()
