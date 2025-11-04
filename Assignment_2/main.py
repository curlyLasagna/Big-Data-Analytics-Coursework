import marimo

__generated_with = "0.16.5"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    from kagglehub import dataset_download
    import pandas as pd
    from pathlib import Path
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    import re
    from sklearn.metrics.pairwise import cosine_similarity
    from scipy.sparse import csr_matrix
    from sklearn.neighbors import NearestNeighbors
    from lightfm.evaluation import precision_at_k
    from lightfm import LightFM
    from sklearn.model_selection import train_test_split
    from lightfm.cross_validation import random_train_test_split
    from sklearn.metrics import average_precision_score
    return (
        LightFM,
        NearestNeighbors,
        Path,
        average_precision_score,
        csr_matrix,
        dataset_download,
        np,
        pd,
        precision_at_k,
        re,
        train_test_split,
    )


@app.cell
def _(Path, dataset_download, pd):
    dataset = dataset_download("CooperUnion/anime-recommendations-database")

    ratings_df = pd.read_csv(Path(dataset) / "rating.csv").sample(n=20000, random_state=42)

    anime_df = pd.read_csv(Path(dataset) / "anime.csv")
    return anime_df, ratings_df


@app.cell
def _(ratings_df):
    ratings_df
    return


@app.cell
def _(ratings_df):
    (ratings_df[ratings_df["rating"] != -1])
    return


@app.cell
def _(anime_df):
    anime_df
    return


@app.cell
def _(anime_df, np, pd, ratings_df):
    collab_df = pd.merge(
        anime_df, ratings_df, on="anime_id", suffixes=[None, "_user"]
    )

    collab_df.replace(to_replace=-1, value=np.nan, inplace=True)
    collab_df = collab_df.dropna()
    return (collab_df,)


@app.cell
def _(collab_df, re):
    def text_cleaning(text):
        text = re.sub(r"&quot;", "", text)
        text = re.sub(r".hack//", "", text)
        text = re.sub(r"&#039;", "", text)
        text = re.sub(r"A&#039;s", "", text)
        text = re.sub(r"I&#039;", "I'", text)
        text = re.sub(r"&amp;", "and", text)

        return text


    collab_df["name"] = collab_df["name"].apply(text_cleaning)
    return


@app.cell
def _(collab_df, train_test_split):
    X_train, X_test, y_train, y_test = train_test_split(
        collab_df.drop("rating_user", axis=1),
        collab_df["rating_user"],
        test_size=0.20,
        random_state=42,
    )
    return X_test, X_train


@app.cell
def _(X_train):
    X_train
    return


@app.cell
def _(csr_matrix):
    def create_user_item_matrix(df, row='anime_id', col='user_id', value='rating'):
        """
        Given a DataFrame, returns the user-item pivot table and its CSR matrix.
    
        Parameters:
            df: pandas DataFrame with columns for row, col, and value
            row: column name to use as rows (default: 'anime_id')
            col: column name to use as columns (default: 'user_id')
            value: column name to use as values (default: 'rating')
    
        Returns:
            pivot_table: pandas DataFrame (user-item matrix)
            piv_sparse: scipy.sparse.csr_matrix
        """
        pivot_table = df.pivot_table(index=row, columns=col, values=value)
        pivot_table = pivot_table.fillna(0)
        piv_sparse = csr_matrix(pivot_table.values)
        return pivot_table, piv_sparse
    return (create_user_item_matrix,)


@app.cell
def _(X_test, X_train, create_user_item_matrix):
    # For train data
    train_pivot, train_sparse = create_user_item_matrix(X_train)

    # For test data
    test_pivot, test_sparse = create_user_item_matrix(X_test)
    return test_sparse, train_pivot, train_sparse


@app.cell
def _(NearestNeighbors, train_sparse):
    model_knn = NearestNeighbors(metric="cosine", algorithm="brute")
    model_knn.fit(train_sparse)
    return (model_knn,)


@app.cell
def _(X_test, average_precision_score, model_knn, np, train_pivot):
    def mean_average_precision(model_knn, pivot_table, test_df, k=10):
        user_ids = test_df['user_id'].unique()
        ap_scores = []
    
        for user_id in user_ids:
            # Items this user has rated in test set
            true_items = test_df[test_df['user_id'] == user_id]['anime_id'].values
        
            # If user not in training, skip
            if user_id not in pivot_table.columns:
                continue
        
            # Get all items for this user in the pivot table
            user_ratings = pivot_table[user_id]
            # Recommend top k items the user hasn't rated yet
            already_rated = user_ratings[user_ratings > 0].index
            unrated_items = pivot_table.index.difference(already_rated)
        
            # For each unrated item, get predicted score (e.g., via KNN similarity)
            # Here, we use the mean similarity of the k nearest neighbors
            scores = []
            for item in unrated_items:
                item_vector = pivot_table.loc[item].values.reshape(1, -1)
                distances, indices = model_knn.kneighbors(item_vector, n_neighbors=k)
                # Score: mean rating of neighbors for this user
                neighbor_items = pivot_table.index[indices.flatten()]
                neighbor_ratings = pivot_table.loc[neighbor_items, user_id].values
                scores.append(np.nanmean(neighbor_ratings))
        
            # Get top-k recommendations
            top_k_idx = np.argsort(scores)[-k:][::-1]
            recommended_items = unrated_items[top_k_idx]
        
            # Create binary relevance vector for MAP calculation
            y_true = np.isin(recommended_items, true_items).astype(int)
            y_score = np.arange(k, 0, -1)  # Higher rank = higher score
        
            if y_true.sum() > 0:
                ap = average_precision_score(y_true, y_score)
                ap_scores.append(ap)
    
        map_score = np.mean(ap_scores) if ap_scores else 0.0
        return map_score

    map_score = mean_average_precision(model_knn, train_pivot, X_test, k=10)
    map_score
    return


@app.cell
def _(LightFM):
    lightfm_model = LightFM(loss="warp")

    return (lightfm_model,)


@app.cell
def _(lightfm_model, precision_at_k, train_sparse):
    precision_at_k(lightfm_model.fit(train_sparse, epochs=10), train_sparse, k=10).mean()
    return


@app.cell
def _(test_sparse):
    test_sparse
    return


@app.cell
def _(train_sparse):
    train_sparse
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
