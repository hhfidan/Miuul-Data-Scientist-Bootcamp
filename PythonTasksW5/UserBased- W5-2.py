import pandas as pd
from unittest.mock import inplace
from sphinx.builders.gettext import timestamp

# Display settings for Pandas
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.expand_frame_repr", False)

# User-Based Recommendation System
# Task 1: Data Preparation

# Step 1: Read movie and rating datasets
movie = pd.read_csv(r"C:\Users\hhfid\Desktop\miuul\week5\2\datasets\movie.csv")
rating = pd.read_csv(r"C:\Users\hhfid\Desktop\miuul\week5\2\datasets\rating.csv")

# Step 2: Merge movie names and genres into the rating dataset
rating = rating.merge(movie, how="left", on="movieId")

# Step 3: Filter out movies with less than 1000 total ratings
underk = [title for title in rating.groupby("title").agg({"userId": "count"}).query("userId < 1000").index]
common_movies = rating[~rating["title"].isin(underk)]

# Step 4: Create a pivot table with user IDs as index, movie titles as columns, and ratings as values
user_movie_df = common_movies.pivot_table(index="userId", columns="title", values="rating")

# Step 5: Convert all the above steps into a function
def create_user_movie_df():
    movie = pd.read_csv(r"C:\Users\hhfid\Desktop\miuul\week5\2\datasets\movie.csv")
    rating = pd.read_csv(r"C:\Users\hhfid\Desktop\miuul\week5\2\datasets\rating.csv")
    rating = rating.merge(movie, how="left", on="movieId")
    underk = [title for title in rating.groupby("title").agg({"userId": "count"}).query("userId < 1000").index]
    common_movies = rating[~rating["title"].isin(underk)]
    return common_movies.pivot_table(index="userId", columns="title", values="rating")

# Task 2: Identify Movies Watched by the Selected User
randomUser = user_movie_df.index.to_series().sample(1).iloc[0]
random_user_df = user_movie_df[user_movie_df.index == randomUser]
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

# Task 3: Access Data of Other Users Who Watched the Same Movies
movies_watched_df = user_movie_df[movies_watched]
user_movie_count = movies_watched_df.notnull().sum(axis=1).reset_index()
user_movie_count.columns = ["userId", "movie_count"]
users_same_movies = user_movie_count[user_movie_count["movie_count"] > len(movies_watched) * 60 / 100]["userId"]

# Task 4: Identify the Most Similar Users
final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)], random_user_df[movies_watched]])
final_df.reset_index()

corr_df = final_df.T.corr().stack().sort_values().drop_duplicates()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ["userId1", "userId2"]
corr_df = corr_df.reset_index()

top_users = corr_df[(corr_df["userId1"] == randomUser) & (corr_df["corr"] > 0.65)][["userId2", "corr"]].reset_index(drop=True)
top_users = top_users.sort_values(by="corr", ascending=False)
top_users.rename(columns={"userId2": "userId"}, inplace=True)

top_users_rating = top_users.merge(rating[["userId", "movieId", "rating"]], how="inner")

# Task 5: Calculate Weighted Average Recommendation Score and Select Top 5 Movies
top_users_rating["weighted_rating"] = top_users_rating["corr"] * top_users_rating["rating"]
recomm_df = top_users_rating.groupby("movieId").agg({"weighted_rating": "mean"})
recomm_df = recomm_df[recomm_df["weighted_rating"] > 3.5].sort_values(by="weighted_rating", ascending=False).reset_index()
movie_recc = recomm_df.merge(movie[["movieId", "title"]]).iloc[:5]

# Item-Based Recommendation System
# Task 1: Generate Item-Based Recommendations for the Most Recently Watched and Highly Rated Movie
df = rating.merge(movie, how="left", on="movieId")
df = df[~df["title"].isin(underk)]
df["timestamp"] = pd.to_datetime(df["timestamp"])

new = df[(df["userId"] == randomUser) & (df["rating"] == 5)].sort_values(by="timestamp", ascending=False).iloc[:1]["movieId"]
movie_Id = new.iloc[0]

name_finder = df[df["movieId"] == movie_Id]["title"].drop_duplicates().head(1).tolist()
name = name_finder[0]

user_movie_df_filtered = user_movie_df[name]
corr_with_selected_movie = user_movie_df.corrwith(user_movie_df_filtered).sort_values(ascending=False).head(10)
corr_with_selected_movie.iloc[1:6]
