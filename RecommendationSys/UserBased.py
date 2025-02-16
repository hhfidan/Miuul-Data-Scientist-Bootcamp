import pandas as pd

# Set pandas display options for better readability
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.expand_frame_repr", False)

def create_user_movie_df():
    """
    Creates a user-movie matrix using the MovieLens dataset.
    """
    movie = pd.read_csv(r"C:\Users\hhfid\Desktop\miuul\recommender_systems\datasets\movie_lens_dataset\movie.csv")
    rating = pd.read_csv(r"C:\Users\hhfid\Desktop\miuul\recommender_systems\datasets\movie_lens_dataset\rating.csv")
    
    df = movie.merge(rating, how="left", on="movieId")
    
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["count"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values=["rating"])
    
    return user_movie_df

# Generate user-movie matrix
user_movie_df = create_user_movie_df()
user_movie_df.columns = user_movie_df.columns.droplevel(0)

# Select a random user
random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)
random_user_df = user_movie_df[user_movie_df.index == random_user]
random_user_df.columns = random_user_df.columns.droplevel(0)

# Extract watched movies
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

# Verify selection
user_movie_df.loc[user_movie_df.index == random_user, user_movie_df.columns == "Schindler's List (1993)"]

# Filter users who watched the same movies
movies_watched_df = user_movie_df[movies_watched]
user_movie_count = movies_watched_df.T.notnull().sum().reset_index()
user_movie_count.columns = ["userId", "movie_count"]

# Find users with significant overlap in watched movies
users_same_movies = user_movie_count[user_movie_count["movie_count"] > 20]["userId"]

# Create final dataframe of similar users
final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)], random_user_df[movies_watched]])
final_df.columns = final_df.columns.droplevel(0)

# Compute correlation between users
corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ["userId1", "userId2"]
corr_df = corr_df.reset_index()

# Select top correlated users
top_users = corr_df[(corr_df["userId1"] == random_user) & (corr_df["corr"] >= 0.65)][["userId2", "corr"]].reset_index(drop=True)
top_users = top_users.sort_values(by="corr", ascending=False)
top_users.rename(columns={"userId2": "userId"}, inplace=True)

# Merge with rating data
rating = pd.read_csv(r"C:\Users\hhfid\Desktop\miuul\recommender_systems\datasets\movie_lens_dataset\rating.csv")
top_users_rating = top_users.merge(rating[["userId", "movieId", "rating"]], how="inner")
top_users_rating = top_users_rating[top_users_rating["userId"] != random_user]

# Compute weighted ratings
print("Calculating weighted ratings...")
top_users_rating["weighted_rating"] = top_users_rating["corr"] * top_users_rating["rating"]

# Aggregate recommendations
recommendation_df = top_users_rating.groupby("movieId").agg({"weighted_rating": "mean"}).reset_index()
movies_recc = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)

# Merge with movie titles
movie = pd.read_csv(r"C:\Users\hhfid\Desktop\miuul\recommender_systems\datasets\movie_lens_dataset\movie.csv")
final_recommendations = movies_recc.merge(movie[["movieId", "title"]])

print("Recommended Movies:")
print(final_recommendations)
