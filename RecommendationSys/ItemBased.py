import pandas as pd

# Set pandas options for better display
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.expand_frame_repr", False)

# Load datasets
movie = pd.read_csv(r"C:\Users\hhfid\Desktop\miuul\recommender_systems\datasets\movie_lens_dataset\movie.csv")
rating = pd.read_csv(r"C:\Users\hhfid\Desktop\miuul\recommender_systems\datasets\movie_lens_dataset\rating.csv")

# Merge datasets on 'movieId'
df = movie.merge(rating, how="left", on="movieId")

# Display the first few rows
df.head()

# Count the number of ratings per movie
comment_counts = pd.DataFrame(df["title"].value_counts())

# Identify rare movies (movies with 1000 or fewer ratings)
rare_movies = comment_counts[comment_counts["count"] <= 1000].index

# Filter out rare movies to keep only commonly rated ones
common_movies = df[~df["title"].isin(rare_movies)]

# Create a user-movie rating matrix
user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values=["rating"])
user_movie_df.columns = user_movie_df.columns.droplevel(0)  # Remove multi-index column

# Example: Select a movie and calculate similarity
name = "Lord of the Rings: The Fellowship of the Ring, The (2001)"
movie_name = user_movie_df[name]

# Compute correlation with other movies
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(11).iloc[1:]

#### Random Movie Selection Example ####
name = pd.Series(user_movie_df.columns).sample(1).values[0]

# Function to search for movies by keyword
def check_movie(keyword, dataframe):
    return [col for col in dataframe.columns if keyword in col]

# Example: Search for movies containing "Lord"
check_movie("Lord", user_movie_df)

###################### Function Definitions ##############################

def create_user_movie_df():
    """
    Creates a user-movie rating matrix with commonly rated movies.
    """
    import pandas as pd
    movie = pd.read_csv(r"C:\Users\hhfid\Desktop\miuul\recommender_systems\datasets\movie_lens_dataset\movie.csv")
    rating = pd.read_csv(r"C:\Users\hhfid\Desktop\miuul\recommender_systems\datasets\movie_lens_dataset\rating.csv")
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["count"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values=["rating"])
    return user_movie_df

def item_based_recommender(movie_name, user_movie_df):
    """
    Provides item-based movie recommendations based on correlation.
    :param movie_name: Name of the movie to find recommendations for.
    :param user_movie_df: User-movie rating matrix.
    :return: Top 10 recommended movies.
    """
    movie_name = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)
