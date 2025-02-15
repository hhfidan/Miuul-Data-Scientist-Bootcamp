import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Display settings for Pandas
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.expand_frame_repr", False)

# Load the dataset
df = pd.read_csv(r"C:\Users\hhfid\Desktop\miuul\recommender_systems\datasets\the_movies_dataset\movies_metadata.csv", low_memory=False)

# Display the first few rows of the 'overview' column
df["overview"].head()
df.shape

# Initialize TF-IDF Vectorizer
# This will convert text data into numerical format based on word importance
tfidf = TfidfVectorizer(stop_words="english")

# Fill missing values in 'overview' column with an empty string
df["overview"] = df["overview"].fillna("")

# Transform 'overview' text data into a TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(df["overview"])

# Get feature names (words in the dataset)
tfidf.get_feature_names_out()

# Convert TF-IDF matrix to an array (optional for inspection)
tfidf_matrix.toarray()

# Compute the cosine similarity matrix based on TF-IDF values
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create a Series mapping movie titles to their indices
indices = pd.Series(df.index, index=df["title"])

# Remove duplicate movie titles (keeping the last occurrence)
indeces = indices[~indices.index.duplicated(keep="last")]

# Display the count of unique movie titles
indeces.index.value_counts()

# Get the index of a specific movie
movie_index = indeces["Sherlock Holmes"]

# Retrieve similarity scores for the selected movie
cosine_sim[movie_index]

# Convert similarity scores into a DataFrame
similarity_score = pd.DataFrame(cosine_sim[movie_index], columns=["score"])

# Get the top 10 most similar movies (excluding itself)
movie_indices = similarity_score.sort_values("score", ascending=False)[1:11].index

# Display recommended movie titles
df["title"].iloc[movie_indices]


def content_based_recommender(title, cosine_sim, dataframe):
    """
    Recommends movies based on content similarity using cosine similarity scores.
    
    Parameters:
    - title (str): Title of the movie for which recommendations are needed.
    - cosine_sim (ndarray): Precomputed cosine similarity matrix.
    - dataframe (DataFrame): Movie dataset containing movie titles and overviews.
    
    Returns:
    - List of recommended movie titles.
    """
    # Create an index mapping of movie titles
    indeces = pd.Series(dataframe.index, index=dataframe["title"])
    indeces = indeces[~indeces.index.duplicated(keep="last")]
    
    # Get the index of the given movie title
    movie_index = indeces[title]
    
    # Retrieve similarity scores for the given movie
    similar_score = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
    
    # Select top 10 most similar movies (excluding the input movie)
    movie_indice = similar_score.sort_values("score", ascending=False)[1:11].index
    
    return dataframe["title"].iloc[movie_indice]

# Example usage
content_based_recommender("Batman", cosine_sim, df)


def calculate_cosine_sim(dataframe):
    """
    Computes the cosine similarity matrix for a given movie dataset.
    
    Parameters:
    - dataframe (DataFrame): Movie dataset containing the 'overview' column.
    
    Returns:
    - Cosine similarity matrix (ndarray)
    """
    tfidf = TfidfVectorizer(stop_words="english")
    dataframe["overview"] = dataframe["overview"].fillna("")
    
    # Transform text into TF-IDF matrix
    tfidf_matrix = tfidf.fit_transform(dataframe["overview"])
    
    # Compute cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return cosine_sim
