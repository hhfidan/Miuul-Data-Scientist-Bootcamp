import pandas as pd
import math
import scipy.stats as st

# Configure pandas display settings for better readability
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

###############################
# Score Calculation Functions
###############################

def differ_score(up, down):
    """
    Calculates the difference between positive (up) and negative (down) votes.
    """
    return up - down

def average_rating(up, down):
    """
    Calculates the average rating as up / (up + down).
    Returns 0 if there are no votes.
    """
    if up + down == 0:
        return 0
    return up / (up + down)

##########################
# Wilson Lower Bound Score
##########################

def wilson_lower_bound(up, down, confidence=0.95):
    """
    Calculates the Wilson Lower Bound Score.
    This method provides a more reliable ranking by accounting for uncertainty in user ratings.
    
    Parameters:
    - up: Number of positive votes
    - down: Number of negative votes
    - confidence: Confidence level (default: 0.95)
    
    Returns:
    - Lower bound of the Wilson score interval
    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

######################################
# Example Usage with a Comment Dataset
######################################

# Sample upvote and downvote counts
up = [15, 70, 14, 4, 2, 5, 8, 37, 21, 52, 28, 147, 61, 30, 23, 40, 37, 61, 54, 18, 12, 68]
down = [0, 2, 2, 2, 15, 2, 6, 5, 23, 8, 12, 2, 1, 1, 5, 1, 2, 6, 2, 0, 2, 2]

# Create a DataFrame
comments = pd.DataFrame({"up": up, "down": down})

# Calculate different ranking scores
comments["score_pos_neg_diff"] = comments.apply(lambda x: differ_score(x["up"], x["down"]), axis=1)
comments["score_average_rating"] = comments.apply(lambda x: average_rating(x["up"], x["down"]), axis=1)
comments["wilson_lower_bound"] = comments.apply(lambda x: wilson_lower_bound(x["up"], x["down"]), axis=1)

# Sort comments based on different scoring methods
sorted_by_diff = comments.sort_values("score_pos_neg_diff", ascending=False)
sorted_by_avg = comments.sort_values("score_average_rating", ascending=False)
sorted_by_wilson = comments.sort_values("wilson_lower_bound", ascending=False)

# Display sorted results
print("Sorted by Positive-Negative Difference:")
print(sorted_by_diff.head())

print("\nSorted by Average Rating:")
print(sorted_by_avg.head())

print("\nSorted by Wilson Lower Bound:")
print(sorted_by_wilson.head())
