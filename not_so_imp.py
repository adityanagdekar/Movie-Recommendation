def visualize_top_keywords():

    # Collect all keywords from the high popularity group
    high_keywords = []
    for row in df_high_pop['keywords']:
        high_keywords.extend(row)

    # Collect all keywords from the low popularity group
    low_keywords = []
    for row in df_low_pop['keywords']:
        low_keywords.extend(row)

    # Create frequency counters
    high_freq_counter = Counter(high_keywords)
    low_freq_counter  = Counter(low_keywords)

    # Most common in high popularity group
    print("\nTop 20 keywords in HIGH popularity movies:")
    print(high_freq_counter.most_common(20))

    print("\nTop 20 keywords in LOW popularity movies:")
    print(low_freq_counter.most_common(20))

    import math

    # Combine keys from both counters
    all_keywords_set = set(high_freq_counter.keys()) | set(low_freq_counter.keys())

    comparison = []
    for keyword in all_keywords_set:
        high_count = high_freq_counter[keyword]
        low_count  = low_freq_counter[keyword]

        # relative freq. w.r.t High_Popularity subset
        high_rel_freq = high_count / len(df_high_pop) if len(df_high_pop) > 0 else 0

        # relative freq. w.r.t Low_Popularity subset
        low_rel_freq  = low_count / len(df_low_pop)   if len(df_low_pop) > 0 else 0

        # We'll compute a simple ratio or difference
        ratio = (high_rel_freq + 1e-6) / (low_rel_freq + 1e-6)
        comparison.append((keyword, high_rel_freq, low_rel_freq, ratio))

    # Sort by ratio (which indicates how much more (ratio>1) or less (ratio<1) common in high-pop movies)
    comparison.sort(key=lambda x: x[3], reverse=True)

    # Display top 20
    print("Keyword, High_Popularity_Relative_Freq, Low_Popularity_Relative_Freq, Ratio")
    for row in comparison[:20]:
        print(row)


    import matplotlib.pyplot as plt

    top_keywords = comparison[:10]  # top 10 most used keywords in high popularity
    labels = [x[0] for x in top_keywords]
    # rounding ratios to 2 decimal places
    ratios = [round(x[3], 2) for x in top_keywords]    # plt command removed    # plt command removed    # plt command removed    # plt command removed

    '''
    This transforms the Y-axis into a logarithmic scale.
    So instead of showing linear steps like 10,000 → 20,000 → 30,000, it shows powers of 10.
    This is especially useful when your values range from small (e.g., 1x or 10x) to very large (e.g., 30,000x).
    '''    # plt command removed    # plt command removed

    # this will help to print bar-height on-top of the bar
    bars =    # plt command removed
    for bar, ratio in zip(bars, ratios):
        bar_height = bar.get_height()    # plt command removed    # plt command removed    # plt command removed



    # Collect all keywords from the low popularity group
    low_keywords = []
    for row in df_low_pop['keywords']:
        low_keywords.extend(row)

    # Collect all keywords from the low popularity group
    low_keywords = []
    for row in df_low_pop['keywords']:
        low_keywords.extend(row)

    # Create frequency counters
    high_freq_counter = Counter(low_keywords)
    low_freq_counter  = Counter(low_keywords)

    # Most common in high popularity group
    print("\nTop 20 keywords in HIGH popularity movies:")
    print(high_freq_counter.most_common(20))

    print("\nTop 20 keywords in LOW popularity movies:")
    print(low_freq_counter.most_common(20))

def visualize_top_genres():
    # Collect all genres from the high popularity group
    high_genres = []
    for row in df_high_pop['genres']:
        high_genres.extend(row)

    # Collect all keywords from the low popularity group
    low_genres = []
    for row in df_low_pop['keywords']:
        low_genres.extend(row)

    # Create frequency counters
    high_freq_counter = Counter(high_genres)
    low_freq_counter  = Counter(low_genres)

    # Most common in high popularity group
    print("\nTop 20 Genres in HIGH popularity movies:")
    print(high_freq_counter.most_common(20))

    print("\nTop 20 Genres in LOW popularity movies:")
    print(low_freq_counter.most_common(20))

    import math

    # Combine keys from both counters
    all_genres_set = set(high_freq_counter.keys()) | set(low_freq_counter.keys())

    genre_comparison_list = []
    for genre in all_genres_set:
        high_count = high_freq_counter[genre]
        low_count  = low_freq_counter[genre]

        # relative freq. w.r.t High_Popularity subset
        high_rel_freq = high_count / len(df_high_pop) if len(df_high_pop) > 0 else 0

        # relative freq. w.r.t Low_Popularity subset
        low_rel_freq  = low_count / len(df_low_pop)   if len(df_low_pop) > 0 else 0

        # We'll compute a simple ratio or difference
        ratio = (high_rel_freq + 1e-6) / (low_rel_freq + 1e-6)
        genre_comparison_list.append((genre, high_rel_freq, low_rel_freq, ratio))

    # Sort by ratio in descending order
    # (which indicates how much more (ratio>1) or less (ratio<1) common in high-popularity movies)
    # Each ele. in genre_comparison_list is a tuple -> (high_rel_freq, low_rel_freq, ratio), 
    # ratio is the 3rd idx. in the tuple
    genre_comparison_list.sort(key=lambda x: x[3], reverse=True)

    # Display top 20 genres
    print("Genres, High_Popularity_Relative_Freq, Low_Popularity_Relative_Freq, Ratio")
    for row in genre_comparison_list[:20]:
        print(row)


    import matplotlib.pyplot as plt

    top_keywords = genre_comparison_list[:10]  # top 10 most used genres in high popularity
    labels = [x[0] for x in top_keywords]
    # rounding ratios to 2 decimal places
    ratios = [round(x[3], 2) for x in top_keywords]    # plt command removed    # plt command removed    # plt command removed    # plt command removed

    '''
    This transforms the Y-axis into a logarithmic scale.
    So instead of showing linear steps like 10,000 → 20,000 → 30,000, it shows powers of 10.
    This is especially useful when your values range from small (e.g., 1x or 10x) to very large (e.g., 30,000x).
    '''    # plt command removed    # plt command removed

    # this will help to print bar-height on-top of the bar
    bars =    # plt command removed
    for bar, ratio in zip(bars, ratios):
        bar_height = bar.get_height()    # plt command removed    # plt command removed    # plt command removed


    user_inp_movie = "spectre"
    user_inp_movieId = getMovieId(user_inp_movie)
    print(f"input movie = {user_inp_movie} & it's genres: {set(summarised_movies_df.iloc[user_inp_movieId]["genres"])}")
    movie_ids_list = getRecommendationsList(user_inp_movie)
    for id in movie_ids_list:
        score = f"{get_jaccard_score(user_inp_movieId, id):.2f}"
        print(f"{summarised_movies_df.iloc[id]["title"]:50s} score={score}  genres={set(summarised_movies_df.iloc[id]["genres"])}")
