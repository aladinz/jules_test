from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def analyze_sentiment_vader(text_list: list[str]) -> list[dict]:
    """
    Analyzes the sentiment of a list of text strings using VADER.

    Args:
        text_list (list[str]): A list of text strings (e.g., news headlines).

    Returns:
        list[dict]: A list of dictionaries, where each dictionary contains:
                    - 'text': The original text string.
                    - 'sentiment': A dictionary of sentiment scores ('neg', 'neu', 'pos', 'compound')
                                   from VADER.
                    Returns an empty list if the input is not a list or is empty.
    """
    if not isinstance(text_list, list):
        print("Error: Input must be a list of strings.")
        return []
    if not text_list:
        return []

    analyzer = SentimentIntensityAnalyzer()
    results = []

    for text in text_list:
        if not isinstance(text, str):
            print(f"Warning: Skipping non-string item in list: {text}")
            sentiment_scores = {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0, 'error': 'Input was not a string'}
        else:
            sentiment_scores = analyzer.polarity_scores(text)

        results.append({
            "text": text,
            "sentiment": sentiment_scores
        })

    return results

def get_average_sentiment_score(sentiment_scores_list: list[dict]) -> float:
    """
    Calculates the average 'compound' sentiment score from a list of sentiment results.

    Args:
        sentiment_scores_list (list[dict]): A list of dictionaries, where each dictionary
                                           is expected to have a 'sentiment' key, which
                                           in turn has a 'compound' key.
                                           (Output from analyze_sentiment_vader).
    Returns:
        float: The average compound score. Returns 0.0 if the list is empty or
               no valid compound scores are found.
    """
    if not isinstance(sentiment_scores_list, list) or not sentiment_scores_list:
        return 0.0

    compound_scores = []
    for item in sentiment_scores_list:
        # Check if 'sentiment' key exists and is a dictionary, and 'compound' key exists within it
        if isinstance(item, dict) and \
           'sentiment' in item and isinstance(item['sentiment'], dict) and \
           'compound' in item['sentiment'] and isinstance(item['sentiment']['compound'], (int, float)):
            compound_scores.append(item['sentiment']['compound'])
        # else:
            # print(f"Warning: Skipping item due to missing or invalid sentiment/compound score: {item}")


    if not compound_scores:
        return 0.0

    return sum(compound_scores) / len(compound_scores)


if __name__ == '__main__':
    sample_headlines = [
        "Stock skyrockets on amazing new tech discovery!", # Positive
        "Market crashes बुरी तरह after unexpected downturn.", # Negative (includes non-ASCII)
        "Company X announces regular quarterly earnings.", # Neutral
        "This is a great and wonderful achievement for the team.", # Positive
        "Awful news today, the project has failed miserably.", # Negative
        "The weather today is just normal.", # Neutral
        "", # Empty string
        None, # Non-string input
        f"Another day, another dollar for a steady company." # Neutral-ish
    ]

    print("Analyzing sample headlines with VADER:")
    sentiment_results = analyze_sentiment_vader(sample_headlines)

    for result in sentiment_results:
        print(f"Text: \"{result['text']}\"")
        print(f"  Sentiment Scores: {result['sentiment']}")
        if 'error' in result['sentiment']:
            print(f"  Error: {result['sentiment']['error']}")
        else:
            # Determine overall sentiment based on compound score for display
            compound = result['sentiment']['compound']
            overall_sentiment = "Neutral"
            if compound >= 0.05:
                overall_sentiment = "Positive"
            elif compound <= -0.05:
                overall_sentiment = "Negative"
            print(f"  Overall VADER sentiment: {overall_sentiment} (Compound: {compound:.4f})")
        print("-" * 20)

    # Test with empty list
    print("\nAnalyzing empty list:")
    empty_results = analyze_sentiment_vader([])
    print(f"Result for empty list: {empty_results}")

    # Test with non-list input
    print("\nAnalyzing non-list input:")
    non_list_results = analyze_sentiment_vader("this is not a list")
    print(f"Result for non-list input: {non_list_results}")

    # Test get_average_sentiment_score
    # Filter out results with errors for a cleaner average calculation for this test
    valid_results_for_average = [r for r in sentiment_results if 'error' not in r['sentiment']]
    if valid_results_for_average:
        average_score = get_average_sentiment_score(valid_results_for_average)
        print(f"\nAverage compound score for valid headlines: {average_score:.4f}")
    else:
        print("\nNo valid headlines to calculate an average score.")

    # Test average with an empty list
    avg_empty = get_average_sentiment_score([])
    print(f"Average score for empty list: {avg_empty}")

    # Test average with list containing items without proper sentiment structure
    avg_malformed = get_average_sentiment_score([{"text": "test", "sentiment": {"score": 0.5}}]) # Missing 'compound'
    print(f"Average score for malformed list: {avg_malformed}")

    avg_malformed_2 = get_average_sentiment_score([{"text": "test"}]) # Missing 'sentiment'
    print(f"Average score for malformed list (2): {avg_malformed_2}")
