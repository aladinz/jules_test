import datetime
import random

def fetch_news(ticker: str) -> list[dict]:
    """
    Simulates fetching news headlines for a given stock ticker.

    Args:
        ticker (str): The stock ticker symbol.

    Returns:
        list[dict]: A list of news item dictionaries.
                    Each dictionary has 'date', 'headline', and 'source'.
                    Returns an empty list if an error occurs, though unlikely for simulation.
    """
    if not isinstance(ticker, str) or not ticker.strip():
        print("Error: Ticker symbol must be a non-empty string.")
        return []

    ticker = ticker.upper() # Standardize ticker format

    # Predefined templates for headlines
    positive_templates = [
        f"{ticker} stock surges on positive earnings report!",
        f"Analysts upgrade {ticker} to 'Buy' with new price target.",
        f"Successful product launch boosts {ticker} market sentiment.",
        f"{ticker} announces record profits for the quarter.",
        f"Innovation at {ticker} drives strong future outlook."
    ]
    negative_templates = [
        f"{ticker} shares fall after missing earnings expectations.",
        f"Regulatory concerns pressure {ticker} stock.",
        f"Technical issues plague {ticker}'s new platform.",
        f"Increased competition impacts {ticker}'s market share.",
        f"CEO of {ticker} steps down unexpectedly, stock tumbles."
    ]
    neutral_templates = [
        f"{ticker} to present at upcoming industry conference.",
        f"Market awaits {ticker}'s next earnings call.",
        f"{ticker} maintains steady performance in a volatile market.",
        f"Trading volume for {ticker} remains consistent.",
        f"What's next for {ticker}? Investors watch closely."
    ]

    simulated_news = []
    num_headlines = random.randint(3, 5) # Generate 3 to 5 headlines

    # Ensure a mix of sentiments if possible, or random selection
    all_templates = positive_templates + negative_templates + neutral_templates

    # Generate unique dates for the news items (e.g., last few days)
    today = datetime.date.today()
    possible_dates = [(today - datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(num_headlines + 5)]


    for i in range(num_headlines):
        # Randomly pick a template type, then a specific template
        chosen_template = random.choice(all_templates)

        # Assign a somewhat unique recent date
        news_date = random.choice(possible_dates)
        # Remove the chosen date to avoid duplicate dates for different headlines (simple approach)
        possible_dates.remove(news_date)


        simulated_news.append({
            "date": news_date,
            "headline": chosen_template,
            "source": "Simulated News Co."
        })

    # Sort news by date (descending - most recent first)
    try:
        simulated_news.sort(key=lambda x: datetime.datetime.strptime(x['date'], "%Y-%m-%d"), reverse=True)
    except ValueError:
        # This might happen if date format is inconsistent, though unlikely here
        print("Warning: Could not sort news items by date.")

    return simulated_news

if __name__ == '__main__':
    print("Fetching simulated news for AAPL:")
    aapl_news = fetch_news("AAPL")
    for item in aapl_news:
        print(f"- {item['date']}: {item['headline']} (Source: {item['source']})")

    print("\nFetching simulated news for GOOG:")
    goog_news = fetch_news("GOOG")
    for item in goog_news:
        print(f"- {item['date']}: {item['headline']} (Source: {item['source']})")

    print("\nFetching simulated news for a less common ticker (TSLA):")
    tsla_news = fetch_news("TSLA")
    for item in tsla_news:
        print(f"- {item['date']}: {item['headline']} (Source: {item['source']})")

    print("\nFetching news for an empty ticker (should show error or return empty):")
    empty_news = fetch_news("")
    if not empty_news:
        print("Returned empty list as expected for empty ticker.")

    print("\nFetching news for a ticker with spaces (should be handled):")
    space_news = fetch_news("  MSFT  ")
    for item in space_news:
        print(f"- {item['date']}: {item['headline']} (Source: {item['source']})")
