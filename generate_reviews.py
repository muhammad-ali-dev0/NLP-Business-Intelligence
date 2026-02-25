"""
generate_reviews.py
-------------------
Generates a realistic synthetic dataset of 2,000 product reviews
across 5 categories for the NLP BI pipeline demo.

Run: python data/generate_reviews.py
Output: data/reviews.csv
"""

import random
import csv
import os
from datetime import datetime, timedelta

random.seed(42)

# ── REVIEW TEMPLATES ─────────────────────────────────────────────────────────

CATEGORIES = ["Electronics", "Clothing", "Home & Kitchen", "Sports", "Beauty"]

PRODUCTS = {
    "Electronics": ["wireless headphones", "smartwatch", "laptop stand", "USB hub", "webcam"],
    "Clothing":    ["running shoes", "winter jacket", "yoga pants", "cotton t-shirt", "backpack"],
    "Home & Kitchen": ["air fryer", "coffee maker", "knife set", "cutting board", "blender"],
    "Sports":      ["resistance bands", "foam roller", "jump rope", "water bottle", "gym gloves"],
    "Beauty":      ["moisturizer", "vitamin C serum", "sunscreen", "lip balm", "face wash"],
}

POSITIVE_PHRASES = [
    "Absolutely love this product!", "Exceeded my expectations by far.",
    "Best purchase I've made this year.", "Works perfectly right out of the box.",
    "Incredibly well-built and durable.", "The quality is outstanding.",
    "Shipping was fast and packaging was excellent.", "Highly recommend to everyone.",
    "Great value for the price.", "Will definitely buy again.",
    "Customer service was fantastic when I had questions.",
    "Easy to set up and use immediately.", "Looks exactly like the pictures.",
    "My family loves it, using it every single day.",
    "Solved my problem completely, very happy.",
]

NEUTRAL_PHRASES = [
    "Decent product for the price.", "Does what it says, nothing special.",
    "Average quality, met basic expectations.", "OK but not great.",
    "Arrived on time, product is fine.", "Works as described.",
    "Some minor issues but overall acceptable.", "Not bad, not amazing.",
    "Would consider buying again if price drops.", "Reasonable for the cost.",
    "Instructions could be clearer.", "Took a while to figure out.",
    "Looks a bit cheap but functions okay.", "Average experience overall.",
]

NEGATIVE_PHRASES = [
    "Very disappointed with this purchase.", "Broke after just two weeks of use.",
    "Does not match the product description at all.", "Poor quality materials.",
    "Terrible customer service experience.", "Would not recommend to anyone.",
    "Complete waste of money.", "Arrived damaged, return process was a nightmare.",
    "Misleading product photos online.", "Stopped working after a month.",
    "Cheaply made, definitely not worth the price.", "Had to return it immediately.",
    "Size was way off from what was listed.", "Nothing like advertised.",
]

TOPICS = {
    "product_quality": [
        "quality", "build", "durable", "sturdy", "cheap", "flimsy", "well-made",
        "breaks", "lasts", "solid", "premium", "material"
    ],
    "shipping_delivery": [
        "shipping", "delivery", "arrived", "package", "fast", "late", "damaged",
        "tracking", "delay", "box", "courier", "transit"
    ],
    "customer_service": [
        "service", "support", "response", "refund", "return", "helpful", "rude",
        "contacted", "customer care", "resolved", "unhelpful", "team"
    ],
    "value_for_money": [
        "price", "value", "worth", "expensive", "cheap", "affordable", "money",
        "cost", "budget", "overpriced", "deal", "reasonable"
    ],
    "ease_of_use": [
        "easy", "simple", "instructions", "complicated", "intuitive", "setup",
        "manual", "difficult", "user-friendly", "confusing", "straightforward"
    ],
}


def generate_review(sentiment_bias: float) -> dict:
    """Generate one synthetic review. sentiment_bias: 0=negative, 0.5=neutral, 1=positive."""
    category = random.choice(CATEGORIES)
    product  = random.choice(PRODUCTS[category])

    # Choose sentiment bucket
    r = random.random() + (sentiment_bias - 0.5) * 0.6
    if r > 0.65:
        phrases    = POSITIVE_PHRASES
        stars      = random.choice([4, 5, 5, 5])
        sentiment  = "positive"
    elif r > 0.35:
        phrases    = NEUTRAL_PHRASES
        stars      = random.choice([3, 3, 4])
        sentiment  = "neutral"
    else:
        phrases    = NEGATIVE_PHRASES
        stars      = random.choice([1, 1, 2])
        sentiment  = "negative"

    # Pick topic keywords to inject
    main_topic = random.choice(list(TOPICS.keys()))
    topic_words = random.sample(TOPICS[main_topic], k=random.randint(1, 3))

    # Build review text
    num_phrases = random.randint(2, 4)
    selected = random.sample(phrases, k=min(num_phrases, len(phrases)))
    topic_sentence = f"The {topic_words[0]} {random.choice(['was', 'is', 'seemed'])} {random.choice(['great', 'poor', 'average', 'impressive', 'disappointing'])}."
    text_parts = selected + [topic_sentence]
    random.shuffle(text_parts)
    text = " ".join(text_parts) + f" Bought for my {random.choice(['home', 'office', 'gym', 'travel', 'daily use'])}."

    # Random date in last 24 months
    days_ago = random.randint(1, 730)
    date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")

    return {
        "review_id":    f"REV{random.randint(100000, 999999)}",
        "date":         date,
        "category":     category,
        "product":      product,
        "stars":        stars,
        "sentiment":    sentiment,
        "main_topic":   main_topic,
        "text":         text,
        "verified":     random.choice([True, True, True, False]),
        "helpful_votes": random.randint(0, 142),
    }


def generate_dataset(n: int = 2000) -> list[dict]:
    reviews = []
    # Vary sentiment bias by category
    bias_map = {
        "Electronics":    0.62,
        "Clothing":       0.55,
        "Home & Kitchen": 0.70,
        "Sports":         0.65,
        "Beauty":         0.58,
    }
    for _ in range(n):
        # Pick a category first to get its bias
        cat = random.choice(CATEGORIES)
        bias = bias_map[cat]
        review = generate_review(bias)
        review["category"] = cat
        review["product"]  = random.choice(PRODUCTS[cat])
        reviews.append(review)
    return reviews


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    reviews = generate_dataset(2000)

    fieldnames = ["review_id", "date", "category", "product", "stars",
                  "sentiment", "main_topic", "text", "verified", "helpful_votes"]

    out_path = os.path.join(os.path.dirname(__file__), "reviews.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(reviews)

    print(f"✅ Generated {len(reviews)} reviews → {out_path}")
    sentiments = [r["sentiment"] for r in reviews]
    print(f"   Positive: {sentiments.count('positive')} | "
          f"Neutral: {sentiments.count('neutral')} | "
          f"Negative: {sentiments.count('negative')}")
