from newspaper import Article, build
import os

# Create a folder for real articles
os.makedirs("real_articles", exist_ok=True)

# Build a news source (Reuters Science page as example)
cnn_paper = build('https://edition.cnn.com/tech', memoize_articles=False)

count = 0
for article in cnn_paper.articles:
    if count >= 20:  # limit to 20 articles
        break
    try:
        article.download()
        article.parse()
        text = article.text
        # Save to a txt file
        with open(f"real_articles/real_article_{count}.txt", "w") as f:
            f.write(text)
        print(f"Downloaded article {count}: {article.title}")
        count += 1
    except Exception as e:
        print(f"Error: {e}")
