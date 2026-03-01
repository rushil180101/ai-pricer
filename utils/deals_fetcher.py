import feedparser
import re
import requests
from bs4 import BeautifulSoup
from typing import List, Optional
from models.deals import DealSelection
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

RSS_FEED_URLS = ["https://bargainbabe.com/amazon-deals/feed/"]
MAX_DEALS = 5

SYSTEM_PROMPT = """
    You are responsible for rephrasing product details into a short summary. 
    """
USER_PROMPT = """
    Here is the list of product descriptions. Rephrase the product details\n
    {deals}
    """
MODEL = "gpt-4.1-nano"


class DealsFetcher:

    def __init__(self) -> None:
        self.rss_feed_urls = RSS_FEED_URLS
        self.openai_client = OpenAI()

    def extract_price(self, text: str) -> Optional[float]:
        pattern = r"\$(\d+(?:\.\d{2})?)"
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))

    def get_summary(self, url: str) -> str:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        content = soup.find(attrs={"class": "entry-content"})
        summary = content.get_text(strip=True)
        return summary

    def scrape_deals(self) -> List[str]:
        deals = []

        for rss_feed_url in self.rss_feed_urls:
            d = feedparser.parse(rss_feed_url)
            entries = (d.entries)[:MAX_DEALS]
            for entry in entries:
                title = entry.title
                url = entry.link
                price = self.extract_price(title)
                if price is None:
                    continue
                summary = self.get_summary(url)

                # Combine information to prepare the content
                contents = [
                    f"Title: {title}",
                    f"Summary: {summary}",
                    f"Price: ${price}",
                    f"Url: {url}",
                ]
                deal_content = "\n".join(contents)
                deals.append(deal_content)

        return deals

    def get_deals(self) -> DealSelection:
        scraped_deals = self.scrape_deals()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT.format(deals="\n\n".join(scraped_deals)),
            },
        ]
        response = self.openai_client.chat.completions.parse(
            messages=messages,
            model=MODEL,
            response_format=DealSelection,
        )
        deal_selection = response.choices[0].message.parsed
        return deal_selection
