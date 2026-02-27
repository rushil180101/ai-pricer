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
MAX_DEALS = 10

SYSTEM_PROMPT = """
    You are given a list of deals on various products along with the product description and its deal price.
    You are also given another main product description which the user wants to buy. Your job is to figure out
    if there are any deal products which match the user's requirement. Return the list of matching deal products.
    If no deals match user's product requirements, then do not return anything. Also rephrase the product
    description in matching deals to summarize it in 2-3 sentences. Preserve the url of each product deal page.
    """
USER_PROMPT = """
    Here is the user provided product description: {product_description}\n
    Following is the list of general deals found from various sources\n
    {deals}\n
    Return the most relevant deals which match user's product requirements.
    If no deals match, then do not return anything.
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

    def get_relevant_deals(self, product_description: str) -> DealSelection:
        scraped_deals = self.scrape_deals()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT.format(
                    product_description=product_description,
                    deals="\n\n".join(scraped_deals),
                ),
            },
        ]
        response = self.openai_client.chat.completions.parse(
            messages=messages,
            model=MODEL,
            response_format=DealSelection,
        )
        deal_selection = response.choices[0].message.parsed
        return deal_selection
