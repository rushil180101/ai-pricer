from models.item import Item
from typing import Optional
import re
import json

MIN_PRICE = 0.5
MAX_PRICE = 1000.0
MIN_CHARS = 500
MAX_CHARS = 5000
item_idx = 0


def scrub(contents: str) -> str:
    replacement_regex = r"[\[\]:'\"\{\};\/\\\(\)\*]+"
    multiple_whitespaces_regex = r" {2,}"
    contents = re.sub(replacement_regex, " ", contents)
    contents = re.sub(multiple_whitespaces_regex, " ", contents)
    contents = contents.encode("ascii", "ignore").decode()
    contents = contents.strip()
    return contents


def parse(datapoint: dict, category: str) -> Optional[Item]:
    global item_idx
    title = datapoint["title"]
    price = datapoint["price"]
    price_pattern = r"^\d+(\.\d+)?$"
    if not re.match(price_pattern, price):
        return

    if price.lower() != "none" and (MIN_PRICE <= float(price) <= MAX_PRICE):
        contents = ""
        contents += "\n".join(datapoint["features"]) + "\n"
        contents += "\n".join(datapoint["description"]) + "\n"
        details = [
            f"{key}: {value}"
            for key, value in json.loads(datapoint["details"]).items()
            if not key.lower()
            .strip()
            .endswith("number")  # exclude product or item numbers
        ]
        contents += "\n".join(details)
        contents = scrub(contents)

        if MIN_CHARS <= len(contents):
            item = Item(
                item_id=item_idx,
                title=title,
                category=category,
                description=contents[:MAX_CHARS],
                price=price,
            )
            item_idx += 1
            return item
