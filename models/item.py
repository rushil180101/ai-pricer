from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import Optional


@dataclass_json
@dataclass
class Item:

    item_id: str
    title: str
    category: str
    description: str
    price: float
    summary: Optional[str] = None

    def __post_init__(self):
        if self.price is not None:
            self.price = float(self.price)
