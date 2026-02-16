from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import Any, ClassVar, Optional

MAX_TOKENS = 100


@dataclass_json
@dataclass
class Item:

    item_id: str
    title: str
    category: str
    description: str
    price: float
    summary: Optional[str] = None
    prompt: Optional[str] = None
    completion: Optional[str] = None

    question: ClassVar[str] = (
        "Given the following description, what is the price of this product"
    )
    prefix: ClassVar[str] = "Price of the product is $"

    def __post_init__(self):
        if self.price is not None:
            self.price = float(self.price)

    def make_prompt(
        self,
        tokenizer: Any,
        max_tokens: int = MAX_TOKENS,
    ) -> None:
        tokens = tokenizer.encode(self.summary, add_special_tokens=False)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        summary = tokenizer.decode(tokens).rstrip()
        # create prompt
        self.prompt = f"{self.question}\n{summary}\n{self.prefix}"
        # create completion
        self.completion = f"{self.price:.2f}"
