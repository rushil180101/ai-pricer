from pydantic import BaseModel, Field
from typing import List


class Deal(BaseModel):

    title: str = Field(description="Title of the product")
    summary: str = Field(description="Summary of the product in 3-4 sentences")
    price: float = Field(description="Price of the product")
    url: str = Field(description="Link to the product deal page")


class DealSelection(BaseModel):

    deals: List[Deal] = Field(description="List of deals")
