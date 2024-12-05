import typing

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: typing.Literal["ok"] = Field(default="ok")
    version: typing.Text = Field(default="0.1.0")
    name: typing.Text = Field(default="DVS")
    description: typing.Text = Field(default="")
