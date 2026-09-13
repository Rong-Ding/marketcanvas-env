"""Validated, bounded data contracts. No network or model inference in the simulator."""
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

Color = Annotated[str, Field(pattern=r"^#[0-9a-fA-F]{6}$")]
Content = Annotated[str, Field(max_length=256)]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TaskSpec(Record):
    headline: str = Field(default="Summer Sale", min_length=1, max_length=48)
    cta: str = Field(default="Shop now", min_length=1, max_length=24)
    cta_color: Color = "#FFFF00"
    alignment: Literal["center", "left"] = "center"
    min_contrast: float = Field(default=4.5, ge=1, le=21)

    @field_validator("headline", "cta")
    @classmethod
    def printable(cls, value):
        if not value.strip() or any(ord(c) < 32 or ord(c) > 126 for c in value):
            raise ValueError("Task text must be nonempty printable ASCII in this prototype")
        return value

    def prompt(self):
        return (f'Create a banner with exactly one headline "{self.headline}" and one CTA button labeled '
                f'"{self.cta}", filled {self.cta_color}. Use {self.alignment} alignment, '
                f'place the CTA at least 16px below the headline, and keep text contrast at least {self.min_contrast}:1. '
                'Required text must be at least 16px, fully fit its box with padding, and remain visible.')


class ElementSpec(Record):
    type: Literal["text", "shape", "image"]
    role: Literal["headline", "cta", "decoration"] = "decoration"
    x: int = Field(default=100, strict=True, ge=0, le=799)
    y: int = Field(default=100, strict=True, ge=0, le=599)
    width: int = Field(default=200, strict=True, ge=1, le=800)
    height: int = Field(default=60, strict=True, ge=1, le=600)
    z_index: int = Field(default=0, strict=True, ge=-100, le=100)
    color: Color | None = None
    text_color: Color = "#172033"
    content: Content = ""
    font_size: int = Field(default=28, strict=True, ge=8, le=96)

    @field_validator("content")
    @classmethod
    def printable(cls, value):
        if any(ord(c) < 32 or ord(c) > 126 for c in value):
            raise ValueError("Only single-line printable ASCII is supported")
        return value

    @model_validator(mode="after")
    def geometry(self):
        if self.x + self.width > 800 or self.y + self.height > 600:
            raise ValueError("Element must fit within the 800 x 600 canvas")
        if self.type in ("shape", "image") and self.color is None:
            raise ValueError("Shapes and image placeholders require an opaque fill color")
        return self


class Element(ElementSpec):
    id: str
    creation_index: int


class Add(Record):
    op: Literal["add_element"]
    element: ElementSpec


class Move(Record):
    op: Literal["move_element"]
    id: str
    x: int = Field(strict=True)
    y: int = Field(strict=True)


class Update(Record):
    op: Literal["update_element"]
    id: str
    # Full validation is applied atomically after merging. IDs and creation order cannot be edited.
    properties: dict


class Delete(Record):
    op: Literal["delete_element"]
    id: str


class Submit(Record):
    op: Literal["submit"]


Action = Annotated[Add | Move | Update | Delete | Submit, Field(discriminator="op")]
ACTION_ADAPTER = TypeAdapter(Action)


class State(Record):
    width: int = 800
    height: int = 600
    background: Color = "#FFFFFF"
    task: TaskSpec = TaskSpec()
    elements: tuple[Element, ...] = ()
    next_id: int = 1
    steps: int = 0
    max_steps: int = 40
    max_elements: int = 24
    terminated: bool = False
    seed: int = 0


def ordered(state: State):
    return sorted(state.elements, key=lambda e: (e.z_index, e.creation_index))


def observation(state: State):
    """Complete simulator state plus derived, non-editable spatial relationships."""
    result = state.model_dump(mode="json")
    result["prompt"] = state.task.prompt()
    result["remaining_steps"] = state.max_steps - state.steps
    relations = []
    for a in ordered(state):
        for b in ordered(state):
            if a.id == b.id:
                continue
            facts = []
            if a.y + a.height <= b.y:
                facts.append("above")
            if a.x + a.width <= b.x:
                facts.append("left_of")
            if a.x <= b.x and a.y <= b.y and a.x+a.width >= b.x+b.width and a.y+a.height >= b.y+b.height:
                facts.append("contains")
            if max(a.x, b.x) < min(a.x+a.width, b.x+b.width) and max(a.y, b.y) < min(a.y+a.height, b.y+b.height):
                facts.append("overlaps")
            if facts:
                relations.append({"subject": a.id, "object": b.id, "relations": facts})
    result["relationships"] = relations
    return result
