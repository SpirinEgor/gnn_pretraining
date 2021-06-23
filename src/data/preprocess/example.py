import dataclasses
from typing import Optional


@dataclasses.dataclass
class Example:
    language: str
    project_name: str
    file_name: str
    source_code: str
    docstring: Optional[str] = None
