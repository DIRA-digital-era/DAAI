#DAAI/api/input.py
import unicodedata
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError
from typing import Optional


def normalize_text(s: Optional[str]) -> str:
    if not s:
        return ""
    try:
        s = s.strip()
        s = unicodedata.normalize("NFC", s)
        s = s.encode("utf-8", "ignore").decode("utf-8")
        # remove control characters
        s = "".join(ch for ch in s if ch.isprintable() and ch not in "\r\n\t\x0b\x0c")
        # collapse multiple spaces
        s = " ".join(s.split())
        return s
    except Exception:
        return ""


def sanitize_filename(name: str) -> str:
    try:
        # just the name, no path traversal
        name = Path(name).name

        # normalize to NFC
        name = unicodedata.normalize("NFC", name)

        # drop garbage chars
        name = name.encode("utf-8", "ignore").decode("utf-8")
        name = "".join(ch for ch in name if ch.isprintable())

        # enforce size and fallback
        if not name or len(name) > 200:
            return "upload.jpg"

        return name
    except Exception:
        return "upload.jpg"


class InputMetadata(BaseModel):
    file_name: Optional[str] = Field(None, max_length=200)
    notes: Optional[str] = Field(None, max_length=2000)
    locale: Optional[str] = Field(None, max_length=10)
    crop_id: Optional[str] = None
    variety_id: Optional[str] = None
    uploader_id: Optional[str] = None

    def sanitize(self):
        self.file_name = sanitize_filename(self.file_name) if self.file_name else None
        self.notes = normalize_text(self.notes)
        self.locale = normalize_text(self.locale)
        self.crop_id = normalize_text(self.crop_id)
        self.variety_id = normalize_text(self.variety_id)
        self.uploader_id = normalize_text(self.uploader_id)
        return self


def safe_metadata(raw: dict) -> InputMetadata:
    """
    Safe constructor that prevents poisoned input from entering the pipeline.
    """
    try:
        model = InputMetadata(**raw)
        return model.sanitize()
    except ValidationError as e:
        # Return a blank sanitized metadata object if validation fails
        return InputMetadata().sanitize()
