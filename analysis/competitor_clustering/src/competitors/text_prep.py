import re
import unicodedata

_LEGAL_SUFFIXES = re.compile(
    r"\b(inc\.?|llc|ltd\.?|limited|corp\.?|corporation|company|co\.?)\b", flags=re.I
)


def clean_text(text: str | None) -> str:
    """Normalise a short company description."""
    if not text:
        return ""

    text = (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode()
        .lower()
        .strip()
    )
    
    # Remove legal suffixes
    text = _LEGAL_SUFFIXES.sub("", text)
    
    # Preserve important business terms and market distinctions
    # Don't over-clean - keep specific terms like "ecommerce", "conversational", etc.
    text = re.sub(r"\s+", " ", text)
    
    # Remove trailing punctuation but keep internal structure
    text = text.strip(" .")
    
    return text
