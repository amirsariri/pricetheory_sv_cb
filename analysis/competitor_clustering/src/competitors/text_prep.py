import re
import unicodedata

_LEGAL_SUFFIXES = re.compile(
    r"\b(inc\.?|llc|ltd\.?|limited|corp\.?|corporation)\b", flags=re.I
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
    text = _LEGAL_SUFFIXES.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text
