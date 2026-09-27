"""Cheap text normalization — no external APIs / geocoding."""

from __future__ import annotations

import re
import unicodedata

_SUFFIX_EXPAND = {
    r"\bcorp\.?\b": "corporation",
    r"\binc\.?\b": "incorporated",
    r"\bltd\.?\b": "limited",
    r"\bllc\.?\b": "limited liability company",
    r"\bpvt\.?\b": "private",
    r"\bco\.?\b": "company",
    r"\bllp\.?\b": "limited liability partnership",
    r"\bsarl\.?\b": "sarl",
    r"\bsa\.?\b": "sa",
    r"\bgmb\.?h\.?\b": "gmbh",
}

_SUFFIX_STRIP = re.compile(
    r"\b(corporation|incorporated|limited|private|company|llc|llp|"
    r"limited liability company|limited liability partnership|"
    r"pvt|ltd|inc|corp|sarl|gmbh|sa|plc)\b",
    re.I,
)

_ADDR_EXPAND = {
    r"\brd\.?\b": "road",
    r"\bst\.?\b": "street",
    r"\bave\.?\b": "avenue",
    r"\bblvd\.?\b": "boulevard",
    r"\bdr\.?\b": "drive",
    r"\bln\.?\b": "lane",
    r"\bct\.?\b": "court",
    r"\bhwy\.?\b": "highway",
    r"\bp\.?\s*o\.?\s*box\b": "po box",
    r"\bnr\.?\b": "near",
    r"\bopp\.?\b": "opposite",
    # France (test-only country) — local expansions only, no geocoding.
    r"\brue\b": "street",
    r"\bav\.?\b": "avenue",
    r"\bbd\.?\b": "boulevard",
    r"\bboul\.?\b": "boulevard",
    r"\bste\.?\b": "sainte",
    r"\bcedex\b": "cedex",
    r"\bpl\.?\b": "place",
    r"\ballee\b": "allee",
    r"\ballée\b": "allee",
    r"\bimpasse\b": "impasse",
    r"\bchemin\b": "chemin",
}

# Very common tokens that make terrible solo blocking keys
_STOP_NAME = {
    "the", "and", "of", "for", "new", "best", "global", "india", "indian",
    "american", "general", "national", "international", "united", "first",
    "services", "service", "group", "trading", "traders", "enterprises",
    "solutions", "technologies", "technology", "systems", "system",
}

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")
_DIGITS = re.compile(r"\d+")


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower()


def _apply_map(text: str, mapping: dict[str, str]) -> str:
    for pat, repl in mapping.items():
        text = re.sub(pat, repl, text, flags=re.I)
    return text


def normalize_name(name: str, strip_suffix: bool = True) -> str:
    name = _fold(name or "")
    name = name.replace("&", " and ")
    name = _apply_map(name, _SUFFIX_EXPAND)
    name = _PUNCT.sub(" ", name)
    name = _WS.sub(" ", name).strip()
    if strip_suffix:
        name = _SUFFIX_STRIP.sub(" ", name)
        name = _WS.sub(" ", name).strip()
    return name


def normalize_address(addr: str) -> str:
    addr = _fold(addr or "")
    addr = _apply_map(addr, _ADDR_EXPAND)
    addr = _PUNCT.sub(" ", addr)
    addr = _WS.sub(" ", addr).strip()
    return addr


def name_tokens(name: str) -> list[str]:
    return [t for t in normalize_name(name).split() if len(t) > 1]


def significant_name_tokens(name: str) -> list[str]:
    return [t for t in name_tokens(name) if t not in _STOP_NAME and not t.isdigit()]


def address_tokens(addr: str) -> list[str]:
    return [t for t in normalize_address(addr).split() if len(t) > 1]


def extract_digits(text: str) -> set[str]:
    return set(_DIGITS.findall(text or ""))


def phonetic_key(name: str) -> str:
    toks = significant_name_tokens(name) or name_tokens(name)
    if not toks:
        return ""
    return "".join(t[:3] for t in toks[:3])


def blocking_keys(name: str, address: str, country: str) -> set[str]:
    """Multiple overlapping keys → higher blocking recall without ANN."""
    keys: set[str] = set()
    c = (country or "").strip().upper() or "UNK"
    toks = significant_name_tokens(name) or name_tokens(name)
    raw_toks = name_tokens(name)
    digits = extract_digits(address)
    addr_toks = address_tokens(address)

    if toks:
        keys.add(f"{c}|n1|{toks[0]}")
        if len(toks) >= 2:
            keys.add(f"{c}|n1b|{toks[1]}")
            keys.add(f"{c}|n2|{toks[0]}_{toks[1]}")
            a, b = sorted((toks[0], toks[1]))
            keys.add(f"{c}|ns|{a}_{b}")
        if len(toks) >= 3:
            keys.add(f"{c}|n13|{toks[0]}_{toks[2]}")
            keys.add(f"{c}|n1c|{toks[2]}")
        if len(toks) >= 2:
            keys.add(f"{c}|nfl|{toks[0]}_{toks[-1]}")
        keys.add(f"{c}|ph|{phonetic_key(name)}")
        joined = "".join(toks)[:12]
        if len(joined) >= 4:
            keys.add(f"{c}|np|{joined}")
            keys.add(f"{c}|np4|{joined[:4]}")
            keys.add(f"{c}|np6|{joined[:6]}")

    # Fallback if only stopwords / empty after filter
    if not toks and raw_toks:
        keys.add(f"{c}|n1|{raw_toks[0]}")

    for d in digits:
        if len(d) >= 5:
            keys.add(f"{c}|pin|{d}")
        elif len(d) >= 3:
            # Street / house numbers often 3–4 digits
            if toks:
                keys.add(f"{c}|hn|{toks[0]}_{d}")

    if addr_toks and toks:
        # Prefer longer address tokens (city / locality-ish)
        long_addr = [t for t in addr_toks if len(t) >= 4 and not t.isdigit()]
        for at in long_addr[-2:]:
            keys.add(f"{c}|nc|{toks[0]}_{at}")

    return keys
