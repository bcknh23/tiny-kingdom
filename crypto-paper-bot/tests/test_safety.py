"""Guard rails: the code base must stay read-only toward exchanges."""

import io
import re
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = [p for d in ("paperbot", "dashboard") for p in (ROOT / d).rglob("*.py")]

FORBIDDEN = [
    r"\.\s*(post|put|delete|patch)\s*\(",  # only HTTP GET is allowed
    r"api_?key",
    r"api_?secret",
    r"passphrase",
    r"withdraw",
    r"/orders",
    r"ccxt",
    r"environ",  # no credentials pulled from the environment (config paths aside)
]
ALLOWED_ENV = {"PAPERBOT_DB", "PAPERBOT_LOG"}


def code_without_docs(path: Path) -> str:
    """Source text with comments and docstrings removed (string literals kept)."""
    out = []
    tokens = tokenize.generate_tokens(io.StringIO(path.read_text()).readline)
    prev = tokenize.INDENT
    for tok in tokens:
        if tok.type == tokenize.COMMENT:
            continue
        is_docstring = tok.type == tokenize.STRING and prev in (tokenize.INDENT, tokenize.NEWLINE, tokenize.DEDENT)
        if not is_docstring:
            out.append(tok.string)
        if tok.type not in (tokenize.NL, tokenize.COMMENT):
            prev = tok.type
    return " ".join(out).lower()


def test_sources_found():
    assert len(SOURCES) >= 8


def test_no_order_or_credential_code():
    for path in SOURCES:
        code = code_without_docs(path)
        for env in ALLOWED_ENV:
            code = code.replace(f'environ . get ( "{env.lower()}"', "")
        for pattern in FORBIDDEN:
            assert not re.search(pattern, code), f"{path.name} matches forbidden pattern {pattern!r}"


def test_http_layer_only_issues_get():
    code = code_without_docs(ROOT / "paperbot" / "data.py")
    assert "session . get (" in code
    assert not re.search(r"session \. (request|post|put|delete|patch)", code)
