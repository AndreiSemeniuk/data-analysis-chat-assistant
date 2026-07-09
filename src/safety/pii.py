"""Deterministic PII masking.

Defense in depth, three layers:
  1. Column layer  - drop/mask columns whose names indicate PII before the
                     result ever reaches the LLM context.
  2. Value layer   - regex sweep over every string cell (emails, phone numbers)
                     to catch PII hiding in free-text columns.
  3. Output layer  - final regex sweep over the LLM-generated report, so even a
                     jailbroken model cannot leak PII to the user.

Masking is code, not a prompt instruction: the model never sees raw PII, so it
cannot be tricked into revealing it.
"""

import re

import pandas as pd

EMAIL_TOKEN = "[EMAIL REDACTED]"
PHONE_TOKEN = "[PHONE REDACTED]"

# Column names that must never surface raw values.
PII_COLUMN_PATTERN = re.compile(
    r"(^|_)(email|e_mail|phone|phone_number|mobile|telephone)($|_)", re.IGNORECASE
)

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# International-ish phone numbers: optional +country, separators, 7+ digits total.
PHONE_RE = re.compile(
    r"(?<![\w.])(\+?\d{1,3}[\s.-]?)?(\(?\d{2,4}\)?[\s.-]?)\d{3}[\s.-]?\d{2,4}([\s.-]?\d{2,4})?(?![\w.])"
)


def is_pii_column(column_name: str) -> bool:
    return bool(PII_COLUMN_PATTERN.search(column_name))


def mask_text(text: str) -> str:
    """Value/output layer: redact emails and phone numbers in free text."""
    text = EMAIL_RE.sub(EMAIL_TOKEN, text)

    def _phone_sub(match: re.Match) -> str:
        digits = re.sub(r"\D", "", match.group(0))
        # Require enough digits to plausibly be a phone number; avoids mangling
        # order ids, revenue figures, years etc.
        return PHONE_TOKEN if 9 <= len(digits) <= 15 else match.group(0)

    return PHONE_RE.sub(_phone_sub, text)


def mask_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Column + value layers. Returns the masked frame and a list of columns
    that were redacted (surfaced in traces for observability)."""
    df = df.copy()
    masked_columns: list[str] = []
    for col in df.columns:
        if is_pii_column(str(col)):
            df[col] = EMAIL_TOKEN if "mail" in str(col).lower() else PHONE_TOKEN
            masked_columns.append(str(col))
        elif pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
            df[col] = df[col].map(lambda v: mask_text(v) if isinstance(v, str) else v)
    return df, masked_columns
