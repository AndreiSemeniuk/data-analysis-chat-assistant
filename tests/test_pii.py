import pandas as pd

from src.safety.pii import EMAIL_TOKEN, PHONE_TOKEN, mask_dataframe, mask_text


def test_masks_emails_in_text():
    assert mask_text("contact dana.levi+vip@shop-mail.co.il today") == \
        f"contact {EMAIL_TOKEN} today"


def test_masks_phone_numbers_in_text():
    out = mask_text("call +1-555-123-4567 or 052 123 4567")
    assert PHONE_TOKEN in out
    assert "555" not in out


def test_does_not_mask_business_numbers():
    text = "Revenue was 1,234,567.89 in 2026; order id 4521987 had 3 items"
    assert mask_text(text) == text


def test_mask_dataframe_pii_columns_and_values():
    df = pd.DataFrame({
        "email": ["a@b.com"],
        "phone_number": ["+1-555-000-1111"],
        "notes": ["reach me at x@y.org"],
        "total_spend": [199.99],
    })
    masked, cols = mask_dataframe(df)
    assert set(cols) == {"email", "phone_number"}
    assert masked["email"].iloc[0] == EMAIL_TOKEN
    assert masked["phone_number"].iloc[0] == PHONE_TOKEN
    assert EMAIL_TOKEN in masked["notes"].iloc[0]          # value-layer sweep
    assert masked["total_spend"].iloc[0] == 199.99          # numbers untouched
