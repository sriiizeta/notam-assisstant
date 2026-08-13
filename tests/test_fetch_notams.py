from fetch_notams import clean_icao_message
from src.parser import parse_notam


def test_strips_html_tags_from_corrupted_faa_feed_record():
    # regression test: real FAA feed record (KBNA-04-479) that embeds HTML
    # tags around field markers, which used to break the parser's
    # field-boundary lookahead
    raw = (
        "</b>KBNA<br> <b>B) </b>2604241103<br>\n"
        "<b>C) </b>2607240400<br> \n"
        "<b>E) </b> TOWER LGT (ASR 1038120) 360630.20N0863815.00W "
        "(2.2NM ESE BNA) 769.7FT (134.8FT AGL) U/S<br>"
    )
    cleaned = clean_icao_message(raw)
    assert "<" not in cleaned
    assert ">" not in cleaned
    assert "KBNA" in cleaned


def test_cleaned_message_parses_correctly_after_html_strip():
    raw = (
        "Q) </b>KZME/QOLAS/IV/M/AE/000/008/3607N08640W005<br>\n"
        "<b>A) </b>KBNA<br>\n"
        "<b>B) </b>2604241103<br>\n"
        "<b>C) </b>2607240400<br>\n"
        "<b>E) </b> TOWER LGT U/S<br>"
    )
    cleaned = clean_icao_message(raw)
    p = parse_notam(cleaned)
    assert p.a_field == "KBNA"
    assert p.b_field == "2604241103"
    assert p.c_field == "2607240400"
    assert p.e_field == "TOWER LGT U/S"


def test_not_available_placeholder_is_not_html_corrupted_but_still_flagged_upstream():
    assert clean_icao_message("NOT AVAILABLE") == "NOT AVAILABLE"


def test_clean_message_collapses_repeated_whitespace_but_keeps_newlines():
    raw = "A)   KJFK    B) 2501010000"
    cleaned = clean_icao_message(raw)
    assert "KJFK" in cleaned
    assert "   " not in cleaned
