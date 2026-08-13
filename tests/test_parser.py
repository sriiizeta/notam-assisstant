from src.parser import parse_notam, parse_notam_to_dict, extract_field, parse_q_field

WELL_FORMED = (
    "06/359 NOTAMR \r\n"
    "Q) KZNY/QMNHW/IV/BO/A/000/999/4038N07346W005 \r\n"
    "A) KJFK \r\n"
    "B) 2506180334 \r\n"
    "C) 2609052300 \r\n\r\n"
    "E) TERMINAL 1 RAMP WIP CONST LGTD AND BARRICADED"
)


def test_well_formed_notam_extracts_all_fields():
    p = parse_notam(WELL_FORMED)
    assert p.a_field == "KJFK"
    assert p.b_field == "2506180334"
    assert p.c_field == "2609052300"
    assert p.e_field == "TERMINAL 1 RAMP WIP CONST LGTD AND BARRICADED"
    assert p.q_line == "KZNY/QMNHW/IV/BO/A/000/999/4038N07346W005"


def test_q_field_splits_into_eight_parts():
    q = parse_q_field("KZNY/QMNHW/IV/BO/A/000/999/4038N07346W005")
    assert q["fir"] == "KZNY"
    assert q["q_code"] == "QMNHW"
    assert q["traffic"] == "IV"
    assert q["purpose"] == "BO"
    assert q["scope"] == "A"
    assert q["lower_limit"] == "000"
    assert q["upper_limit"] == "999"
    assert q["coords_radius"] == "4038N07346W005"


def test_q_field_with_missing_parts_does_not_crash():
    # real corpus example: KBNA-01-276 has empty fields between slashes
    q = parse_q_field("KZME//IV/M//000//3607N08640W005")
    assert q["fir"] == "KZME"
    assert q["q_code"] == ""


def test_q_field_none_returns_all_none():
    q = parse_q_field(None)
    assert all(v is None for v in q.values())


def test_missing_d_field_does_not_break_other_fields():
    text = (
        "Q) KZAU/QMXLC/IV/M/A/000/999/4359N08833W005\n"
        "A) KOSH\n"
        "B) 2607061448\n"
        "C) 2608072200\n\n"
        "E) TWY C3 CLSD"
    )
    p = parse_notam(text)
    assert p.d_field is None
    assert p.a_field == "KOSH"
    assert p.e_field == "TWY C3 CLSD"


def test_multiline_e_field_captured_in_full():
    text = (
        "Q) KZTL/QPIXX/I/NBO/A/000/999/3338N08425W005\n"
        "A) KATL\n"
        "B) 2507011526\n"
        "C) 2602101526EST\n"
        "E) ATL HARTSFIELD/JACKSON ATLANTA INTL, ATLANTA, GA.\n"
        "RNAV (RNP) Z RWY 8L, AMDT 1A...\n"
        "RNP 0.30 DA 1516/HAT 501 ALL CATS, VIS ALL CATS RVR 5500."
    )
    p = parse_notam(text)
    assert "RNAV (RNP) Z RWY 8L" in p.e_field
    assert "RVR 5500" in p.e_field


def test_extract_field_returns_none_when_absent():
    assert extract_field("Q) FOO A) BAR", "E") is None


def test_parse_notam_to_dict_matches_parse_notam():
    d = parse_notam_to_dict(WELL_FORMED)
    p = parse_notam(WELL_FORMED)
    assert d["a_field"] == p.a_field
    assert d["e_field"] == p.e_field
    assert set(d.keys()) == {
        "q_line", "fir", "q_code", "traffic", "purpose", "scope",
        "lower_limit", "upper_limit", "coords_radius",
        "a_field", "b_field", "c_field", "d_field", "e_field",
    }


def test_parenthetical_abbreviation_does_not_spawn_spurious_d_field():
    # regression test: real corpus example (KBED-06-054) -- "(2.0NM SW BED)"
    # ends in "D)" and used to get mistaken for a D) field marker, since the
    # D field is rare enough that there's often no genuine earlier "D)" to
    # match first
    text = (
        "Q) KZBW/QOLAS/IV/M/AE/000/004/4228N07117W005\n"
        "A) KBED\nB) 2606201718\nC) 2609181718\n"
        "E) TOWER LGT (ASR 1240303) 422645.70N0711907.70W (2.0NM SW BED) "
        "314.0FT (87.9FT AGL) U/S"
    )
    p = parse_notam(text)
    assert p.d_field is None
    assert "TOWER LGT" in p.e_field


def test_part_n_of_m_marker_stripped_from_a_field():
    # real corpus case: multi-part NOTAMs jam "PART 1 OF 2" after the airport
    # code in the A field; it should not survive into a_field
    text = "Q) KZNY/QARXX//////\nA) KZNY PART 1 OF 2\nB) 1003121253\nC) UFN\nE) TEST"
    assert parse_notam(text).a_field == "KZNY"


def test_multi_fir_a_field_is_preserved():
    # a legitimate multi-FIR A field (two ICAO codes) must NOT be truncated
    text = "Q) KZDV/QARLC/IV/NBO/E/000/999/\nA) KZDV KZMP\nB) 2508181752\nC) 2603301748EST\nE) TEST"
    assert parse_notam(text).a_field == "KZDV KZMP"


def test_field_boundaries_do_not_leak_into_neighboring_field():
    # regression test: field regex must not swallow the next field's marker
    text = "Q) FIR/CODE A) KJFK B) 2501010000 C) 2601010000 E) RWY 09 CLSD"
    p = parse_notam(text)
    assert "B)" not in (p.a_field or "")
    assert "C)" not in (p.b_field or "")
    assert "E)" not in (p.c_field or "")
