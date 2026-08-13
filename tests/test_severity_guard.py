from src.severity_guard import guard_severity


def test_conditional_closure_downgraded_from_critical():
    assert guard_severity("critical", "RWY 13/31 CLSD EXC TAX") == "advisory"


def test_ppr_downgraded_from_critical():
    assert guard_severity("critical", "AP CLSD NON-SKED TRANSIENT GA ACFT PPR 617-561-2500") == "advisory"


def test_critical_without_real_trigger_downgraded():
    # regression test for failure_analysis.md Classifier Example 1: route
    # qualification text that merely resembles genuinely unusable navaid text
    text = "ROUTE KZBW. V489 WEARD, NY TO ALBANY (ALB) VORTAC, NY NA EXCEPT FOR AIRCRAFT EQUIPPED WITH SUITABLE RNAV SYSTEM WITH GPS."
    assert guard_severity("critical", text) == "advisory"


def test_unconditional_clsd_stays_critical():
    assert guard_severity("critical", "TWY C3 CLSD") == "critical"


def test_us_stays_critical():
    assert guard_severity("critical", "ILS RWY 27L IM U/S") == "critical"


def test_unusable_stays_critical():
    assert guard_severity("critical", "TDD VOR/DME UNUSABLE.") == "critical"


def test_non_critical_prediction_passes_through_unchanged():
    assert guard_severity("advisory", "RWY 24 SIGNS NOT STD") == "advisory"
    assert guard_severity("informational", "AP SFC COND NOT REP") == "informational"


def test_scheduled_closure_downgraded_via_d_field():
    # regression test for KHPN-07-096: an E-field that reads as an
    # unconditional closure is actually conditional if the D) field encodes
    # a recurring schedule
    assert guard_severity("critical", "RWY 16/34 CLSD", "MON-FRI 0400-0930") == "advisory"


def test_unconditional_clsd_with_no_d_field_stays_critical():
    assert guard_severity("critical", "TWY C3 CLSD", "") == "critical"


def test_sign_light_out_downgraded_from_critical():
    # a sign's illumination being out is a degraded marking, not the
    # fixture itself being closed -- regression test for KJFK-03-408 /
    # KHPN-07-109 (see failure_analysis.md, "Zero-shot LLM comparison")
    assert guard_severity("critical", "TWY B TWY DIRECTION SIGN FOR TWY YA LGT U/S") == "advisory"
    assert guard_severity("critical", "RWY 16 RWY EXIT SIGN AT TWY L LGT U/S") == "advisory"


def test_direct_fixture_us_without_sign_stays_critical():
    assert guard_severity("critical", "RWY 36 PAPI U/S") == "critical"
