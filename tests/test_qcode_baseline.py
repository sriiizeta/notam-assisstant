from src.qcode_baseline import QCodeLookupClassifier, qcode_subject, qcode_condition


def test_qcode_subject_and_condition_extraction():
    assert qcode_subject("QMRLC") == "MR"
    assert qcode_condition("QMRLC") == "LC"
    assert qcode_subject("") == ""
    assert qcode_condition("QMR") == ""  # too short for a condition


def test_lookup_predicts_training_majority_per_group():
    q = ["QMRLC", "QMRLC", "QMXLC", "QOLAS"]
    y = ["runway", "runway", "taxiway", "lighting"]
    clf = QCodeLookupClassifier(key="subject").fit(q, y)
    assert clf.predict(["QMRLC"]) == ["runway"]
    assert clf.predict(["QMXLC"]) == ["taxiway"]
    assert clf.predict(["QOLAS"]) == ["lighting"]


def test_unseen_group_falls_back_to_global_majority():
    q = ["QMRLC", "QMRLC", "QMXLC"]
    y = ["runway", "runway", "taxiway"]
    clf = QCodeLookupClassifier(key="subject").fit(q, y)
    # QPIXX subject "PI" never seen in training -> global majority "runway"
    assert clf.predict(["QPIXX"]) == ["runway"]


def test_condition_key_uses_letters_4_5():
    q = ["QMRLC", "QMRAS"]
    y = ["critical", "advisory"]
    clf = QCodeLookupClassifier(key="condition").fit(q, y)
    assert clf.predict(["QMRLC"]) == ["critical"]   # LC
    assert clf.predict(["QMRAS"]) == ["advisory"]   # AS
