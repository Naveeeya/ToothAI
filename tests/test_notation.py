from tooth_ai.notation import FDI_TO_UNIVERSAL, UNIVERSAL_TO_FDI, label, tooth_name


def test_round_trip_all_32():
    assert sorted(FDI_TO_UNIVERSAL.values()) == list(range(1, 33))
    for fdi, uni in FDI_TO_UNIVERSAL.items():
        assert UNIVERSAL_TO_FDI[uni] == fdi


def test_known_pairs():
    assert FDI_TO_UNIVERSAL[18] == 1 and FDI_TO_UNIVERSAL[28] == 16
    assert FDI_TO_UNIVERSAL[38] == 17 and FDI_TO_UNIVERSAL[48] == 32
    assert label(36, "Universal") == "19"
    assert tooth_name(36) == "Lower left first molar"
