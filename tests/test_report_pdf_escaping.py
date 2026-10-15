"""User text never becomes PDF markup: an <img> tag as the Case ID must not be interpreted."""

import pytest

from tooth_ai.demo import analyze_sample
from tooth_ai.report_pdf import build_pdf, safe_case_id

pytest.importorskip("reportlab")

HOSTILE = '<img src="/nonexistent/secret.png" width="40" height="40"/>'


@pytest.mark.parametrize("raw, expected", [
    (HOSTILE, "img src"),
    ("Case 12_b.3-x", "Case 12_b.3-x"),
    ("A&B<script>", "ABscript"),
    (None, ""),
    ("x" * 80, "x" * 40),
])
def test_case_id_allow_list(raw, expected):
    assert safe_case_id(raw).startswith(expected)
    assert not set(safe_case_id(raw)) & set("<>&\"'/=")


def test_pdf_builds_with_markup_in_every_user_field():
    import numpy as np

    result = analyze_sample("sample_02")
    result.reliability.reasons.append(HOSTILE + " & </para>")
    result.reliability.detail = "<b>bold?</b> " + HOSTILE
    pdf = build_pdf(result, np.zeros((300, 600, 3), np.uint8), case_id=HOSTILE)
    # An interpreted <img> pointing at a missing file makes ReportLab raise; escaped text renders.
    assert pdf.startswith(b"%PDF")
