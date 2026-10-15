from tooth_ai.report_md import region


def test_region_uses_image_width():
    # x-centre 1500 on a 2000-px image is the right third; the old code divided by 1000
    assert region([1450, 0, 1550, 10], 2000).startswith("Image right")
    assert region([950, 0, 1050, 10], 2000) == "Centre"
    assert region([100, 0, 200, 10], 2000).startswith("Image left")
