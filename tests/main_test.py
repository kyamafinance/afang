from afang.main import calc_addition


def test_calc_addition():
    output = calc_addition(2, 4)
    assert output == 6
