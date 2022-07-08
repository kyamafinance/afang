from afang.main import calc_addition, calc_mul


def test_calc_addition():
    output = calc_addition(2, 4)
    assert output == 6


def test_calc_mul():
    output = calc_mul(2, 4)
    assert output == 7
