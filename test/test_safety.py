"""急症预检（纯函数）"""
import pytest

from src.agents.safety import detect_emergency


@pytest.mark.parametrize("text", [
    "我胸痛得厉害",
    "我爸突然昏迷了",
    "孩子呼吸困难",
    "我不想活了",
])
def test_emergency_is_detected(text):
    assert detect_emergency(text) is not None


@pytest.mark.parametrize("text", [
    "感冒两天了",
    "没有胸痛，就是头疼",
    "没有呼吸困难，只是有点咳嗽",
    "你好",
])
def test_normal_or_negated_is_not_flagged(text):
    assert detect_emergency(text) is None


def test_negation_in_one_place_does_not_hide_real_symptom():
    # 前半句否认头晕，后半句确实胸痛
    assert detect_emergency("没有头晕，但是胸痛很厉害") is not None
