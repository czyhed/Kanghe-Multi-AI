"""
急症预检：每轮用户消息在路由之前先过一遍。

用确定性的关键词规则，不依赖 LLM 是否遵守提示词。
宁可多提醒，也不漏掉；命中后只给出就医指引，不做诊断。
"""

EMERGENCY_KEYWORDS = (
    "胸痛", "胸口痛", "胸口疼", "胸闷痛", "心梗", "心脏骤停",
    "呼吸困难", "喘不上气", "喘不过气", "无法呼吸", "窒息",
    "意识不清", "意识模糊", "昏迷", "晕倒", "昏倒", "不省人事",
    "抽搐", "癫痫发作",
    "大出血", "咯血", "咳血", "呕血", "便血不止",
    "中风", "半身不遂", "口角歪斜", "说话不清",
    "过敏性休克", "喉头水肿",
    "自杀", "不想活", "服毒", "吞药", "割腕",
    "中毒",
)

# 关键词前出现这些词，视为否定（"没有胸痛"）
_NEGATIONS = ("没有", "没", "无", "不是", "不", "未", "否认")
_NEGATION_WINDOW = 3

EMERGENCY_REPLY = (
    "您描述的情况可能比较紧急，请不要等待线上问诊：\n\n"
    "1. 立即拨打 120，或让身边的人尽快送您前往最近医院的急诊。\n"
    "2. 等待救援时保持安静，不要独自行动，不要自行服药。\n"
    "3. 如果是他人出现上述情况，请保持其呼吸道通畅并陪同，等待救护人员。\n\n"
    "线上问诊暂停。平安后需要，我再继续帮您。"
)


def _is_negated(text: str, start: int) -> bool:
    prefix = text[max(0, start - _NEGATION_WINDOW):start]
    return any(neg in prefix for neg in _NEGATIONS)


def detect_emergency(text: str) -> str | None:
    """命中急症信号时返回命中的关键词，否则返回 None。"""
    for kw in EMERGENCY_KEYWORDS:
        start = text.find(kw)
        while start != -1:
            if not _is_negated(text, start):
                return kw
            start = text.find(kw, start + 1)
    return None
