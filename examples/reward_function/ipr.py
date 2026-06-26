import re

def extract_tag_content(content: str, tag: str) -> list:
    """
    提取指定标签(tag)之间的内容。

    Args:
        content (str): 输入的字符串（通常是文本）。
        tag (str): 标签名称，例如 'answer' 或 'think'。

    Returns:
        List[str]: 提取的指定标签的内容列表。
    """
    pattern = f"<{tag}>(.*?)</{tag}>"  # 定义正则匹配模式
    return re.findall(pattern, content, re.DOTALL)  # 使用 re.DOTALL 允许跨多行匹配


from typing import List, Dict, Any, Tuple
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
ANS_RE   = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

def _find_tag_spans(s: str, tag: str) -> Tuple[List[Tuple[int,int]], List[str]]:
    pat = THINK_RE if tag == "think" else ANS_RE
    spans, contents = [], []
    for m in pat.finditer(s):
        spans.append((m.start(), m.end()))
        contents.append(m.group(1))
    return spans, contents

def _non_overlapping(a: Tuple[int,int], b: Tuple[int,int]) -> bool:
    return a[1] <= b[0] or b[1] <= a[0]

# === 新增：判断是否包含“实质性思考文本” ===
# 规则：必须出现至少一个字母/数字/中文字符；纯空白、纯标点、纯换行都不算“思考”
_SUBSTANTIVE_RE = re.compile(r"[\w\u4e00-\u9fff]", re.UNICODE)
def _has_substantive_text(s: str) -> bool:
    return bool(_SUBSTANTIVE_RE.search(s.strip()))

def _format_is_strict_single_pair(s: str) -> Tuple[bool, str, str]:
    """
    满足：
      1) 恰好一个 <think>…</think> 且恰好一个 <answer>…</answer>
      2) think 在 answer 之前，且两者不重叠
      3) think 内容须为“实质性思考”（非空白/非纯标点）
      4) 整个字符串必须以 <think> 开头，并以 </answer> 结尾
    返回: (是否合规, think_content, answer_content)
    """
    # 新增：必须以<think>开头，</answer>结尾
    # if not s.strip().startswith("<think>") :#or not s.strip().endswith("</answer>"):
    #     return (False, "", "")

    think_spans, think_contents = _find_tag_spans(s, "think")
    ans_spans, ans_contents     = _find_tag_spans(s, "answer")

    if len(think_spans) != 1 or len(ans_spans) != 1:
        return (False, "", "")

    (ts, te) = think_spans[0]
    (as_, ae) = ans_spans[0]
    if not (te <= as_ and _non_overlapping((ts, te), (as_, ae))):
        return (False, "", "")

    think_txt = think_contents[0]
    if not _has_substantive_text(think_txt):
        return (False, "", "")

    return (True, think_txt, ans_contents[0])

def _extract_clean_answer(ans_text: str) -> str:
    return ans_text.strip().upper()

def compute_score(reward_inputs: List[Dict[str, Any]], format_weight: float = 0.1) -> List[Dict[str, float]]:
    """
    输入: [{"response": str, "ground_truth": str}, ...]
    输出: [{"overall": float, "format": float, "accuracy": float}, ...]
    规则：
      - 格式硬门控：不是严格“一对 think + 一对 answer”或 think 非实质内容 => overall=0
      - 否则 overall = format_weight*format + (1-format_weight)*accuracy
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for this reward function.")

    results = []
    for item in reward_inputs:
        resp = item["response"]
        gt   = item.get("ground_truth", "").strip().upper()

        ok, think_txt, ans_txt = _format_is_strict_single_pair(resp)
        format_score = 1.0 if ok else 0.0

        accuracy = 0.0
        if ok:
            pred = _extract_clean_answer(ans_txt)
            if pred in {"YES", "NO"}:   # 修改这里 ✅
                accuracy = 1.0 if (gt in {"YES", "NO"} and pred == gt) else 0.0
            else:
                ok = False
                format_score = 0.0

        overall = 0.0 if not ok else (format_weight * format_score + (1.0 - format_weight) * accuracy)
        results.append({
            "overall": float(overall),
            "format": float(format_score),
            "accuracy": float(accuracy)
        })
    return results