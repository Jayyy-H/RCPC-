from mathruler.grader import extract_boxed_content

def product_compute_score(predict_str: str, ground_truth: str) -> float:
    answer = extract_boxed_content(predict_str)
    if answer == "None":
        return 0.0  # no answer

    if answer.strip().lower() == ground_truth.strip().lower():
        return 1.0  # correct answer

    return 0.1  # wrong answer
