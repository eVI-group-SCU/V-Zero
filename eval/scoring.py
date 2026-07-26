from __future__ import annotations

import re
import string
import unicodedata

from .schema import EvalExample


# Common suffix after a captured option letter: closing bracket, punctuation, end tag/end
# of answer, or a short reason marker. This keeps "Answer: B" valid without treating
# the article in "answer is a dog" as option A.
LETTER_CAPTURE_SUFFIX = r"[*_`]*(?:\s*[\)\]]|[).:：、。．]|(?=\s*(?:$|<))|(?=\s+(?:because|since|as)\b))"

# Matches XML-style answer tags, e.g. "<answer>B" or "<answer>(B)</answer>".
ANSWER_TAG_RE = re.compile(
    rf"<answer>\s*[*_`]*[\(\[]?\s*([A-D]){LETTER_CAPTURE_SUFFIX}",
    re.IGNORECASE,
)
ANSWER_TAG_CONTENT_RE = re.compile(
    r"<answer>\s*(.*?)(?:</answer>|$)",
    re.IGNORECASE | re.DOTALL,
)

# Matches explicit English answer phrases, e.g. "Answer: B." or "final answer is (B)".
ANSWER_PHRASE_RE = re.compile(
    rf"\b(?:correct\s+|final\s+)?answer\s*(?:is|=)?\s*[:：]?\s*"
    rf"[*_`]*[\(\[]?\s*([A-D]){LETTER_CAPTURE_SUFFIX}",
    re.IGNORECASE,
)

# Matches explicit English choice phrases, e.g. "I choose B", "selected: C", or "go with D".
OPTION_PHRASE_RE = re.compile(
    rf"\b(?:option|choice|letter|choose|chose|select|selected|pick|picked|go\s+with)"
    rf"\s*(?:is|=)?\s*[:：]?\s*[*_`]*[\(\[]?\s*([A-D]){LETTER_CAPTURE_SUFFIX}",
    re.IGNORECASE,
)

# Matches explicit Chinese answer phrases, e.g. "答案：B", "正确答案是 C", or "我选 D".
ZH_ANSWER_PHRASE_RE = re.compile(
    rf"(?:最佳答案|最終答案|最终答案|正確答案|正确答案|正確選項|正确选项|最佳選項|最佳选项|答案|選項|选项|"
    rf"选择|選擇|我选择|我選擇|我选|我選|故选|故選|应选|應選)"
    rf"\s*(?:是|为|為|选|選|=|的是)?\s*[:：]?\s*[*_`]*[\(\[]?\s*([A-D]){LETTER_CAPTURE_SUFFIX}",
    re.IGNORECASE,
)

# Matches responses that contain only the option letter, e.g. "B", "(C)", or "D.".
LEADING_SINGLE_LETTER_RE = re.compile(
    r"^\s*[*_`]*[\(\[]?\s*([A-D])\s*[\)\]]?\s*[*_`]*[.!。．]?\s*$",
    re.IGNORECASE,
)

# Matches an option label at the start of an answer, e.g. "B. blue", "(C) green", or "D、black".
LEADING_OPTION_LABEL_RE = re.compile(
    r"^\s*[*_`]*[\(\[]?\s*([A-D])\s*(?:[\)\]]|[.:：、])",
    re.IGNORECASE,
)

# Matches concise answers that start with a choice and explanation, e.g. "B because ...".
LEADING_REASON_RE = re.compile(
    r"^\s*[*_`]*([A-D])\s+(?:because|since|as)\b",
    re.IGNORECASE,
)

# Conservative fallback for explicit English choice mentions, e.g. "selected option B.".
CONTEXT_LETTER_RE = re.compile(
    rf"\b(?:answer|option|choice|letter|choose|chose|select|selected|pick|picked|go\s+with)"
    rf"\s*(?:is|=)?\s*[:：]?\s*[*_`]*[\(\[]?\s*([A-D]){LETTER_CAPTURE_SUFFIX}",
    re.IGNORECASE,
)

# Some MCQ datasets use five or six choices. Keep the default parser constrained to
# A-D, and use these wider A-F patterns only when the sample actually has extended
# choices.
MME_ANSWER_TAG_RE = re.compile(
    rf"<answer>\s*[*_`]*[\(\[]?\s*([A-F]){LETTER_CAPTURE_SUFFIX}",
    re.IGNORECASE,
)
MME_ANSWER_PHRASE_RE = re.compile(
    rf"\b(?:correct\s+|final\s+)?answer\s*(?:is|=)?\s*[:：]?\s*"
    rf"[*_`]*[\(\[]?\s*([A-F]){LETTER_CAPTURE_SUFFIX}",
    re.IGNORECASE,
)
MME_OPTION_PHRASE_RE = re.compile(
    rf"\b(?:option|choice|letter|choose|chose|select|selected|pick|picked|go\s+with)"
    rf"\s*(?:is|=)?\s*[:：]?\s*[*_`]*[\(\[]?\s*([A-F]){LETTER_CAPTURE_SUFFIX}",
    re.IGNORECASE,
)
MME_ZH_ANSWER_PHRASE_RE = re.compile(
    rf"(?:最佳答案|最終答案|最终答案|正確答案|正确答案|正確選項|正确选项|最佳選項|最佳选项|答案|選項|选项|"
    rf"选择|選擇|我选择|我選擇|我选|我選|故选|故選|应选|應選)"
    rf"\s*(?:是|为|為|选|選|=|的是)?\s*[:：]?\s*[*_`]*[\(\[]?\s*([A-F]){LETTER_CAPTURE_SUFFIX}",
    re.IGNORECASE,
)
MME_LEADING_SINGLE_LETTER_RE = re.compile(
    r"^\s*[*_`]*[\(\[]?\s*([A-F])\s*[\)\]]?\s*[*_`]*[.!。．]?\s*$",
    re.IGNORECASE,
)
MME_LEADING_OPTION_LABEL_RE = re.compile(
    r"^\s*[*_`]*[\(\[]?\s*([A-F])\s*(?:[\)\]]|[.:：、])",
    re.IGNORECASE,
)
MME_LEADING_REASON_RE = re.compile(
    r"^\s*[*_`]*([A-F])\s+(?:because|since|as)\b",
    re.IGNORECASE,
)
MME_CONTEXT_LETTER_RE = re.compile(
    rf"\b(?:answer|option|choice|letter|choose|chose|select|selected|pick|picked|go\s+with)"
    rf"\s*(?:is|=)?\s*[:：]?\s*[*_`]*[\(\[]?\s*([A-F]){LETTER_CAPTURE_SUFFIX}",
    re.IGNORECASE,
)

# Parses choices embedded in prompts, e.g. "(A) red", "B. blue".
CHOICE_LINE_RE = re.compile(r"^\s*[\(\[]?([A-F])[\)\].]\s*(.+?)\s*$", re.IGNORECASE)
POPE_EXPLICIT_ANSWER_RE = re.compile(
    r"\b(?:final\s+answer|answer)\s*(?:is|=)?\s*[:：]?\s*(yes|no)\b",
    re.IGNORECASE,
)
POPE_SINGLE_ANSWER_RE = re.compile(
    r"^\s*[*_`]*(yes|no)[*_`]*\s*[.!。．]?\s*$",
    re.IGNORECASE,
)
POPE_NEGATIVE_TOKEN_RE = re.compile(r"\b(?:no|not)\b", re.IGNORECASE)


def choice_map_from_prompt(prompt: str) -> dict[str, str]:
    choices: dict[str, str] = {}
    for line in prompt.splitlines():
        match = CHOICE_LINE_RE.match(line)
        if match:
            choices[match.group(1).upper()] = match.group(2).strip()
    return choices


def parse_mcq_prediction(text: str, choices: dict[str, str] | None = None) -> str | None:
    return _parse_mcq_prediction_with_patterns(
        text,
        choices,
        answer_tag_re=ANSWER_TAG_RE,
        answer_phrase_re=ANSWER_PHRASE_RE,
        option_phrase_re=OPTION_PHRASE_RE,
        zh_answer_phrase_re=ZH_ANSWER_PHRASE_RE,
        leading_single_letter_re=LEADING_SINGLE_LETTER_RE,
        leading_option_label_re=LEADING_OPTION_LABEL_RE,
        leading_reason_re=LEADING_REASON_RE,
        context_letter_re=CONTEXT_LETTER_RE,
    )


def parse_mme_realworld_prediction(text: str, choices: dict[str, str] | None = None) -> str | None:
    return _parse_mcq_prediction_with_patterns(
        text,
        choices,
        answer_tag_re=MME_ANSWER_TAG_RE,
        answer_phrase_re=MME_ANSWER_PHRASE_RE,
        option_phrase_re=MME_OPTION_PHRASE_RE,
        zh_answer_phrase_re=MME_ZH_ANSWER_PHRASE_RE,
        leading_single_letter_re=MME_LEADING_SINGLE_LETTER_RE,
        leading_option_label_re=MME_LEADING_OPTION_LABEL_RE,
        leading_reason_re=MME_LEADING_REASON_RE,
        context_letter_re=MME_CONTEXT_LETTER_RE,
    )


def _parse_mcq_prediction_with_patterns(
    text: str,
    choices: dict[str, str] | None,
    *,
    answer_tag_re: re.Pattern[str],
    answer_phrase_re: re.Pattern[str],
    option_phrase_re: re.Pattern[str],
    zh_answer_phrase_re: re.Pattern[str],
    leading_single_letter_re: re.Pattern[str],
    leading_option_label_re: re.Pattern[str],
    leading_reason_re: re.Pattern[str],
    context_letter_re: re.Pattern[str],
) -> str | None:
    text = unicodedata.normalize("NFKC", text)
    tag_match = ANSWER_TAG_CONTENT_RE.search(text)
    if tag_match:
        tag_text = tag_match.group(1)
        for pattern in (
            answer_phrase_re,
            option_phrase_re,
            zh_answer_phrase_re,
            leading_single_letter_re,
            leading_option_label_re,
            leading_reason_re,
        ):
            match = pattern.search(tag_text)
            if match:
                return match.group(1).upper()
        match = answer_tag_re.search(text)
        if match:
            return match.group(1).upper()

    for pattern in (answer_phrase_re, option_phrase_re, zh_answer_phrase_re):
        match = pattern.search(text)
        if match:
            return match.group(1).upper()

    stripped = text.strip()
    if stripped:
        for pattern in (leading_single_letter_re, leading_option_label_re, leading_reason_re):
            match = pattern.match(stripped)
            if match:
                return match.group(1).upper()

    if choices:
        matched = match_choice_text(stripped, choices)
        if matched:
            return matched

    context_matches = {match.group(1).upper() for match in context_letter_re.finditer(text)}
    if len(context_matches) == 1:
        return next(iter(context_matches))
    return None


def match_choice_text(text: str, choices: dict[str, str]) -> str | None:
    normalized_response = normalize_open_answer(text)
    exact_matches = [
        label
        for label, option in choices.items()
        if normalize_open_answer(option) == normalized_response
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]

    return None


def expected_mcq_letter(answer: str, choices: dict[str, str]) -> str:
    parsed = (
        parse_mme_realworld_prediction(answer, choices)
        if _has_extended_choices(choices)
        else parse_mcq_prediction(answer, choices)
    )
    if parsed and parsed in _choice_labels(choices):
        return parsed
    normalized_answer = normalize_open_answer(answer)
    matches = [
        label
        for label, option in choices.items()
        if normalize_open_answer(option) == normalized_answer
    ]
    if len(matches) == 1:
        return matches[0]
    return answer.strip().upper()


def normalize_open_answer(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.strip().lower()
    marker = "final answer:"
    marker_index = text.rfind(marker)
    if marker_index != -1:
        text = text[marker_index + len(marker) :].strip()
    text = re.sub(r"^\s*(?:the\s+)?answer\s*(?:is|:)?\s*", "", text)
    text = text.strip().strip(string.punctuation + " \t\r\n")
    text = re.sub(r"\s+", " ", text)
    return text


def parse_pope_prediction(text: str) -> str:
    """Extract a POPE yes/no answer using the official first-sentence rule.

    Explicit ``<answer>`` and final-answer markers take precedence so thinking-mode
    reasoning does not override the model's final response. The fallback mirrors the
    official POPE evaluator: a first sentence containing ``no`` or ``not`` is negative;
    every other first sentence is positive.
    """
    normalized = unicodedata.normalize("NFKC", text).strip()
    tag_match = ANSWER_TAG_CONTENT_RE.search(normalized)
    if tag_match:
        tag_prediction = _parse_explicit_pope_answer(tag_match.group(1))
        if tag_prediction is not None:
            return tag_prediction

    explicit_matches = list(POPE_EXPLICIT_ANSWER_RE.finditer(normalized))
    if explicit_matches:
        return explicit_matches[-1].group(1).lower()

    first_sentence = re.split(r"[.!?。！？]", normalized, maxsplit=1)[0]
    direct_match = POPE_SINGLE_ANSWER_RE.match(first_sentence)
    if direct_match:
        return direct_match.group(1).lower()
    if POPE_NEGATIVE_TOKEN_RE.search(first_sentence.replace(",", " ")):
        return "no"
    return "yes"


def _parse_explicit_pope_answer(text: str) -> str | None:
    direct_match = POPE_SINGLE_ANSWER_RE.match(text)
    if direct_match:
        return direct_match.group(1).lower()
    explicit_matches = list(POPE_EXPLICIT_ANSWER_RE.finditer(text))
    if explicit_matches:
        return explicit_matches[-1].group(1).lower()
    return None


def score_response(example: EvalExample, response: str) -> tuple[str | None, bool]:
    if example.dataset == "pope":
        prediction = parse_pope_prediction(response)
        expected = example.answer.strip().lower()
        return prediction, prediction == expected

    if example.choices:
        expected = expected_mcq_letter(example.answer, example.choices)
        prediction = parse_prediction_for_choices(response, example.choices)
        return prediction, prediction == expected

    prediction = normalize_open_answer(response)
    expected = normalize_open_answer(example.answer)
    return prediction or None, bool(prediction and prediction == expected)


def _has_extended_choices(choices: dict[str, str]) -> bool:
    return any(label.upper() not in {"A", "B", "C", "D"} for label in choices)


def parse_prediction_for_choices(text: str, choices: dict[str, str]) -> str | None:
    prediction = (
        parse_mme_realworld_prediction(text, choices)
        if _has_extended_choices(choices)
        else parse_mcq_prediction(text, choices)
    )
    if prediction and prediction in _choice_labels(choices):
        return prediction
    return None


def _choice_labels(choices: dict[str, str]) -> set[str]:
    return {label.upper() for label in choices}
