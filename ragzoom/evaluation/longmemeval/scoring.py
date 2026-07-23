"""Scoring for the LongMemEval benchmark: type-specific LLM-as-Judge.

LongMemEval grades each answer with a prompt tailored to its question type —
temporal questions tolerate off-by-one day errors, knowledge-update questions
accept answers that carry both old and new facts (as long as the new one is
present), preference questions grade against a rubric, and abstention questions
check that the model declined to answer. The judge returns "yes" or "no".

The judge prompts are taken verbatim from the official LongMemEval evaluation
(``src/evaluation/evaluate_qa.py``) so our numbers are comparable to the
published leaderboard. Unlike the upstream reference adapter, this judge follows
the RagZoom "no fallback code" rule: a judge that cannot produce a verdict after
its retries raises ``LLMError`` rather than silently scoring the answer "no".
"""

from __future__ import annotations

import re

from ragzoom.agent.protocol import BenchmarkingAgent
from ragzoom.evaluation.benchmark_common import judge_with_retry
from ragzoom.evaluation.longmemeval.types import JudgeVerdict, QuestionType

# ---------------------------------------------------------------------------
# Type-specific judge prompts (verbatim from the official LongMemEval eval)
# ---------------------------------------------------------------------------

_STANDARD_TEMPLATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains "
    "all the intermediate steps to get the correct answer, you should also answer "
    "yes. If the response only contains a subset of the information required by "
    "the answer, answer no. \n\n"
    "Question: {question}\n\n"
    "Correct Answer: {answer}\n\n"
    "Model Response: {response}\n\n"
    "Is the model response correct? Answer yes or no only."
)

_TEMPORAL_TEMPLATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains "
    "all the intermediate steps to get the correct answer, you should also answer "
    "yes. If the response only contains a subset of the information required by "
    "the answer, answer no. In addition, do not penalize off-by-one errors for "
    "the number of days. If the question asks for the number of days/weeks/months, "
    "etc., and the model makes off-by-one errors (e.g., predicting 19 days when "
    "the answer is 18), the model's response is still correct. \n\n"
    "Question: {question}\n\n"
    "Correct Answer: {answer}\n\n"
    "Model Response: {response}\n\n"
    "Is the model response correct? Answer yes or no only."
)

_KNOWLEDGE_UPDATE_TEMPLATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response contains some previous information along with an "
    "updated answer, the response should be considered as correct as long as the "
    "updated answer is the required answer.\n\n"
    "Question: {question}\n\n"
    "Correct Answer: {answer}\n\n"
    "Model Response: {response}\n\n"
    "Is the model response correct? Answer yes or no only."
)

_PREFERENCE_TEMPLATE = (
    "I will give you a question, a rubric for desired personalized response, and "
    "a response from a model. Please answer yes if the response satisfies the "
    "desired response. Otherwise, answer no. The model does not need to reflect "
    "all the points in the rubric. The response is correct as long as it recalls "
    "and utilizes the user's personal information correctly.\n\n"
    "Question: {question}\n\n"
    "Rubric: {answer}\n\n"
    "Model Response: {response}\n\n"
    "Is the model response correct? Answer yes or no only."
)

_ABSTENTION_TEMPLATE = (
    "I will give you an unanswerable question, an explanation, and a response "
    "from a model. Please answer yes if the model correctly identifies the "
    "question as unanswerable. The model could say that the information is "
    "incomplete, or some other information is given but the asked information "
    "is not.\n\n"
    "Question: {question}\n\n"
    "Explanation: {answer}\n\n"
    "Model Response: {response}\n\n"
    "Does the model correctly identify the question as unanswerable? "
    "Answer yes or no only."
)

# Question types that use the plain "does it contain the answer" rubric.
_STANDARD_TYPES = frozenset(
    {
        QuestionType.SINGLE_SESSION_USER,
        QuestionType.SINGLE_SESSION_ASSISTANT,
        QuestionType.MULTI_SESSION,
    }
)


def build_judge_prompt(
    question_type: QuestionType,
    question: str,
    answer: str,
    response: str,
    *,
    is_abstention: bool = False,
) -> str:
    """Select and fill the appropriate judge prompt for a question.

    Abstention takes precedence over type: an unanswerable question is graded
    on whether the model declined, regardless of its nominal type. Every
    ``QuestionType`` maps to exactly one template — there is no fallback branch,
    so adding a new type without a template is a hard error at the match below.
    """
    if is_abstention:
        return _ABSTENTION_TEMPLATE.format(
            question=question, answer=answer, response=response
        )

    if question_type in _STANDARD_TYPES:
        template = _STANDARD_TEMPLATE
    elif question_type == QuestionType.TEMPORAL_REASONING:
        template = _TEMPORAL_TEMPLATE
    elif question_type == QuestionType.KNOWLEDGE_UPDATE:
        template = _KNOWLEDGE_UPDATE_TEMPLATE
    elif question_type == QuestionType.SINGLE_SESSION_PREFERENCE:
        template = _PREFERENCE_TEMPLATE
    else:  # pragma: no cover - exhaustive over the enum; guards future additions
        raise ValueError(f"No judge template for question type {question_type!r}")

    return template.format(question=question, answer=answer, response=response)


_YES_NO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def _parse_yes_no(text: str) -> JudgeVerdict:
    """Extract a yes/no verdict from the judge's raw answer, or raise."""
    match = _YES_NO_RE.search(text)
    if match is None:
        raise ValueError(f"No yes/no verdict found in: {text!r}")
    return "yes" if match.group(1).lower() == "yes" else "no"


async def judge_answer(
    backend: BenchmarkingAgent,
    question_type: QuestionType,
    question: str,
    gold_answer: str,
    generated_answer: str,
    *,
    is_abstention: bool = False,
    model_id: str = "unknown",
    max_retries: int = 3,
) -> JudgeVerdict:
    """Type-specific LLM-as-Judge. Returns "yes" (correct) or "no" (incorrect).

    Raises:
        LLMError: if the judge fails or returns no parseable verdict after all
            retries. We fail hard rather than defaulting to "no" — a silently
            mis-scored abstention question would corrupt the safety tripwire.
    """
    prompt = build_judge_prompt(
        question_type,
        question,
        gold_answer,
        generated_answer,
        is_abstention=is_abstention,
    )
    return await judge_with_retry(
        backend,
        "You are a careful grader.",
        prompt,
        _parse_yes_no,
        operation="longmemeval_judge",
        model_id=model_id,
        max_retries=max_retries,
    )
