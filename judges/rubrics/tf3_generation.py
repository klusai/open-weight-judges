"""TF3 Romanian native fable generation rubric.

5 dimensions, 1-5 scale.
Derived from TF3 paper evaluation criteria (Gemini 2.5 Flash baseline).
"""

from judges.rubrics.base import Rubric

SYSTEM_PROMPT = (
    "You are an expert literary critic specializing in Romanian fables and moral "
    "tales. Your evaluations should be objective, consistent, and based on "
    "established literary and linguistic standards for Romanian. Provide your "
    "assessment in valid, properly-formatted JSON only. Do not include any text "
    "outside the JSON object. Balance critical analysis with constructive "
    "feedback, focusing on both strengths and weaknesses."
)

EVALUATION_PROMPT = """\
Evaluate the following Romanian fable according to these criteria. For each \
dimension, assign a score from 1 to 5 (integer only, 5 = excellent, 1 = very poor) \
and give a brief one-sentence justification.

1. **Grammar & Language Quality** (1-5):
   Evaluate Romanian grammar, syntax, spelling, diacritics usage, and overall \
readability. Consider morphological correctness and natural phrasing.
   5 = Perfect Romanian; 4 = Minor errors; 3 = Some errors distract; \
2 = Frequent mistakes; 1 = Largely ungrammatical.

2. **Creativity & Originality** (1-5):
   Does the fable offer a fresh perspective or novel narrative elements while \
maintaining the classic fable structure?
   5 = Highly original; 4 = Some fresh elements; 3 = Follows familiar patterns; \
2 = Predictable; 1 = Derivative or cliched.

3. **Coherence** (1-5):
   Does the story flow logically from beginning to end? Are characters, events, \
and the narrative arc consistent?
   5 = Fully coherent; 4 = Slight lapse; 3 = Some gaps; 2 = Notably disjointed; \
1 = Incoherent.

4. **Moral Clarity** (1-5):
   Is the moral lesson clear, meaningful, and well-integrated into the narrative?
   5 = Clear and impactful; 4 = Present but could be stronger; 3 = Underdeveloped; \
2 = Confused; 1 = Absent or contradictory.

5. **Prompt Adherence** (1-5):
   Does the fable incorporate all required elements from the prompt (character, \
trait, setting, conflict, resolution, moral)?
   5 = All elements addressed; 4 = Most elements present; 3 = Some missing; \
2 = Multiple missing; 1 = Largely ignores prompt.

Format your response as valid JSON:
{
    "grammar": {"score": <1-5>, "justification": "<one sentence>"},
    "creativity": {"score": <1-5>, "justification": "<one sentence>"},
    "coherence": {"score": <1-5>, "justification": "<one sentence>"},
    "moral_clarity": {"score": <1-5>, "justification": "<one sentence>"},
    "adherence": {"score": <1-5>, "justification": "<one sentence>"}
}

Prompt:
{{prompt}}

Romanian fable:
{{fable}}"""

JSON_SCHEMA = {
    "name": "romanian_fable_evaluation",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "grammar": {
                "type": "object",
                "properties": {
                    "score": {"type": "integer"},
                    "justification": {"type": "string"},
                },
                "required": ["score", "justification"],
            },
            "creativity": {
                "type": "object",
                "properties": {
                    "score": {"type": "integer"},
                    "justification": {"type": "string"},
                },
                "required": ["score", "justification"],
            },
            "coherence": {
                "type": "object",
                "properties": {
                    "score": {"type": "integer"},
                    "justification": {"type": "string"},
                },
                "required": ["score", "justification"],
            },
            "moral_clarity": {
                "type": "object",
                "properties": {
                    "score": {"type": "integer"},
                    "justification": {"type": "string"},
                },
                "required": ["score", "justification"],
            },
            "adherence": {
                "type": "object",
                "properties": {
                    "score": {"type": "integer"},
                    "justification": {"type": "string"},
                },
                "required": ["score", "justification"],
            },
        },
        "required": ["grammar", "creativity", "coherence", "moral_clarity", "adherence"],
    },
}

SCORE_FIELDS = ["grammar", "creativity", "coherence", "moral_clarity", "adherence"]


def extract_scores(response: dict) -> dict[str, int]:
    """Extract flat score dict from nested TF3 response format."""
    return {dim: response[dim]["score"] for dim in SCORE_FIELDS if dim in response}


RUBRIC = Rubric(
    task_name="tf3_generation",
    system_prompt=SYSTEM_PROMPT,
    evaluation_prompt_template=EVALUATION_PROMPT,
    json_schema=JSON_SCHEMA,
    score_fields=SCORE_FIELDS,
    score_range=(1, 5),
    text_field_map={"prompt": "prompt", "fable": "fable"},
    extra_fields=[],
)
