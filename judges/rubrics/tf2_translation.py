"""TF2 English-Romanian literary translation rubric.

5 dimensions, 1-5 scale (MQM-style).
Source: tinyfabulist-tf2/lib/conf/evaluator.yaml (system_prompt_ro + user_prompt_ro)
"""

from judges.rubrics.base import Rubric

SYSTEM_PROMPT = """\
You are a professional translation evaluator tasked with assessing the quality \
of Romanian translations of English fables. Follow state-of-the-art translation \
evaluation standards to ensure your judgments align with expert human evaluators.

Evaluation Procedure: Carefully read the English source text and the Romanian \
translated text provided. Analyze the translation across multiple dimensions \
of quality, considering both sentence-level details and the overall passage. \
Focus on how accurately and eloquently the translation conveys the fable's \
meaning, style, and effect.

Provide the evaluation results in JSON format only. Do not include any text \
outside the JSON structure. Remain objective and consistent in your evaluations, \
as a human evaluator would."""

EVALUATION_PROMPT = """\
Evaluate the following English-to-Romanian translation of a fable according to \
these criteria. For each dimension, assign a score from 1 to 5 (integer only, \
5 = excellent, 1 = very poor) and give a brief one-sentence justification.

1. **Accuracy (Meaning Preservation)** (1-5):
   Does the Romanian translation faithfully convey the same meaning as the \
English source? Check for mistranslations, omissions, or additions.
   5 = All meaning preserved; 4 = Minor nuances off; 3 = One significant error \
but main idea understandable; 2 = Multiple serious errors; 1 = Major parts lost.

2. **Fluency (Language Quality in Romanian)** (1-5):
   How natural and well-formed is the translation in Romanian? Evaluate grammar, \
syntax, spelling, and general readability.
   5 = Reads like native Romanian; 4 = Minor errors; 3 = Some errors distract; \
2 = Frequent mistakes; 1 = Incomprehensible in parts.

3. **Coherence and Consistency** (1-5):
   Does the translated story flow logically? Are pronouns, references, and tenses \
consistent? Is terminology uniform throughout?
   5 = Fully cohesive; 4 = Slight lapse; 3 = Some coherence issues; \
2 = Notably incoherent; 1 = Very confusing.

4. **Style and Narrative Voice** (1-5):
   Does the translation maintain the style, tone, and voice of the original fable?
   5 = Style matches excellently; 4 = Minor differences; 3 = Noticeable divergence; \
2 = Significant mismatch; 1 = Completely inappropriate.

5. **Cultural and Pragmatic Fidelity** (1-5):
   Does the translation preserve implied meanings, morals, and cultural references? \
Are idioms appropriately adapted?
   5 = All nuances conveyed; 4 = Slight loss; 3 = Some loss of meaning; \
2 = Important elements lost; 1 = Pragmatic failure.

Format your response as valid JSON:
{
    "accuracy": {"score": <1-5>, "justification": "<one sentence>"},
    "fluency": {"score": <1-5>, "justification": "<one sentence>"},
    "coherence": {"score": <1-5>, "justification": "<one sentence>"},
    "style": {"score": <1-5>, "justification": "<one sentence>"},
    "cultural_pragmatic": {"score": <1-5>, "justification": "<one sentence>"}
}

Original fable:
{{original_fable}}

Translation (to Romanian):
{{translated_fable}}"""

JSON_SCHEMA = {
    "name": "translation_evaluation",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "accuracy": {
                "type": "object",
                "properties": {
                    "score": {"type": "integer"},
                    "justification": {"type": "string"},
                },
                "required": ["score", "justification"],
            },
            "fluency": {
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
            "style": {
                "type": "object",
                "properties": {
                    "score": {"type": "integer"},
                    "justification": {"type": "string"},
                },
                "required": ["score", "justification"],
            },
            "cultural_pragmatic": {
                "type": "object",
                "properties": {
                    "score": {"type": "integer"},
                    "justification": {"type": "string"},
                },
                "required": ["score", "justification"],
            },
        },
        "required": ["accuracy", "fluency", "coherence", "style", "cultural_pragmatic"],
    },
}

# For TF2, scores are nested inside dimension objects.
# The validation layer needs to extract {"accuracy": obj["accuracy"]["score"], ...}
SCORE_FIELDS = ["accuracy", "fluency", "coherence", "style", "cultural_pragmatic"]


def extract_scores(response: dict) -> dict[str, int]:
    """Extract flat score dict from nested TF2 response format."""
    return {dim: response[dim]["score"] for dim in SCORE_FIELDS if dim in response}


RUBRIC = Rubric(
    task_name="tf2_translation",
    system_prompt=SYSTEM_PROMPT,
    evaluation_prompt_template=EVALUATION_PROMPT,
    json_schema=JSON_SCHEMA,
    score_fields=SCORE_FIELDS,
    score_range=(1, 5),
    text_field_map={
        "original_fable": "fable",
        "translated_fable": "translated_fable",
    },
    extra_fields=[],
)
