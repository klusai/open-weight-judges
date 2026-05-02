"""TF1 English fable generation rubric.

4 dimensions, 1-10 scale, plus age group classification.
Source: tinyfabulist-tf1/tinyfabulist/conf/evaluator.yaml
"""

from judges.rubrics.base import Rubric

SYSTEM_PROMPT = (
    "You are an expert literary critic specializing in fables and moral tales. "
    "Your evaluations should be objective, consistent, and based on established "
    "literary standards. Age-appropriateness is a key consideration in your "
    "assessment. Provide your assessment in valid, properly-formatted JSON only. "
    "Do not include any text outside the JSON object. Your response must be "
    "parseable by a JSON parser with no preprocessing. Balance critical analysis "
    "with constructive feedback, focusing on both strengths and weaknesses."
)

EVALUATION_PROMPT = """\
Evaluate the following fable according to these specific criteria:

1. **Grammar & Style (1-10)**:
   • 1-3: Significant errors that impede understanding
   • 4-6: Some errors but generally readable
   • 7-10: Clean, polished writing with appropriate language and style for a fable

2. **Creativity & Originality (1-10)**:
   • 1-3: Derivative, predictable, or clichéd
   • 4-6: Contains some original elements but follows familiar patterns
   • 7-10: Fresh perspective, innovative approach while maintaining classic fable structure

3. **Moral Clarity (1-10)**:
   • 1-3: Moral absent, confused, or contradictory
   • 4-6: Moral present but underdeveloped or lacking impact
   • 7-10: Clear, meaningful moral that provides genuine insight

4. **Adherence to Prompt (1-10)**:
   • 1-3: Missing multiple required elements from the prompt
   • 4-6: Incorporates main elements but overlooks some instructions
   • 7-10: Thoroughly addresses all prompt requirements while maintaining narrative cohesion

5. **Age Group Fit**:
   Determine which age group this fable is most appropriate for based on:
   • Vocabulary complexity and sentence structure
   • Conceptual difficulty of the moral lesson
   • Story length and complexity
   • Content appropriateness

   Age groups are defined as:
     - A: 3 years or under
     - B: 4-7 years
     - C: 8-11 years
     - D: 12-15 years
     - E: 16 years or above

Format your response as valid JSON with this structure:
{
    "type": "Fable Evaluation",
    "grammar": <integer 1-10>,
    "creativity": <integer 1-10>,
    "moral_clarity": <integer 1-10>,
    "adherence_to_prompt": <integer 1-10>,
    "best_age_group": "<letter: A, B, C, D, or E>",
    "explanation": [
        "<One sentence explaining grammar & style score>",
        "<One sentence explaining creativity & originality score>",
        "<One sentence explaining moral clarity score>",
        "<One sentence explaining adherence to prompt score>",
        "<One sentence explaining why this fable best fits the chosen age group>"
    ]
}

Be critical but fair. Ensure your entire evaluation is concise yet informative.

Original Prompt:
{{prompt}}

Fable:
{{fable}}"""

JSON_SCHEMA = {
    "name": "fable_evaluation",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "type": {"type": "string"},
            "grammar": {"type": "integer"},
            "creativity": {"type": "integer"},
            "moral_clarity": {"type": "integer"},
            "adherence_to_prompt": {"type": "integer"},
            "best_age_group": {
                "type": "string",
                "enum": ["A", "B", "C", "D", "E"],
            },
            "explanation": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "type", "grammar", "creativity", "moral_clarity",
            "adherence_to_prompt", "best_age_group", "explanation",
        ],
    },
}

RUBRIC = Rubric(
    task_name="tf1_generation",
    system_prompt=SYSTEM_PROMPT,
    evaluation_prompt_template=EVALUATION_PROMPT,
    json_schema=JSON_SCHEMA,
    score_fields=["grammar", "creativity", "moral_clarity", "adherence_to_prompt"],
    score_range=(1, 10),
    text_field_map={"prompt": "prompt", "fable": "fable"},
    extra_fields=["type", "best_age_group", "explanation"],
)
