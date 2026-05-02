"""Base rubric interface for task-specific evaluation rubrics."""

from dataclasses import dataclass, field


@dataclass
class Rubric:
    """Defines a complete evaluation rubric for a specific task.

    Attributes:
        task_name: Short identifier (e.g., "tf1_generation").
        system_prompt: System message for the judge.
        evaluation_prompt_template: User prompt template with {{placeholders}}.
        json_schema: Full JSON schema dict for strict mode on Ollama.
        score_fields: List of field names that contain numeric scores.
        score_range: (min, max) inclusive range for score validation.
        text_field_map: Maps placeholder names to input data keys.
        extra_fields: Non-score fields expected in the response (e.g., age group, explanations).
    """

    task_name: str
    system_prompt: str
    evaluation_prompt_template: str
    json_schema: dict
    score_fields: list[str]
    score_range: tuple[int, int]
    text_field_map: dict[str, str] = field(default_factory=dict)
    extra_fields: list[str] = field(default_factory=list)

    def render_prompt(self, item: dict) -> str:
        """Render the evaluation prompt template with item data."""
        prompt = self.evaluation_prompt_template
        for placeholder, data_key in self.text_field_map.items():
            prompt = prompt.replace("{{" + placeholder + "}}", item.get(data_key, ""))
        return prompt

    @property
    def expected_fields(self) -> list[str]:
        """All fields expected in the JSON response."""
        return self.score_fields + self.extra_fields
