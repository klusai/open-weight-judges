"""Verify each rubric produces valid prompts and schemas."""

import pytest

from judges.rubrics import RUBRICS
from judges.rubrics.base import Rubric


SAMPLE_TF1_ITEM = {
    "fable": "Once upon a time, a fox learned that honesty is the best policy.",
    "prompt": "Write a fable about a fox who learns about honesty.",
}

SAMPLE_TF2_ITEM = {
    "fable": "Once upon a time, a fox learned that honesty is the best policy.",
    "translated_fable": "A fost odată o vulpe care a învățat că onestitatea este cea mai bună politică.",
}

SAMPLE_TF3_ITEM = {
    "fable": "A fost odată un iepure care a învățat că răbdarea aduce roadele.",
    "prompt": "Scrie o fabulă despre un iepure care învață despre răbdare.",
}

SAMPLE_ITEMS = {
    "tf1_generation": SAMPLE_TF1_ITEM,
    "tf2_translation": SAMPLE_TF2_ITEM,
    "tf3_generation": SAMPLE_TF3_ITEM,
}


class TestRubricRegistry:
    def test_all_tasks_registered(self):
        assert "tf1_generation" in RUBRICS
        assert "tf2_translation" in RUBRICS
        assert "tf3_generation" in RUBRICS

    def test_rubrics_are_rubric_instances(self):
        for name, rubric in RUBRICS.items():
            assert isinstance(rubric, Rubric), f"{name} is not a Rubric"


class TestRubricPrompts:
    @pytest.mark.parametrize("task_name", ["tf1_generation", "tf2_translation", "tf3_generation"])
    def test_system_prompt_not_empty(self, task_name):
        rubric = RUBRICS[task_name]
        assert len(rubric.system_prompt) > 50

    @pytest.mark.parametrize("task_name", ["tf1_generation", "tf2_translation", "tf3_generation"])
    def test_render_prompt_fills_placeholders(self, task_name):
        rubric = RUBRICS[task_name]
        item = SAMPLE_ITEMS[task_name]
        rendered = rubric.render_prompt(item)
        assert "{{" not in rendered, f"Unresolved placeholder in {task_name}"
        assert len(rendered) > 100


class TestRubricSchemas:
    @pytest.mark.parametrize("task_name", ["tf1_generation", "tf2_translation", "tf3_generation"])
    def test_schema_has_required_structure(self, task_name):
        rubric = RUBRICS[task_name]
        schema = rubric.json_schema
        assert "name" in schema
        assert "schema" in schema
        assert "properties" in schema["schema"]
        assert "required" in schema["schema"]

    @pytest.mark.parametrize("task_name", ["tf1_generation", "tf2_translation", "tf3_generation"])
    def test_score_fields_in_schema(self, task_name):
        rubric = RUBRICS[task_name]
        props = rubric.json_schema["schema"]["properties"]
        for field in rubric.score_fields:
            assert field in props, f"Score field {field} missing from schema in {task_name}"

    @pytest.mark.parametrize("task_name", ["tf1_generation", "tf2_translation", "tf3_generation"])
    def test_score_range_valid(self, task_name):
        rubric = RUBRICS[task_name]
        lo, hi = rubric.score_range
        assert lo >= 0
        assert hi > lo
        assert hi <= 10
