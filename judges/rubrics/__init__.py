"""Task-specific scoring rubrics for TF1/TF2/TF3."""

from judges.rubrics.tf1_generation import RUBRIC as TF1_RUBRIC
from judges.rubrics.tf2_translation import RUBRIC as TF2_RUBRIC
from judges.rubrics.tf3_generation import RUBRIC as TF3_RUBRIC

RUBRICS = {
    "tf1_generation": TF1_RUBRIC,
    "tf2_translation": TF2_RUBRIC,
    "tf3_generation": TF3_RUBRIC,
}
