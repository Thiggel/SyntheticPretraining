from __future__ import annotations

import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .base import TaskBase
from .capo_biosbr import CapoGenerator, validate_capo_example

ASSETS_DIR = Path(__file__).resolve().parent / "capo_assets"


class CapoTask(TaskBase):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        self.base_dir = ASSETS_DIR
        self.generator = CapoGenerator(base_dir=self.base_dir)

    def generate_example(self, rng: random.Random) -> dict[str, Any]:
        person_id = rng.randint(0, max(0, int(self.config["num_people"]) - 1))
        person = self.generator.sample_person(rng, person_id=person_id)
        exposure_idx = rng.randint(0, max(0, int(self.config["exposures"]) - 1))
        return self.generator.generate_training_example(rng, person, self.config, exposure_idx)

    def validate_example(self, example: Mapping[str, Any]) -> bool:
        return bool(validate_capo_example(dict(example)))
