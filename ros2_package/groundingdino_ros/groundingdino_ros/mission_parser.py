"""
Mission config parser for extracting text prompts and entities of interest
from a mission briefing.

Real mission briefings (/mission_briefing/config.json, and the
scenario description.json) describe targets as:

    {"entities_of_interest": [
        {"entity_id": "Car495",
         "entity_type": "Car",
         "attributes": {"color": "violet", "class": "SEDAN.1"}},
        ...]}

description.json nests the same records under
scenario_objective.entities_of_interest. An older hand-written shape,
{"mission": {"entities": [{"type": "car"}]}}, is still accepted.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class EntityOfInterest:
    """One mission target, as named by the episode."""

    entity_id: str
    entity_type: str = ""      # e.g. "Car"
    entity_class: str = ""     # e.g. "SEDAN.1"  (AirSim class token)
    color: str = ""            # e.g. "violet"   (GT colour vocabulary)
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_noun(self) -> str:
        """The coarse noun GroundingDINO can actually ground.

        "SEDAN.1" and "SEDAN.POLICE" are not distinguishable by an
        open-vocabulary detector, so the prompt uses the entity_type
        ("Car" -> "car") rather than the class token.
        """
        return (self.entity_type or "object").strip().lower()


def _entity_records(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Pull entity records out of whichever briefing shape this is."""

    if not isinstance(config, dict):
        return []

    # config.json: top-level
    records = config.get("entities_of_interest")

    # description.json: nested under the objective
    if not records:
        records = config.get("scenario_objective", {}).get(
            "entities_of_interest")

    # legacy hand-written shape: {"mission": {"entities": [{"type": ...}]}}
    if not records:
        legacy = config.get("mission", {}).get("entities")
        if legacy:
            records = [
                {"entity_id": e.get("id", ""),
                 "entity_type": e.get("type", ""),
                 "attributes": {}}
                for e in legacy if isinstance(e, dict)
            ]
            # These have no entity_id, so parse_entities_of_interest drops
            # them; parse_mission_config still gets the prompt noun.

    return records if isinstance(records, list) else []


def _load_config(config_path: str) -> Optional[Dict[str, Any]]:
    """Read and parse a briefing JSON file, or None if unusable."""

    try:
        config_file = Path(config_path)
        if not config_file.exists():
            print(f"[mission_parser] Config file not found: {config_path}")
            return None

        with open(config_file, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"[mission_parser] Failed to parse JSON: {e}")
        return None
    except Exception as e:
        print(f"[mission_parser] Error reading mission config: {e}")
        return None


def parse_entities_of_interest(config_path: str) -> List[EntityOfInterest]:
    """Parse the mission briefing into EntityOfInterest records.

    Only records carrying an entity_id are returned: an entity we cannot
    name cannot be an association target for Perception.target_entity_id.

    Returns [] if the file is missing or unparseable -- callers fall back
    to their configured defaults.
    """

    config = _load_config(config_path)
    if config is None:
        return []

    entities: List[EntityOfInterest] = []
    for record in _entity_records(config):
        if not isinstance(record, dict):
            continue
        attributes = record.get("attributes") or {}
        entity_id = str(record.get("entity_id", "")).strip()
        if not entity_id:
            continue
        entities.append(EntityOfInterest(
            entity_id=entity_id,
            entity_type=str(record.get("entity_type", "") or "").strip(),
            entity_class=str(attributes.get("class", "") or "").strip(),
            color=str(attributes.get("color", "") or "").strip().lower(),
            raw=record,
        ))

    if entities:
        print(f"[mission_parser] Parsed {len(entities)} entities of interest: "
              + ", ".join(f"{e.entity_id}({e.color} {e.entity_class})"
                          for e in entities))
    else:
        print("[mission_parser] No entities of interest found in config")

    return entities


def parse_mission_config(config_path: str) -> Optional[List[str]]:
    """Extract the groundable class nouns from a mission config.

    Kept for backward compatibility with callers that only want prompt
    nouns. Returns None when nothing could be parsed.

    This reads the raw records rather than parse_entities_of_interest(),
    because the legacy {"mission": {"entities": [...]}} shape carries a
    groundable noun but no entity_id -- usable as a prompt, not as an
    association target.
    """

    config = _load_config(config_path)
    if config is None:
        return None

    # De-duplicate, preserving order.
    class_names: List[str] = []
    for record in _entity_records(config):
        if not isinstance(record, dict):
            continue
        noun = str(record.get("entity_type", "") or "").strip().lower()
        if noun and noun not in class_names:
            class_names.append(noun)

    return class_names or None


def construct_text_prompt(class_names: List[str]) -> str:
    """Construct a GroundingDINO text prompt from class names.

    GroundingDINO expects "class1. class2. class3."
    """

    if not class_names:
        return ""

    class_names = [name.strip() for name in class_names if name.strip()]

    prompt = ". ".join(class_names)
    if not prompt.endswith("."):
        prompt += "."

    return prompt


def get_text_prompt_from_mission(
    config_path: str,
    default_classes: Optional[List[str]] = None
) -> str:
    """Get a text prompt from the mission config, falling back to defaults."""

    if default_classes is None:
        default_classes = ["car", "pedestrian"]

    class_names = parse_mission_config(config_path)

    if not class_names:
        print(f"[mission_parser] Using default classes: {default_classes}")
        class_names = default_classes

    return construct_text_prompt(class_names)


if __name__ == "__main__":
    import sys

    config_path = sys.argv[1] if len(sys.argv) > 1 \
        else "/mission_briefing/config.json"

    print(f"Testing mission parser with: {config_path}")
    print("=" * 60)

    entities = parse_entities_of_interest(config_path)
    for entity in entities:
        print(f"  {entity.entity_id}: type={entity.entity_type!r} "
              f"class={entity.entity_class!r} color={entity.color!r} "
              f"-> prompt noun {entity.prompt_noun!r}")

    print(f"Class names: {parse_mission_config(config_path)}")
    print(f"Text prompt: {get_text_prompt_from_mission(config_path)!r}")
