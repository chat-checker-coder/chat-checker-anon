from enum import StrEnum
from pathlib import Path
import yaml


yaml.SafeDumper.add_multi_representer(
    StrEnum,
    yaml.representer.SafeRepresenter.represent_str,
)


def yaml_equivalent_of_path(dumper, data):
    return dumper.represent_str(str(data))


yaml.SafeDumper.add_multi_representer(Path, yaml_equivalent_of_path)
