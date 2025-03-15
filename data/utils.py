import yaml


def load_yaml_config(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def load_filenames(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]