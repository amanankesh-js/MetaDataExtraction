import yaml

def read_yaml(config_path):
    with open(config_path) as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            exit()
    return config