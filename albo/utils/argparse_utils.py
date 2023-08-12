import os
import re
from argparse import Action


class KeyValueAction(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        kv_dict = {}
        for value in values:
            key, val = value.split('=')
            kv_dict[key] = val
        setattr(namespace, self.dest, kv_dict)


def add_override_arguments(parser):
    parser.add_argument("-D", dest="overrides", nargs="*", metavar="KEY=VALUE", action=KeyValueAction)


def convert_value(value_str):
    if value_str.lower() == 'true':
        return True
    elif value_str.lower() == 'false':
        return False
    try:
        value = int(value_str)
    except ValueError:
        try:
            value = float(value_str)
        except ValueError:
            value = value_str
    return value


def update_config(config, keys, value):
    key = keys.pop(0)
    if len(keys) == 0:
        if key.isdigit():
            key = int(key)
        config[key] = convert_value(value)
    else:
        if key.isdigit():
            key = int(key)
        if key not in config:
            config[key] = {}
        update_config(config[key], keys, value)


def apply_updates(config, updates):
    for update_key, update_value in updates.items():
        keys = update_key.split(".")
        update_config(config, keys, update_value)


def nested_get(d, keys):
    for key in keys:
        try:
            d = d[key]
        except KeyError:
            return "{" + ".".join(keys) + "}"
    return d


def format_nested(input_string, params):
    output_string = ''
    chunks = input_string.split('{')
    for chunk in chunks:
        if '}' in chunk:
            key, rest = chunk.split('}', 1)
            keys = key.split('.')
            value = nested_get(params, keys)
            output_string += str(value) + rest
        else:
            output_string += chunk
    return output_string


def replace_env_vars(data):
    if not isinstance(data, (dict, list)):
        raise TypeError("Input must be a dictionary or list.")

    def cast_value(value):
        try:
            if value.lower() == 'true':
                return True
            elif value.lower() == 'false':
                return False
            elif '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            return value

    def repl(match):
        env_var = os.getenv(match.group(1))
        if env_var is None:
            raise EnvironmentError(f"Environmental variable ${match.group(1)} not found.")
        return env_var

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (str, dict, list)):
                if isinstance(value, str):
                    new_value = re.sub(r'\{\$(\w+)\}', repl, value)
                    if new_value != value and not re.search(r'\{\$(\w+)\}', new_value):
                        data[key] = cast_value(new_value)
                    else:
                        data[key] = new_value
                else:
                    replace_env_vars(value)
    elif isinstance(data, list):
        for i in range(len(data)):
            if isinstance(data[i], (str, dict, list)):
                if isinstance(data[i], str):
                    new_value = re.sub(r'\{\$(\w+)\}', repl, data[i])
                    if new_value != data[i] and not re.search(r'\{\$(\w+)\}', new_value):
                        data[i] = cast_value(new_value)
                    else:
                        data[i] = new_value
                else:
                    replace_env_vars(data[i])

    return data