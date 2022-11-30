import argparse
import yaml

''' ArgParser
'''
def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--train",
            help="train mode",
            action='store_true',
            )
    parser.add_argument(
            "--val",
            help="val mode",
            action='store_true',
            )
    parser.add_argument(
            "--test",
            help="test mode",
            action='store_true',
            )
    parser.add_argument(
            "--dev",
            help="dev mode",
            action='store_true',
            )
    parser.add_argument(
            "--ckpt", 
            help="specify ckpt name", 
            default="", 
            type=str
            )
    args = parser.parse_args()
    
    return args

''' YamlParser
'''
def create_yaml_parser():
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
