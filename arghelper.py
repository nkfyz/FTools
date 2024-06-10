import argparse

# parser = argparse.ArgumentParser(description="This is a program that does something")
# options = ["input_len"]

def generateArgparser(parser, options):
    for option in options:
        parser.add_argument("--" + option)
    return parser

# args = parser.parse_args()
# print(args.filename, args.count, args.verbose)
