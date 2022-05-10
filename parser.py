import argparse

class Parser():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--cam', action='store_true', help='turn on camera visualization')
        parser.add_argument('-v', '--vis', action='store_true', help='turn on graph visualization')
        self.args = vars(parser.parse_args())
        print("[INFO] Parser Options")
        parser.print_help() 
        print()
        print(self.args)
    
    def get_arg(self, arg):
        return self.args.get(arg)
