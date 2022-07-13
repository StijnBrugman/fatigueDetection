import argparse

class Parser():
    def __init__(self):
        parser = argparse.ArgumentParser()
        
        parser.add_argument('-c', '--cam', action='store_true', help='turn on camera visualization')
        parser.add_argument('-v', '--vis', action='store_true', help='turn on graph visualization')
        parser.add_argument('-s', '--safe', action='store_true', help='safe obtained images')
        
        self.args = vars(parser.parse_args())
    
    def get_arg(self, arg):
        return self.args.get(arg)