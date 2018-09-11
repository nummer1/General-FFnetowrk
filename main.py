import argument_parser
import gann_base

LOG_DIR = "probeview"




def main():
    parser = argument_parser.argument_parser()
    parser.parse()
    network = parser.build()


main()
