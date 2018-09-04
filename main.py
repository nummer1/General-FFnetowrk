import argparse
import tensorflow as tf
import tflowtools as TFT

LOG_DIR = "probeview"


# layers_size = sys.argv[0]
# act_func = sys.argv[1]
# out_act_func = sys.argv[2]
# cost_func = sys.argv[3]
# lrate = sys.argv[4]
# wr_init = sys.argv[5]

class GANN():
    def parse_arguments(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-d", "--dims", nargs='+', type=int, required=True, \
            help="dimensions of the neural network")
        parser.add_argument("-s", "--datasource", required=True,
            help="data source")
        parser.add_argument("-a", "--afunc", required=False, \
            help="activation function of hidden layers")
        parser.add_argument("--oafunc", required=False, \
            help="activation function of output layer")
        parser.add_argument("-c", "--cfunc", required=False, \
            help="cost function / loss function")
        parser.add_argument("-l", "--lrate", type=float, required=False, \
            help="learning rate")
        parser.add_argument("-w", "--wrange", nargs=2, type=float, required=False, \
            help="lower and higher bound for random initialization of weights")
        parser.add_argument("-o", "--optimizer", required=False, \
            help="optimizer to use")
        parser.add_argument("--casefrac", required=False, \
            help="set fraction of data to use for training validation and testing")
        parser.add_argument("--valfrac", required=False)
        parser.add_argument("--testfrac", required=False)
        parser.add_argument("--vint", required=False, \
            help="number of training minibatches to use between each validation test")
        parser.add_argument("--minibsize", required=False, \
            help="number of cases in minibatch")
        parser.add_argument("--mapbsize", required=False, \
            help="number of training cases to be used for a map test. Zero indicates no map test")
        parser.add_argument("--steps", required=False, \
            help="total number of minibatches to be run through the system during training")
        parser.add_argument("--mlayers", required=False, \
            help="the layers to be visualized during map test")
        parser.add_argument("--mapdend", nargs='+', type=int, required=False, \
            help="list of layers whose activation layers will be used to produce dendograms")
        parser.add_argument("--dispw", required=False, \
            help="list of the weight arrays to be visualized at the end of run")
        parser.add_argument("--dispbias", required=False, \
            help="list of bias vectors to be visualized at the end or run")
        args = parser.parse_args()

        build()
        # do something with args
        # build network
        # return network (or not, maybe save as part of self)
        return args


def main():
    net = GANN()
    net.parse_arguments()


main()
