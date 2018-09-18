import argparse
import tensorflow as tf
import tflowtools as TFT

class argument_parser():
    # parses arguments given on command line
    def parse(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-d", "--dims", nargs='+', type=int, required=True,
                help="dimensions of the neural network")
        # parser.add_argument("-s", "--source", required=True,
        #         help="data source")
        parser.add_argument("-a", "--afunc", required=True, \
                help="activation function for hidden layers")
        parser.add_argument("--ofunc", required=True, \
                help="activation function of output layer")
        parser.add_argument("-c", "--cfunc", required=True, \
                help="cost function / loss function")
        parser.add_argument("-l", "--lrate", type=float, required=True, \
                help="learning rate")
        parser.add_argument("-w", "--wrange", nargs=2, type=float, required=True, \
                help="lower and higher bound for random initialization of weights")
        parser.add_argument("-o", "--optimizer", required=True, \
                help="what optimizer to use")
        parser.add_argument("--casefrac", type=float, required=True, \
                help="set fraction of data to use for training validation and testing")
        parser.add_argument("--vfrac", type=float, required=True, \
                help="validation fraction")
        parser.add_argument("--tfrac", type=float, required=True, \
                help="test fraction")
        parser.add_argument("--vint", type=int, required=True, \
                help="number of training minibatches to use between each validation test")
        parser.add_argument("--mbs", type=int, required=True, \
                help="number of cases in a minibatch")
        # parser.add_argument("--mapbs", required=True, \
        # help="number of training cases to be used for a map test. Zero indicates no map test")
        parser.add_argument("--steps", type=int, required=True, \
                help="total number of minibatches to be run through the system during training")
        # parser.add_argument("--maplayers", required=True, \
        # help="the layers to be visualized during map test")
        # parser.add_argument("--mapdend", nargs='+', type=int, required=True, \
        # help="list of layers whose activation layers will be used to produce dendograms")
        # parser.add_argument("--dispw", required=True, \
        # help="list of the weight arrays to be visualized at the end of run")
        # parser.add_argument("--dispb", required=True, \
        # help="list of bias vectors to be visualized at the end or run")
        self.args = parser.parse_args()

    def dims(self):
        print("dimensions:", self.args.dims)
        return self.args.dims

    def afunc(self):
        print("activation function:", self.args.afunc)
        dict = {"sigmoid": tf.nn.sigmoid, "relu": tf.nn.relu, "relu6": tf.nn.relu6, "tanh": tf.nn.tanh}
        if self.args.afunc in dict.keys():
            return dict[self.args.afunc]
        else:
            print("'", self.args.afunc, "' is invalid for argument --afunc", sep='')
            print("Valid arguments are:", dict.keys())
            quit()

    def ofunc(self):
        print("output activation function:", self.args.ofunc)
        dict = {"linear": None, "softmax": tf.nn.softmax, "sigmoid": tf.nn.sigmoid}
        if self.args.ofunc in dict.keys():
            return dict[self.args.ofunc]
        else:
            print("'", self.args.ofunc, "' is invalid for argument --ofunc", sep='')
            print("Valid arguments are:", dict.keys())
            quit()

    def cfunc(self):
        print("cost / lostt function:", self.args.cfunc)
        dict = {"mse": tf.losses.mean_squared_error, "softmax_ce": tf.losses.softmax_cross_entropy}
        if self.args.cfunc in dict.keys():
            return dict[self.args.cfunc]
        else:
            print("'", self.args.cfunc, "' is invalid for argument --cfunc", sep='')
            quit()

    def optimizer(self):
        print("optimizer:", self.args.optimizer)
        dict = {"gd": tf.train.GradientDescentOptimizer, "adagrad": tf.train.AdagradOptimizer, "adam": tf.train.AdamOptimizer,
                "rmsprop": tf.train.RMSPropOptimizer}
        if self.args.optimizer in dict.keys():
            return dict[self.args.optimizer]
        else:
            print("'", self.args.optimizer, "' is invalid for argument --optimizer", sep="")
            quit()

    def lrate(self):
        print("learning rate:", self.args.lrate)
        return self.args.lrate

    def wrange(self):
        print("weight range:", self.args.wrange)
        if self.args.wrange[0] > self.args.wrange[1]:
            print("wrange start (", self.args.wrange[0], ") is larger than finish (", self.args.wrange[1], ")", sep="")
            quit()
        return self.args.wrange

    def casefrac(self):
        print("casefrac:", self.args.casefrac)
        if self.args.casefrac > 1 or self.args.casefrac < 0:
            print("casefrac is larger than 1 or smaller than 0")
            quit()
        return self.args.casefrac

    def vfrac(self):
        print("validation fraction:", self.args.vfrac)
        if self.args.vfrac > 1 or self.args.vfrac < 0:
            print("vfrac is larger than 1 or smaller than 0")
            quit()
        return self.args.vfrac

    def tfrac(self):
        print("test fraction:", self.args.tfrac)
        if self.args.tfrac > 1 or self.args.tfrac < 0:
            print("tfrac is larger than 1 or smaller than 0")
            quit()
        return self.args.tfrac

    def vint(self):
        print("validation intervals:", self.args.vint)
        return self.args.vint

    def mbs(self):
        print("minibatch size:", self.args.mbs)
        return self.args.mbs

    def steps(self):
        print("steps:", self.args.steps)
        return self.args.steps

    # def data_source(self):
    #     # returns a list of inputs and targets
    #     data_set = []
    #
    #     if self.args.datasource[-4:] == ".txt":
    #         with open(self.args.datasource) as file:
    #             data_set_line = []
    #             for line in file.readlines():
    #                 for element in line.split(','):
    #                     data_set_line.append(float(element))
    #             data_set.append(data_set_line)
    #     else:
    #         if self.args.datasourse == "parity":
    #             data_set = TFT.gen_all_parity_cases()
    #         elif self.args.datasourse == "symmetry":
    #             data_set = TFT.gen_symvect_cases()
    #         elif self.args.datasourse == "one_hot":
    #             data_set = TFT.gen_all_one_hot_cases()
    #         elif self.args.datasourse == "auto_dense":
    #             data_set = TFT.gen_dense_autoencoder_cases()
    #         elif self.args.datasourse == "bit_counter":
    #             data_set = TFT.gen_vector_count_cases()
    #         elif self.args.datasourse == "segment_counter":
    #             data_set = TFT.gen_segmented_vector_cases()
    #
    #     return data_set
