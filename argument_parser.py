import argparse
import re
import numpy
import tensorflow as tf
import tflowtools as TFT
import mnist_basics

class argument_parser():
    # parses arguments given on command line

    def __init__(self):
        self.source_is_called = False

    def parse(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-d", "--dims", nargs='+', type=int, required=True,
                help="dimensions of the neural network")
        parser.add_argument("-s", "--source", required=True,
                help="data source")
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
        parser.add_argument("--mapbs", type=int, required=True, \
                help="number of training cases to be used for a map test. Zero indicates no map test")
        parser.add_argument("--steps", type=int, required=True, \
                help="total number of minibatches to be run through the system during training")
        parser.add_argument("--maplayers", nargs='*', type=int, required=True, \
                help="the layers to be visualized during map test")
        parser.add_argument("--mapdend", nargs='*', type=int, required=True, \
                help="list of layers whose activation layers will be used to produce dendograms")
        parser.add_argument("--dispw", nargs='*', type=int, required=True, \
                help="list of the weight matrices to be visualized at the end of run")
        parser.add_argument("--dispb", nargs='*', type=int, required=True, \
                help="list of bias matrices to be visualized at the end or run")
        self.args = parser.parse_args()

    def organize(self):
        self.data_set_v = self.source()
        self.dims_v = self.dims()
        self.afunc_v = self.afunc()
        self.ofunc_v = self.ofunc()
        self.cfunc_v = self.cfunc()
        self.lrate_v = self.lrate()
        self.wrange_v = self.wrange()
        self.optimizer_v = self.optimizer()
        self.casefrac_v = self.casefrac()
        self.vfrac_v = self.vfrac()
        self.tfrac_v = self.tfrac()
        self.vint_v = self.vint()
        self.mbs_v = self.mbs()
        self.mapbs_v = self.mapbs()
        self.steps_v = self.steps()
        self.maplayers_v = self.maplayers()
        self.mapdend_v = self.mapdend()
        self.dispw_v = self.dispw()
        self.dispb_v = self.dispb()

    def dims(self):
        if not self.source_is_called:
            print("source() must be called before dims() is called")
            quit()
        self.args.dims = [len(self.data_set_v[0][0])] + self.args.dims + [len(self.data_set_v[0][1])]
        print("dimensions:", self.args.dims)
        return self.args.dims

    def source(self):
        def normalize(cases):
            input = [c[0] for c in cases]
            target = [c[1] for c in cases]
            input = numpy.array(input)
            min_arr = numpy.min(input, axis=0)
            max_arr = numpy.max(input, axis=0)
            for element in input:
                for i, e in enumerate(element):
                    element[i] = (e - min_arr[i])/(max_arr[i] - min_arr[i])
            return list(zip(input, target))

        def to_float(inp):
            # returns 0 if input is ? (questionmark)
            return 0 if inp == '?' else float(inp)

        self.source_is_called = True
        print("source:", self.args.source)
        data_set = []
        if self.args.source[-4:] == ".txt":
            with open("data_set_files/" + self.args.source) as file:
                data = list(map(lambda x: re.split("[;,]", x), file.readlines()))
                data = list(map(lambda x: list(map(to_float, x)), data))
            max_d = max(map(lambda x: int(x[-1]), data))
            for element in data:
                input = element[:-1]
                target = TFT.int_to_one_hot(int(element[-1])-1, max_d)
                data_set.append([input, target])
        elif self.args.source == "parity":
            data_set = TFT.gen_all_parity_cases(10)
        elif self.args.source == "symmetry":
            vecs = TFT.gen_symvect_dataset(101, 2000)
            inputs = list(map(lambda x: x[:-1], vecs))
            targets = list(map(lambda x: TFT.int_to_one_hot(x[-1], 2), vecs))
            data_set = list(zip(inputs, targets))
        elif self.args.source == "auto_onehot":
            data_set = TFT.gen_all_one_hot_cases(100)
        elif self.args.source == "auto_dense":
            data_set = TFT.gen_dense_autoencoder_cases(2000, 100)
        elif self.args.source == "bitcounter":
            data_set = TFT.gen_vector_count_cases(500, 15)
        elif self.args.source == "segmentcounter":
            data_set = TFT.gen_segmented_vector_cases(25, 1000, 0, 8)
        elif self.args.source == "mnist":
            # mnist_basics.load_all_flat_cases(type='testing')
            cases = mnist_basics.load_all_flat_cases(type='training')
            cases = list(zip(cases[0], cases[1]))
            data_set = list(map(lambda x: [x[0], TFT.int_to_one_hot(x[1], 10)], cases))

        if data_set == []:
            print(self.args.source, " is illegal for argument --source")
            print("Legal values are: <filenme>.txt, parity, symmetry, \
                        auto_onehot, auto_dense, bitcounter, segmentcounter", sep="")
            quit()
        if self.args.source[-4:] == ".txt":
            data_set = normalize(data_set)
        return data_set

    def afunc(self):
        print("activation function:", self.args.afunc)
        dict = {"sigmoid": tf.nn.sigmoid, "relu": tf.nn.relu, "relu6": tf.nn.relu6, "tanh": tf.nn.tanh}
        if self.args.afunc in dict:
            return dict[self.args.afunc]
        else:
            print("'", self.args.afunc, "' is invalid for argument --afunc", sep='')
            print("Valid arguments are:", dict.keys())
            quit()

    def ofunc(self):
        print("output activation function:", self.args.ofunc)
        dict = {"linear": None, "softmax": tf.nn.softmax, "sigmoid": tf.nn.sigmoid}
        if self.args.ofunc in dict:
            return dict[self.args.ofunc]
        else:
            print("'", self.args.ofunc, "' is invalid for argument --ofunc", sep='')
            print("Valid arguments are:", dict.keys())
            quit()

    def cfunc(self):
        print("cost / loss function:", self.args.cfunc)
        dict = {"mse": tf.losses.mean_squared_error, "softmax_ce": tf.losses.softmax_cross_entropy}
        if self.args.cfunc in dict:
            return dict[self.args.cfunc]
        else:
            print("'", self.args.cfunc, "' is invalid for argument --cfunc", sep='')
            print("Valid arguments are:", dict.keys())
            quit()

    def optimizer(self):
        print("optimizer:", self.args.optimizer)
        dict = {"gd": tf.train.GradientDescentOptimizer, "adagrad": tf.train.AdagradOptimizer, "adam": tf.train.AdamOptimizer,
                "rmsprop": tf.train.RMSPropOptimizer}
        if self.args.optimizer in dict:
            return dict[self.args.optimizer]
        else:
            print("'", self.args.optimizer, "' is invalid for argument --optimizer", sep="")
            print("Valid arguments are:", dict.keys())
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

    def mapbs(self):
        print("map batch size:", self.args.mapbs)
        return self.args.mapbs

    def maplayers(self):
        print("maplayers to visualize", self.args.maplayers)
        if not self.source_is_called:
            print("source must be called before maplayers is called")
            quit()
        for layer in self.args.maplayers:
            if layer > len(self.args.dims) - 2:
                print("maplayer ot visualize is larger than dimension")
                quit()
        return self.args.maplayers

    def mapdend(self):
        print("layers ot make dendograms of:", self.args.mapdend)
        return self.args.mapdend

    def dispw(self):
        print("weights to be displayed:", self.args.dispw)
        return self.args.dispw

    def dispb(self):
        print("biases to be displayed:", self.args.dispb)
        return self.args.dispb
