import argument_parser
import gann_base
import tensorflow as tf
import tflowtools as TFT


# TODO: fix bestk in gann_base.do_testing

def dims(parser):
    print("dimensions: ", parser.args.dims)
    return parser.args.dims


def afunc(parser):
    dict = {"sigmoid": tf.nn.sigmoid, "relu": tf.nn.relu, "relu6": tf.nn.relu6, "tanh": tf.nn.tanh}
    try:
        return dict[parser.args.afunc]
    except KeyError:
        print("'", parser.args.afunc, "' is invalid for argument --afunc", sep="")
        print("Valid arguments are:", dict.keys())
        quit()


def main():
    parser = argument_parser.argument_parser()
    parser.parse()
    cfunc = (lambda : TFT.gen_all_one_hot_cases(2**4))
    caseman = gann_base.Caseman(cfunc, vfrac=0.1, tfrac=0.1)
    gann = gann_base.Gann(dims=dims(parser), cman=caseman, afunc=afunc(parser),
                lrate=.03, showint=100, mbs=10, vint=None, softmax=False)
    gann.run(epochs=30000, sess=None, continued=False, bestk=1)
    TFT.fireup_tensorboard('probeview')

main()
