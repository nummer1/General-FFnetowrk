import argument_parser
import gann_base
# import tensorflow as tf
import tflowtools as TFT


# TODO: fix bestk in gann_base.do_testing
# TODO: fix input and output layer dimensions

# NOTE: use softmax for ofunc and cross_entropy as loss for classification
# NOTE: use sigmoid or tanh for ofunc and MSE for regression


def main():
    parser = argument_parser.argument_parser()
    parser.parse()
    cfunc = (lambda : TFT.gen_all_one_hot_cases(2**4))
    caseman = gann_base.Caseman(cfunc=cfunc, vfrac=parser.vfrac(), tfrac=parser.tfrac())
    gann = gann_base.Gann(dims=parser.dims(), cman=caseman, afunc=parser.afunc(), ofunc=parser.ofunc(), cfunc=parser.cfunc(),
                lrate=parser.lrate(), showint=100, mbs=10, vint=None)
    gann.run(epochs=1000, sess=None, continued=False, bestk=1)
    TFT.fireup_tensorboard('probeview')

main()
