import argument_parser
import gann_base
# import tensorflow as tf
import tflowtools as TFT


# TODO: fix bestk in gann_base.do_testing
# TODO: fix input and output layer dimensions
# TODO: optimizers needs arguments

# NOTE: use softmax for ofunc and cross_entropy as loss for classification
# NOTE: use sigmoid or tanh for ofunc and MSE for regression


def main():
    parser = argument_parser.argument_parser()
    parser.parse()
    # (self, cfunc, vfrac, tfrac, casefrac)
    caseman = gann_base.Caseman(parser.source(), parser.vfrac(), parser.tfrac(), parser.casefrac())
    # (self, dims, cman, afunc, ofunc, cfunc, optimizer, lrate, wrange, vint, mbs, showint=None)
    gann = gann_base.Gann(parser.dims(), caseman, parser.afunc(), parser.ofunc(), parser.cfunc(), parser.optimizer(),
                parser.lrate(), parser.wrange(), parser.vint(), parser.mbs(),
                showint=None)
    gann.run(steps=parser.steps(), sess=None, continued=False, bestk=1)
    TFT.fireup_tensorboard('probeview')

main()
