import argument_parser
import gann_base
# import tensorflow as tf
import tflowtools as TFT


# TODO: fix bestk in gann_base.do_testing
# TODO: optimizers needs arguments
# TODO: must normalize input from .txt files

# NOTE: use softmax for ofunc and cross_entropy as loss for classification
# NOTE: use sigmoid or tanh for ofunc and MSE for regression


def main():
    parser = argument_parser.argument_parser()
    parser.parse()
    # (self, cfunc, vfrac, tfrac, casefrac)
    caseman = gann_base.Caseman(parser.source(), parser.vfrac(), parser.tfrac(), parser.casefrac())
    # (self, dims, cman, afunc, ofunc, cfunc, optimizer, lrate, wrange, vint, mbs, showint=None)
    ann = gann_base.Gann(parser.dims(), caseman, parser.afunc(), parser.ofunc(), parser.cfunc(), parser.optimizer(),
                parser.lrate(), parser.wrange(), parser.vint(), parser.mbs(),
                showint=parser.steps()-1)
    # ann.add_grabvar(0, type='in')
    # ann.add_grabvar(1, type='wgt')
    # ann.add_grabvar(1, type='out')
    # ann.add_grabvar(2)
    # ann.add_grabvar(2)
    ann.run(steps=parser.steps(), sess=None, continued=False, bestk=1)
    ann.reopen_current_session()
    for layer in parser.maplayers():
        ann.add_grabvar(layer, type='wgt')
    for case in caseman.get_testing_cases()[:parser.mapbs()]:
        ann.do_testing(ann.current_session, [case], msg='Mapping', bestk=1)
    ann.close_current_session()
    ann.runmore(steps=100)

    gann_base.PLT.show()
    # TFT.fireup_tensorboard('probeview')

main()
