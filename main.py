import argument_parser
import gann_base
# import tensorflow as tf
import tflowtools as TFT
import numpy as np


# TODO: optimizers needs arguments
# TODO: must normalize input from .txt files
# TODO: source needs more arguments, readfunctions etc

# NOTE: use softmax for ofunc and cross_entropy as loss for classification
# NOTE: use sigmoid or tanh for ofunc and MSE for regression


def main():
    parser = argument_parser.argument_parser()
    parser.parse()
    parser.organize()
    # (self, cases, vfrac, tfrac, casefrac, mapsep)
    caseman = gann_base.Caseman(parser.data_set_v, parser.vfrac_v, parser.tfrac_v, parser.casefrac_v, parser.mapbs_v)
    # (self, dims, cman, afunc, ofunc, cfunc, optimizer, lrate, wrange, vint, mbs, showint=None)
    ann = gann_base.Gann(parser.dims_v, caseman, parser.afunc_v, parser.ofunc_v, parser.cfunc_v, parser.optimizer_v,
                parser.lrate_v, parser.wrange_v, parser.vint_v, parser.mbs_v,
                showint=parser.steps_v-1)

    for layer in parser.dispw_v:
        ann.add_grabvar(layer, type='wgt')
        ann.gen_probe(layer, 'wgt', 'hist')
    for layer in parser.dispb_v:
        ann.add_grabvar(layer, type='bias')
        ann.gen_probe(layer, 'bias', 'hist')

    # run, then map
    ann.run(steps=parser.steps_v, sess=None, continued=False, bestk=1)

    ann.remove_grabvars()
    for layer in parser.maplayers_v:
        if layer == 0:
            ann.add_grabvar(layer, type='in', add_figure=False)
        else:
            ann.add_grabvar(layer - 1, type='out', add_figure=False)
    res, labs = ann.do_mapping()
    results = []
    for i in range(len(res[0])):
        l = np.array([r[i] for r in res])
        l = l.reshape(l.shape[0], l.shape[2])
        TFT.hinton_plot(l, title="mapping test output of layer " + str(parser.maplayers_v[i]))
        results.append(l)

    for i, r in enumerate(results):
        # DENDOGRAM
        if parser.maplayers_v[i] in parser.mapdend_v:
            TFT.dendrogram(r, list(map(TFT.one_hot_to_int, labs)), title="Dendrogram " + str(parser.maplayers_v[i]))

    gann_base.PLT.show()
    TFT.fireup_tensorboard('probeview')

main()
