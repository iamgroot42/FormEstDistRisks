from operator import methodcaller

import torch
import torch.nn as nn
from torch.autograd import Variable


def get_cuda_state(obj):
    """
    Get cuda state of any object.

    :param obj: an object (a tensor or an `torch.nn.Module`)
    :raise TypeError:
    :return: True if the object or the parameter set of the object
             is on GPU
    """
    if isinstance(obj, nn.Module):
        try:
            return next(obj.parameters()).is_cuda
        except StopIteration:
            return None
    elif hasattr(obj, 'is_cuda'):
        return obj.is_cuda
    else:
        raise TypeError('unrecognized type ({}) in args'.format(type(obj)))

def predict(net, inputs):
    """
    Predict labels. The cuda state of `net` decides that of the returned
    prediction tensor.

    :param net: the network
    :param inputs: the input tensor (non Variable), of dimension [B x C x W x H]
    :return: prediction tensor (LongTensor), of dimension [B]
    """
    inputs = make_cuda_consistent(net, inputs)[0]
    inputs_var = Variable(inputs)
    outputs_var = net(inputs_var)
    predictions = torch.max(outputs_var.data, dim=1)[1]
    return predictions
