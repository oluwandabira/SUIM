

from models.deeplabv3 import Deeplabv3
from models.suim_net import SUIM_Net


def pick(arch: str, num_classes: int = 8):

    if arch == "deeplabv3":
        im_res = (320, 320, 3)
        model = Deeplabv3(weights=None, input_shape=im_res,
                          classes=num_classes)
        return model, img_res, arch + ".hdf5"

    elif arch == "suimnet_rsb":
        img_res = (320, 240, 3)
        suimnet = SUIM_Net(base="RSB", im_res=img_res, n_classes=num_classes)
        return suimnet.model, img_res, arch + ".hdf5"
