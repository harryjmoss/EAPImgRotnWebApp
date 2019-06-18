# Image orientation detection
*App inspired by the work of [Deepanshu Thakur](https://github.com/Deepanshu2017/AlligatorOrCrocodile) and [Simon Willison](https://github.com/simonw/cougar-or-not)*

This webapp was created on behalf of the British Library [Endangered Archive Programme](https://eap.bl.uk/) to determine the orientation of images in their collection. The deep learning model is based on a 50-layer version of [resnet](https://arxiv.org/abs/1512.03385) and was trained on around 650 unique images from the EAP archives using the [fast.ai](https://docs.fast.ai/) library. Each image is duplicated into four representations, consisting of the application of 0, 90, 180 or 270 degrees of anti-clockwise rotation.

The 99MB pre-trained model is included in this package and is found within `resnet50_EAP_version6.pkl`

`imgapp.py` is a [Starlette](https://www.starlette.io/) API server which accepts image uploads or image URLs and runs them against the pre-calculated model.

A `Dockerfile` is included for convenience.

