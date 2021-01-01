"""
@author: Haofan Wang - github.com/haofanwang
Modified by : iamgroot42
"""
from PIL import Image
import numpy as np
import torch as ch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms


class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        # If wrapped in DataParallel, extract model inside
        if isinstance(self.model, nn.DataParallel):
            self.model = self.model.module
        self.target_layer = target_layer

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the
            function at given layer
        """
        conv_output = None
        target_layer = self.target_layer
        # Normalize x to [-1, 1]
        x_ = (x.clone() - 0.5) / 0.5
        x_, conv_output = self.model.feature_model(x_,
                                                   with_latent=target_layer)
        return conv_output, x_

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        # x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        x = self.model.dnn(x)
        return conv_output, x


class ScoreCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        # If wrapped in DataParallel, extract model inside
        if isinstance(self.model, nn.DataParallel):
            self.model = self.model.module
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = 1 * (model_output.data.cpu().numpy() >= 0)
        # Get convolution outputs
        target = conv_output[0]
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i in range(len(target)):
            # Unsqueeze to 4D
            saliency_map = ch.unsqueeze(ch.unsqueeze(target[i, :, :],0),0)
            # Upsampling to input size
            saliency_map = F.interpolate(saliency_map, size=(160, 160), mode='bilinear', align_corners=False)
            if saliency_map.max() == saliency_map.min():
                continue
            # Scale between 0-1
            norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
            # Get the target score
            w = ch.sigmoid(self.extractor.forward_pass(input_image*norm_saliency_map)[1])[0]
            if target_class == 1: w = (1 - w)
            cam += w.data.cpu().numpy() * target[i, :, :].data.cpu().numpy()
        cam = np.maximum(cam, 0)
        # Normalize between 0-1
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255
        return cam


if __name__ == '__main__':
    import utils
    # Load model
    MODELPATH = "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/all/64_16/none/10_0.9233484619263742.pth"
    model = utils.FaceModel(512,
                            train_feat=True,
                            weight_init=None).cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(ch.load(MODELPATH))
    model.eval()

    # Get params
    target_example = 0  # Snake

    # Load an image
    transform = transforms.Compose([transforms.ToTensor()])
    td = utils.CelebACustomBinary(
        "/p/adversarialml/as9rw/datasets/celeba_raw_crop/"
        "splits/70_30/all/split_2/test",
        transform=transform)
    dataloader = DataLoader(td, batch_size=100, shuffle=False)

    x, y = next(iter(dataloader))
    prep_img = x[target_example:target_example+1].cuda()

    # Score cam
    score_cam = ScoreCam(model, target_layer=11)
    # Generate cam mask
    cam = score_cam.generate_cam(prep_img,
                                 target_class=0)
    print('Score cam completed')
