from robustness.datasets import ImageNet
from robustness.model_utils import make_and_restore_model

ds = ImageNet('/tmp')
# Load model
model_kwargs = {
	'arch': 'resnet50',
	'dataset': ds,
	'resume_path': '/p/adversarialml/as9rw/models_imagenet/resnet50-19c8e357.pth'
}
model, _ = make_and_restore_model(**model_kwargs)
model.eval()
