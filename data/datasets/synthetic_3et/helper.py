import os, pdb

# custom
from data.utils import load_yaml_config
from data.datasets.synthetic_3et.synthetic_dataset import SyntheticDataset
from data.transforms.helper import get_transforms

def get_3et_dataset(name, training_params, dataset_params):
     
    input_transforms, target_transforms = get_transforms(dataset_params, training_params)

    # Create datasets
    dataset = SyntheticDataset( 
        name=name,
        training_params=training_params,
        dataset_params=dataset_params, 
        input_transforms=input_transforms,
        target_transforms=target_transforms)

    return dataset


if __name__ == "__main__":

    default_params = load_yaml_config("configs/default.yaml")
    training_params = default_params["training_params"]
    dataset_params = default_params["dataset_params"] 
    dataset_params["input_channel"] = 1

    dataset = get_3et_dataset("train", training_params, dataset_params) 

    from torchvision import transforms 
    image = transforms.ToPILImage()(dataset[0][0][0].squeeze(0)) 
    image.save("output_image.png")
