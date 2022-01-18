import os
from train import TrainNN
from dataset import ImagesDataSet

if __name__ == "__main__":
    image_size = 256
    batch_size = 40

    print(f'Image Size = {image_size}x{image_size}')
    dataset = ImagesDataSet(os.path.join(os.getcwd(), "data", "train", "images"),
                            os.path.join(os.getcwd(), "data", "train", "train.csv"),
                            image_size=image_size,
                            )
    dataset.batch_size = batch_size
    # dataset.augmentation = False
    dataset.validation_split = 0.15
    dataset.build()

    tr = TrainNN(dataset)
    tr.monitor = "val_loss"
    tr.es_patience = 20
    tr.keras_model, tr.net_name = resnet50v2_classification_model(input_shape=(tr.dataset.image_size,
                                                                               tr.dataset.image_size) + (3,),
                                                                  num_classes=3)
    tr.train()

    print("Ok")
