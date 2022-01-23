import os
import numpy as np
import pandas as pd
from train import TrainNN
from dataset import ImagesDataSet
from imageutils import AugmentedImageDataGenerator
import albumentations as A

if __name__ == "__main__":
    home_dir = os.getcwd()
    base_dir = os.path.join(home_dir, 'data')
    train_dir = os.path.join(base_dir, 'train')
    weight_dir = os.path.join(base_dir, 'weight')
    test_dir = os.path.join(base_dir, 'test')
    out_dir = os.path.join(base_dir, 'out')

    image_size = 672
    batch_size = 12

    files_list = [str(fname) for fname in os.listdir(test_dir)]

    test_df = pd.DataFrame(data=files_list, columns=['image_name'])


    test_augmentations_list = [A.CenterCrop(height=image_size,
                                            width=image_size,
                                            p=1.0),
                               A.Normalize(mean=(0.5, 0.5, 0.5),
                                           std=(0.5, 0.5, 0.5),
                                           p=1.0),
                               ]

    test_gen = AugmentedImageDataGenerator(dataframe=test_df,
                                           directory=test_dir,
                                           x_col="image_name",
                                           y_col=None,
                                           num_classes=3,
                                           batch_size=batch_size,
                                           target_size=(image_size, image_size),
                                           augmentations_list=test_augmentations_list,
                                           augmentations=True,
                                           shuffle=False,
                                           validate_filenames=True,
                                           cache=True,
                                           subset='test'
                                           )


    print(f'Image Size = {image_size}x{image_size}')

    """ Universal part until this """
    dataset = ImagesDataSet(train_dir,
                            os.path.join(base_dir, "train.csv"),
                            image_size=image_size,
                            )
    dataset.batch_size = batch_size
    dataset.validation_split = 0.1
    dataset.build()
    tr = TrainNN(dataset)
    tr.monitor = "f1_score"

    y_pred = tr.get_predict(test_gen)
    """ Universal part from this """

    y_pred = np.argmax(y_pred, axis=1)
    submission_df = test_df.copy()
    submission_df['class_id'] = y_pred
    path_filename = os.path.join(out_dir, 'submission.csv')
    submission_df.to_csv(path_filename, index=False, sep='\t')
    print("Ok")
