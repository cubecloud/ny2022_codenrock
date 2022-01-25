import os
import numpy as np
import pandas as pd
from train import TrainNN
from dataset import ImagesDataSet
from imageutils import AugmentedImageDataGenerator
import albumentations as A
from tensorflow.keras.preprocessing.image import ImageDataGenerator

if __name__ == "__main__":
    home_dir = os.getcwd()
    base_dir = os.path.join(home_dir, 'data')
    train_dir = os.path.join(base_dir, 'train')
    weight_dir = os.path.join(base_dir, 'weight')
    test_dir = os.path.join(base_dir, 'test')
    out_dir = os.path.join(base_dir, 'out')

    image_size = 224
    batch_size = 12

    files_list = [str(fname) for fname in os.listdir(test_dir)]

    test_df = pd.DataFrame(data=files_list, columns=['image_name'])

    datagen = ImageDataGenerator(rescale=(1 / 127.5) - 1.0)
    test_gen = datagen.flow_from_dataframe(dataframe=test_df,
                                           directory=test_dir,
                                           x_col="image_name",
                                           shuffle=False,
                                           batch_size=batch_size,
                                           class_mode=None,
                                           target_size=(image_size, image_size)
                                           )
    print(f'Image Size = {image_size}x{image_size}')

    # test_augmentations_list = [A.CenterCrop(height=image_size,
    #                                         width=image_size,
    #                                         p=1.0),
    #                            A.Normalize(mean=(0.5, 0.5, 0.5),
    #                                        std=(0.5, 0.5, 0.5),
    #                                        p=1.0),
    #                            ]
    #
    # test_gen = AugmentedImageDataGenerator(dataframe=test_df,
    #                                        directory=test_dir,
    #                                        x_col="image_name",
    #                                        y_col=None,
    #                                        num_classes=3,
    #                                        batch_size=batch_size,
    #                                        target_size=(image_size, image_size),
    #                                        augmentations_list=test_augmentations_list,
    #                                        augmentations=True,
    #                                        shuffle=False,
    #                                        validate_filenames=True,
    #                                        cache=True,
    #                                        subset='test',
    #                                        color_mode='RGB'
    #                                        )

    print(f'Image Size = {image_size}x{image_size}')

    """ Universal part until this """
    tr = TrainNN(dataset=None,
                 image_size=image_size)
    tr.monitor = "f1_score"

    y_pred = tr.get_predict(test_gen)
    """ Universal part from this """

    y_pred = np.argmax(y_pred, axis=1)
    submission_df = test_df.copy()
    submission_df['class_id'] = y_pred
    path_filename = os.path.join(out_dir, 'submission.csv')
    submission_df.to_csv(path_filename, index=False, sep='\t')
    print("Ok")
