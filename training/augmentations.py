import albumentations as A
from albumentations.pytorch import ToTensorV2


def collate_fn(batch):
    """
    To handle the data loading as different images may have different number 
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))


def get_train_transform():
    return A.Compose([
        A.Flip(p=0.5),
        A.RandomRotate90(p=0.5),
        # A.MotionBlur(p=0.2),
        # A.MedianBlur(blur_limit=3, p=0.1),
        # A.Blur(blur_limit=3, p=0.1),
        A.OneOf([
            A.RGBShift(p=0.25, r_shift_limit=40,
                       g_shift_limit=40, b_shift_limit=40),
            A.HueSaturationValue(p=0.25),
            A.ChannelShuffle(p=0.25),
            A.ColorJitter(p=0.25)
        ]),
        A.OneOf([
            A.CLAHE(p=0.2),
            A.RandomContrast(p=0.2, limit=0.3),
            A.RandomGamma(p=0.2, gamma_limit=(40, 160)),
            A.RandomBrightness(p=0.2, limit=0.4)
        ]),
        A.MedianBlur(blur_limit=7, p=0.1),
        # A.JpegCompression(0.1),
        A.BBoxSafeRandomCrop(p=0.5),
        A.ToFloat(p=1.0),
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })

# define the validation transforms


def get_valid_transform():
    return A.Compose([
        A.ToFloat(p=1.0),
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })
