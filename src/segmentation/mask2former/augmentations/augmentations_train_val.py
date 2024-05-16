import albumentations as A
import cv2


def get_train_transform(augmentation_params):
    downscale_interp = {
        'downscale': cv2.INTER_AREA,
        'upscale': cv2.INTER_NEAREST
    }

    train_transform = A.Compose([
        A.Sequential([
            # first augment spatial-level, then augment one pixel-level, the confirm squareness
            A.HorizontalFlip(p=augmentation_params['horizontal_flip_augs']),
            A.OneOf([
                A.Rotate(
                    limit=(-augmentation_params['rotation_augs'][0], augmentation_params['rotation_augs'][0]),
                    interpolation=4,
                    border_mode=4,
                    value=None,
                    mask_value=None,
                    rotate_method='largest_box',
                    crop_border=True,
                    always_apply=False,
                    p=augmentation_params['rotation_augs'][1]
                ),
                # SafeRotateUpdateFrame(limit=rotation_augs[0], interpolation=cv2.INTER_CUBIC, value=0, p=rotation_augs[1]),
                A.RandomCropFromBorders(
                    crop_left=augmentation_params['crop_p_augs'][0],
                    crop_right=augmentation_params['crop_p_augs'][0],
                    crop_top=augmentation_params['crop_p_augs'][0],
                    crop_bottom=augmentation_params['crop_p_augs'][0],
                    p=augmentation_params['crop_p_augs'][1]
                ),
            ], p=augmentation_params['spatial_level_augs']),
            # Because pixel level augs like blur can be sensitive to size
            A.LongestMaxSize(max_size=augmentation_params['dataset_min_size'], interpolation=cv2.INTER_AREA),
            # Pixel level
            A.OneOf([  # Select one pixel-level transform
                A.ColorJitter(
                    brightness=augmentation_params['jitter_augs'][0],
                    contrast=augmentation_params['jitter_augs'][1],
                    saturation=augmentation_params['jitter_augs'][2],
                    hue=augmentation_params['jitter_augs'][3],
                    p=augmentation_params['jitter_augs'][4]
                ),
                A.GaussianBlur(
                    blur_limit=(int(augmentation_params['gauss_blur_augs'][0]), int(augmentation_params['gauss_blur_augs'][1])),
                    p=augmentation_params['gauss_blur_augs'][2]
                ),
                A.Defocus(
                    radius=(int(augmentation_params['defocus_augs'][0]), int(augmentation_params['defocus_augs'][1])),
                    p=augmentation_params['defocus_augs'][2]
                ),
                A.MotionBlur(
                    blur_limit=(int(augmentation_params['motion_blur_augs'][0]), int(augmentation_params['motion_blur_augs'][1])),
                    allow_shifted=False,
                    p=augmentation_params['motion_blur_augs'][2]
                ),
                A.ZoomBlur(
                    max_factor=(augmentation_params['zoom_blur_augs'][0], augmentation_params['zoom_blur_augs'][1]),
                    step_factor=(augmentation_params['zoom_blur_augs'][2], augmentation_params['zoom_blur_augs'][3]),
                    p=augmentation_params['zoom_blur_augs'][4]
                ),
                A.Downscale(
                    scale_min=augmentation_params['downscale_augs'][0],
                    scale_max=augmentation_params['downscale_augs'][1],
                    interpolation=downscale_interp,
                    p=augmentation_params['downscale_augs'][2]
                ),
                # A.GaussNoise(var_limit=(gauss_noise_augs[0], gauss_noise_augs[1]), mean=0, p=gauss_noise_augs[2]),
                A.ImageCompression(
                    quality_lower=augmentation_params['compression_augs'][0],
                    quality_upper=augmentation_params['compression_augs'][1],
                    p=augmentation_params['compression_augs'][2]
                )
            ], p=augmentation_params['pixel_level_augs']),
            # Output level
            # Dont apply augs to paddings, because wont at test time, so pad after
            A.PadIfNeeded(min_height=augmentation_params['output_size'], min_width=augmentation_params['output_size'], border_mode=0, value=0),
            # But do apply normalization to paddings, so model doesnt think it is different than normal black
            A.Normalize(mean=augmentation_params['dataset_mean'], std=augmentation_params['dataset_std'], p=1),
        ], p=1.0)
    ])

    return train_transform


def get_val_transform(augmentation_params):
    val_transform = A.Compose([
        A.Sequential([
            # Because pixel level augs like blur can be sensitive to size
            A.LongestMaxSize(max_size=augmentation_params['dataset_min_size'], interpolation=cv2.INTER_AREA),
            # Dont apply augs to paddings, because wont at test time, so pad after
            A.PadIfNeeded(min_height=augmentation_params['output_size'], min_width=augmentation_params['output_size'], border_mode=0, value=0),
            # But do apply normalization to paddings, so model doesnt think it is different than normal black
            A.Normalize(mean=augmentation_params['dataset_mean'], std=augmentation_params['dataset_std'], p=1),
        ], p=1.0)
    ])

    return val_transform


def get_train_and_val_transform(augmentation_params):
    train_transform = get_train_transform(augmentation_params)
    val_transform = get_val_transform(augmentation_params)

    return train_transform, val_transform
