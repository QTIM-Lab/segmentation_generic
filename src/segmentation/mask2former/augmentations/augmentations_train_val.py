import albumentations as A
import cv2


def get_train_transform(aug_params):
    downscale_interp = {
        'downscale': cv2.INTER_AREA,
        'upscale': cv2.INTER_NEAREST
    }

    train_transform = A.Compose([
        A.Sequential([
            # first augment spatial-level, then augment one pixel-level, the confirm squareness
            A.HorizontalFlip(p=aug_params['horizontal_flip_augs']),
            A.OneOf([
                A.Rotate(
                    limit=(-aug_params['rotation_augs'][0], aug_params['rotation_augs'][0]),
                    interpolation=4,
                    border_mode=4,
                    value=None,
                    mask_value=None,
                    rotate_method='largest_box',
                    crop_border=True,
                    always_apply=False,
                    p=aug_params['rotation_augs'][1]
                ),
                # SafeRotateUpdateFrame(limit=rotation_augs[0], interpolation=cv2.INTER_CUBIC, value=0, p=rotation_augs[1]),
                A.RandomCropFromBorders(
                    crop_left=aug_params['crop_p_augs'][0],
                    crop_right=aug_params['crop_p_augs'][0],
                    crop_top=aug_params['crop_p_augs'][0],
                    crop_bottom=aug_params['crop_p_augs'][0],
                    p=aug_params['crop_p_augs'][1]
                ),
            ], p=aug_params['spatial_level_augs']),
            # Because pixel level augs like blur can be sensitive to size
            A.LongestMaxSize(max_size=aug_params['dataset_min_size'], interpolation=cv2.INTER_AREA),
            # Pixel level
            A.OneOf([  # Select one pixel-level transform
                A.ColorJitter(
                    brightness=aug_params['jitter_augs'][0],
                    contrast=aug_params['jitter_augs'][1],
                    saturation=aug_params['jitter_augs'][2],
                    hue=aug_params['jitter_augs'][3],
                    p=aug_params['jitter_augs'][4]
                ),
                A.UnsharpMask(
                    blur_limit=(aug_params['sharpen_augs'][0], aug_params['sharpen_augs'][1]),
                    p=aug_params['sharpen_augs'][2]
                ),
                A.GaussianBlur(
                    blur_limit=(int(aug_params['gauss_blur_augs'][0]), int(aug_params['gauss_blur_augs'][1])),
                    p=aug_params['gauss_blur_augs'][2]
                ),
                A.Defocus(
                    radius=(int(aug_params['defocus_augs'][0]), int(aug_params['defocus_augs'][1])),
                    p=aug_params['defocus_augs'][2]
                ),
                A.MotionBlur(
                    blur_limit=(int(aug_params['motion_blur_augs'][0]), int(aug_params['motion_blur_augs'][1])),
                    allow_shifted=False,
                    p=aug_params['motion_blur_augs'][2]
                ),
                A.ZoomBlur(
                    max_factor=(aug_params['zoom_blur_augs'][0], aug_params['zoom_blur_augs'][1]),
                    step_factor=(aug_params['zoom_blur_augs'][2], aug_params['zoom_blur_augs'][3]),
                    p=aug_params['zoom_blur_augs'][4]
                ),
                A.Downscale(
                    scale_min=aug_params['downscale_augs'][0],
                    scale_max=aug_params['downscale_augs'][1],
                    interpolation=downscale_interp,
                    p=aug_params['downscale_augs'][2]
                ),
                # A.GaussNoise(var_limit=(gauss_noise_augs[0], gauss_noise_augs[1]), mean=0, p=gauss_noise_augs[2]),
                A.ImageCompression(
                    quality_lower=aug_params['compression_augs'][0],
                    quality_upper=aug_params['compression_augs'][1],
                    p=aug_params['compression_augs'][2]
                )
            ], p=aug_params['pixel_level_augs']),
            # Output level
            # Dont apply augs to paddings, because wont at test time, so pad after
            A.PadIfNeeded(min_height=aug_params['output_size'], min_width=aug_params['output_size'], border_mode=0, value=0),
            # But do apply normalization to paddings, so model doesnt think it is different than normal black
            A.Normalize(mean=aug_params['dataset_mean'], std=aug_params['dataset_std'], p=1),
        ], p=1.0)
    ])

    return train_transform


def get_val_transform(aug_params):
    val_transform = A.Compose([
        A.Sequential([
            # Because pixel level augs like blur can be sensitive to size
            A.LongestMaxSize(max_size=aug_params['dataset_min_size'], interpolation=cv2.INTER_AREA),
            # Dont apply augs to paddings, because wont at test time, so pad after
            A.PadIfNeeded(min_height=aug_params['output_size'], min_width=aug_params['output_size'], border_mode=0, value=0),
            # But do apply normalization to paddings, so model doesnt think it is different than normal black
            A.Normalize(mean=aug_params['dataset_mean'], std=aug_params['dataset_std'], p=1),
        ], p=1.0)
    ])

    return val_transform


def get_train_and_val_transform(aug_params):
    train_transform = get_train_transform(aug_params)
    val_transform = get_val_transform(aug_params)

    return train_transform, val_transform
