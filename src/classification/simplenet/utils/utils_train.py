def get_val_samples(dm_in):
    # To access the x_dataloader we need to call prepare_data and setup.
    dm_in.prepare_data()
    dm_in.setup()

    # Samples required by the custom ImagePredictionLogger callback to log image predictions.
    val_samples = next(iter(dm_in.val_dataloader()))
    return val_samples
