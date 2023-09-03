# Config

## Template

```json
{
    "dsn": {
        // Sizes
        "feature_dim": int, // Dimensionality of the feature vector used in the algorithm
        // Mean Shift parameters (for 3D voting)
        "max_GMS_iters": int, // Maximum number of iterations for the Gaussian Mean Shift algorithm used for 3D voting
        "epsilon": float, // Connected Components parameter
        "sigma": float, // Gaussian bandwidth parameter
        "num_seeds": int, // Used for MeanShift, but not BlurringMeanShift
        "subsample_factor": int, // Used to subsample the input point cloud
        // Misc
        "min_pixels_thresh": int, // Minimum number of pixels required for a region proposal to be considered valid
        "tau": float // Threshold parameter used for region proposal filtering
    },
    "rrn": {
        // Sizes
        "feature_dim": int, // Dimensionality of the feature vector used in the algorithm
        "img_H": int, // Height of the input image
        "img_W": int, // Width of the input image
        // Architecture parameters
        "use_coordconv": bool // Boolean flag that specifies whether or not to use coordinate convolution in the architecture
    },
    "uois3d": {
        // Padding for RGB Refinement Network
        "padding_percentage": float,
        // Open/Close Morphology for IMP (Initial Mask Processing) module
        "use_open_close_morphology": bool,
        "open_close_morphology_ksize": int,
        // Largest Connected Component for IMP module
        "use_largest_connected_component": bool,
    }
}
```
