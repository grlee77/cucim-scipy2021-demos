import matplotlib.pyplot as plt
import napari
import napari.types
import numpy as np
import skimage.filters
from magicgui import magicgui
from skimage import color, data, filters, io

try:
    import cupy as cp
    from cucim.skimage import color as color_gpu
    from cucim.skimage import filters as filters_gpu
    cucim_available = True
except ImportError:
    pass
    cucim_available = False

try:
    # image source:
    # https://www.kaggle.com/c/diabetic-retinopathy-detection/data?select=sample.zip
    retina = io.imread("./17_left.jpeg")
except FileNotFoundError:
    # use low-res retina bundled with scikit-image
    retina = data.retina()


@magicgui(
    auto_call=True,
    function_name={"widget_type": "ComboBox", "choices": ["sato", "frangi", "meijering", "hessian"]},
    sigma_low={"widget_type": "Slider", "min": 1, "max": 25, "step": 1, "tracking": False},
    sigma_high={"widget_type": "Slider", "min": 2, "max": 25, "step": 1, "tracking": False},
    black_ridges={"widget_type": "CheckBox"},
    mode={"widget_type": "ComboBox", "choices": ["reflect", "constant", "nearest", "mirror", "wrap"], "name": "boundary"},
    autoclip={"widget_type": "CheckBox"},
    use_gpu={"widget_type": "CheckBox"},
)
def apply_ridge_filter(
    function_name: str="frangi",
    sigma_low: int=1,
    sigma_high: int=10,
    black_ridges: bool=True,
    mode: str="reflect",
    autoclip: bool=True,
    use_gpu: bool=False,  #  cucim_available
) -> "napari.types.ImageData":

    if use_gpu:
        if not cucim_available:
            raise ValueError("cuCIM could not be imported")
        image = color_gpu.rgb2gray(cp.asarray(retina))
        filters_module = filters_gpu
        xp = cp
    else:
        image = color.rgb2gray(retina)
        filters_module = filters
        xp = np

    if sigma_high < sigma_low:
        raise ValueError("sigma_low must be < sigma_high")

    image = image.astype(np.float32)
    ridge_filter = getattr(filters_module, function_name)
    result = ridge_filter(image,
                          sigmas=range(sigma_low, sigma_high),
                          black_ridges=black_ridges,
                          mode=mode)
    if autoclip:
        # remove lowest and highest 0.5% of values
        vmin, vmax = xp.percentile(result, q=[0.5, 99.5])
        result = xp.clip(result, vmin, vmax)
    # normalize result to range  [0, 1] for display
    result -= result.min()
    result /= result.max()

    if use_gpu:
        result = cp.asnumpy(result)

    return result


viewer = napari.view_image(retina)
viewer.window.add_dock_widget(apply_ridge_filter, name="Ridge Filters")

napari.run()
