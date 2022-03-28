import imgaug
# from imgaug import weather
import numpy as np
import imgaug.parameters as parameters
from imgaug.augmenters import blur,arithmetic,weather
import cv2


def blur_(noise, sigma):
    return blur.blur_gaussian_(noise, sigma=sigma)

def motion_blur(noise, angle, speed):
        size = max(noise.shape[0:2])
        k = int(speed * size)
        if k <= 1:
            return noise

        # we use max(k, 3) here because MotionBlur errors for anything less
        # than 3
        blurer = blur.MotionBlur(
            k=max(k, 3), angle=angle, direction=1.0)
        return blurer.augment_image(noise)

def gate(noise, gate_noise, gate_size, random_state):
        # the beta distribution here has most of its weight around 1.0 and
        # will only rarely sample values around 0.0 the average of the
        # sampled values seems to be at around 0.6-0.75
        gate_noise = gate_noise.draw_samples(gate_size, random_state)
        gate_noise_up = imgaug.imresize_single_image(gate_noise, noise.shape[0:2],
                                                 interpolation="cubic")
        gate_noise_up = np.clip(gate_noise_up, 0.0, 1.0)
        return np.clip(
            noise.astype(np.float32) * gate_noise_up, 0, 255
        ).astype(np.uint8)

def generate_noise(height, width, density, random_state):
    noise = arithmetic.Salt(p=density, random_state=random_state)
    return noise.augment_image(np.zeros((height, width), dtype=np.uint8))

def postprocess_noise(noise_small_blur,nb_channels = 3):
    noise_small_blur_rgb = np.tile(
        noise_small_blur[..., np.newaxis], (1, 1, nb_channels))
    return noise_small_blur_rgb


def rain_gen(height = 512,width = 384,channel =3,drop_size=(0.01, 0.02),speed=(0.04, 0.20)):
    layer = weather.RainLayer(
                density=(0.03, 0.14),
                density_uniformity=(0.8, 1.0),
                drop_size=drop_size,
                drop_size_uniformity=(0.2, 0.5),
                angle=(-30, 30),
                speed=speed,
                blur_sigma_fraction=(0.001, 0.001),
                seed=None,
                random_state='deprecated',
                deterministic='deprecated'
            )


    rss = imgaug.random.RNG(1).duplicate(2)

    flake_size_uniformity = parameters.handle_continuous_param(
                (0.2, 0.5), "flake_size_uniformity",
                value_range=(0.0, 1.0))
    flake_size_sample = flake_size_uniformity.draw_sample()

    downscale_factor = np.clip(1.0 - flake_size_sample, 0.001, 1.0)
    height_down = max(1, int(height*downscale_factor))
    width_down = max(1, int(width*downscale_factor))
    noise = generate_noise(
                height_down,
                width_down,
                [0.01, 0.075],
                rss[0]
            )
    gate_noise = parameters.Beta(1.0, 1.0 - 0.5)
    noise = gate(noise, gate_noise, (8, 8), rss[1])
    noise = imgaug.imresize_single_image(noise, (height, width),
                                            interpolation="cubic")

    angle = parameters.handle_continuous_param([-30, 30], "angle").draw_sample()
    speed = parameters.handle_continuous_param(speed, "speed", value_range=(0.0, 1.0)).draw_sample()
    blur_sigma_fraction = parameters.handle_continuous_param([0.0001, 0.001], "blur_sigma_fraction", value_range=(0.0, 1.0))
    blur_sigma_fraction_sample = blur_sigma_fraction.draw_sample()
    blur_sigma_limits=(0.5, 3.75)


    sigma = max(height, width) * blur_sigma_fraction_sample
    sigma = np.clip(sigma,blur_sigma_limits[0], blur_sigma_limits[1])
    noise_small_blur = blur_(noise, sigma)
    noise_small_blur = motion_blur(noise_small_blur,angle,speed)
    image_f32 = np.zeros((height,width,channel),dtype=np.float32)
    noise_small_blur_rgb= layer._postprocess_noise(noise_small_blur,flake_size_sample, 3)

    return layer._blend(image_f32,speed,noise_small_blur_rgb)


