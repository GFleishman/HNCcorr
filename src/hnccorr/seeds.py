from math import sin, cos, pi

from hnccorr.utils import eight_neighborhood, add_offset_set_coordinates


class PositiveSeedSelector:
    def __init__(self, radius):
        self._radius = radius

    def select(self, center_seed, movie_size):
        num_dimensions = len(movie_size)
        offsets = eight_neighborhood(num_dimensions, self._radius)
        # compute positive seeds
        positive_seeds = add_offset_set_coordinates(offsets, center_seed)
        # check if seeds are within boundaries
        return extract_valid_pixels(positive_seeds, movie_size)


class NegativeSeedSelector:
    def __init__(self, radius, count):
        self._radius = radius
        self._count = count

    def select(self, center_seed, movie_size):
        if len(movie_size) != 2:
            raise ValueError("Only 2-dimensional movies are currently supported.")

        angle_step = 2 * pi / float(self._count)

        angles = [i * angle_step for i in range(self._count)]
        offsets = {
            (round(self._radius * cos(x)), round(self._radius * sin(x))) for x in angles
        }
        negative_seeds = add_offset_set_coordinates(offsets, center_seed)

        # check if seeds are within boundaries
        return extract_valid_pixels(negative_seeds, movie_size)


def extract_valid_pixels(pixel_set, region_size):
    return {
        pixel
        for pixel in pixel_set
        if all([0 <= x < dim_len for x, dim_len in zip(pixel, region_size)])
    }
