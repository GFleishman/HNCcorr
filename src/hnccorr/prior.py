# Copyright © 2017. Regents of the University of California (Regents). All Rights
# Reserved.
#
# Permission to use, copy, modify, and distribute this software and its documentation
# for educational, research, and not-for-profit purposes, without fee and without a
# signed licensing agreement, is hereby granted, provided that the above copyright
# notice, this paragraph and the following two paragraphs appear in all copies,
# modifications, and distributions. Contact The Office of Technology Licensing, UC
# Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-7201,
# for commercial licensing opportunities. Created by Quico Spaen, Roberto Asín-Achá,
# and Dorit S. Hochbaum, Department of Industrial Engineering and Operations Research,
# University of California, Berkeley.
#
# IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
# INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE
# OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE
# SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS
# IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES,
# ENHANCEMENTS, OR MODIFICATIONS.
"""Components for calcium-imaging prior segmentations in HNCcorr."""

import numpy as np
from PIL import Image

from hnccorr.utils import (
    add_offset_to_coordinate,
    add_offset_set_coordinates,
    generate_pixels,
    list_images,
)


class Prior:
    """Calcium imaging prior segmentation class.

    Data is stored in an in-memory numpy array. Class supports both 2- and 3-
    dimensional priors.

    Attributes:
        name(str): Name of the experiment.
        _data (np.array): Prior segmentation data. Array has size N1 x N2.
            N1 and N2 are the number of pixels in the first and second
            dimension respectively.
        _data_size (tuple): Size of array _data.
    """

    def __init__(self, name, data):
        self.name = name
        self._data = data
        self.data_size = data.shape

    @classmethod
    def from_tiff_image(cls, name, image_path):
        """Loads tiff image into a numpy array.

        Data is assumed to be stored as 32-bit unsigned integers.

        Args:
            name (str): Prior name.
            image_path (str): Path of prior segmentation image.

        Returns:
            Prior: Prior object created from image file.
        """

        with Image.open(image_path) as image:
            _image = np.array(image)
        return cls(name, _image)

    def __getitem__(self, key):
        """Provides direct access to the prior segmentation data.

        Prior is stored in array with shape (N_1, N_2, ...), 
        Where N_1, N_2, ... are the number of pixels in the first
        dimension, second dimension, etc.

        Args:
            key (tuple): Valid index for a numpy array.

        Returns:
            np.array
        """
        return self._data.__getitem__(key).astype(np.float64)

    def is_valid_pixel_coordinate(self, coordinate):
        """Checks if coordinate is a coordinate for a pixel in the prior."""
        if self.num_dimensions != len(coordinate):
            return False

        zero_tuple = (0,) * self.num_dimensions
        for i, lower, upper in zip(coordinate, zero_tuple, self.pixel_shape):
            if not lower <= i < upper:
                return False
        return True

    @property
    def pixel_shape(self):
        """Resolution of the prior in pixels."""
        return self.data_size

    @property
    def num_pixels(self):
        """Number of pixels in the prior."""
        return np.product(self.data_size)

    @property
    def num_dimensions(self):
        """Dimension of the prior."""
        return len(self.data_size)

    def extract_valid_pixels(self, pixels):
        """Returns subset of pixels that are valid coordinates for the prior."""
        return {pixel for pixel in pixels if self.is_valid_pixel_coordinate(pixel)}


class Prior_Patch:
    """Square subregion of Prior.

    Patch limits the data used for the segmentation of a potential cell. Given a center
    seed pixel, Patch defines a square subregion centered on the seed pixel with width
    patch_size. If the square extends outside the prior boundaries, then the subregion
    is shifted such that it stays within the prior boundaries.

    The patch also provides an alternative coordinate system with respect to the top
    left pixel of the patch. This pixel is the zero coordinate for the patch coordinate
    system. The coordinate offset is the coordinate of the top left pixel in the patch
    coordinate system.

    Attributes:
        _center_seed (tuple): Seed pixel that marks the potential cell. The pixel is
            represented as a tuple of coordinates. The coordinates are relative to the
            prior. The top left pixel of the prior represents zero.
        _coordinate_offset (tuple): Prior coordinates of the pixel that represents the
            zero coordinate in the Patch object. Similar to the Prior, pixels in the
            Patch are indexed from the top left corner.
        _data (np.array): Subset of the Prior data. Only data for the patch is stored.
        _prior (Prior): Prior for which the Patch object is a subregion.
        _num_dimensions (int): Dimension of the patch. It matches the dimension of the
            prior.
        _patch_size (int): length of the patch in each dimension. Must be an odd number.
    """

    def __init__(self, prior, center_seed, patch_size):
        """Initializes Patch object."""
        if patch_size % 2 == 0:
            raise ValueError("patch_size (%d) should be an odd number.")

        self._num_dimensions = prior.num_dimensions
        self._center_seed = center_seed
        self._patch_size = patch_size
        self._prior = prior
        self._coordinate_offset = self._compute_coordinate_offset()
        self._data = self._prior[self._prior_indices()]

    @property
    def pixel_shape(self):
        """Shape of the patch in pixels."""
        return (self._patch_size,) * self._num_dimensions

    def _compute_coordinate_offset(self):
        """Computes the coordinate offset of the patch.

        Confirms that the patch falls within the prior boundaries and shifts the patch
        if necessary. The center seed pixel may not be in the center of the patch if a
        shift is necessary.
        """
        half_width = int((self._patch_size - 1) / 2)

        topleft_coordinates = add_offset_to_coordinate(
            self._center_seed, (-half_width,) * self._num_dimensions
        )
        # shift left such that top left corner exists
        topleft_coordinates = list(max(x, 0) for x in topleft_coordinates)

        # bottomright corners (python-style index so not included)
        bottomright_coordinates = add_offset_to_coordinate(
            topleft_coordinates, (self._patch_size,) * self._num_dimensions
        )
        # shift right such that bottom right corner exists
        bottomright_coordinates = list(
            min(x, max_value)
            for x, max_value in zip(bottomright_coordinates, self._prior.pixel_shape)
        )

        topleft_coordinates = add_offset_to_coordinate(
            bottomright_coordinates, (-self._patch_size,) * self._num_dimensions
        )

        return topleft_coordinates

    def _prior_indices(self):
        """Computes the indices of the prior that correspond to the patch.

        For a patch with top left pixel (5, 5) and bottom right pixel (9, 9), this
        method returns ``(5:10, 5:10)`` which can be used to acccess the data
        corresponding to the patch in the prior.
        """
        bottomright_coordinates = add_offset_to_coordinate(
            self._coordinate_offset, (self._patch_size,) * self._num_dimensions
        )

        # pixel indices
        idx = []
        for start, stop in zip(self._coordinate_offset, bottomright_coordinates):
            idx.append(slice(start, stop))
        return tuple(idx)

    def to_prior_coordinate(self, patch_coordinate):
        """Converts a prior coordinate into a patch coordinate.

        Args:
            patch_coordinate (tuple): Coordinates of a pixel in patch coordinate system.

        Returns:
            tuple: Coordinate of pixel in prior coordinate system.
        """
        return add_offset_to_coordinate(patch_coordinate, self._coordinate_offset)

    def to_patch_coordinate(self, prior_coordinate):
        """Converts a prior coordinate into a patch coordinate.

        Args:
            prior_coordinate (tuple): Coordinates of a pixel in prior coordinate system.

        Returns:
            tuple: Coordinate of pixel in patch coordinate system.
        """
        return add_offset_to_coordinate(
            prior_coordinate, [-x for x in self._coordinate_offset]
        )

    def enumerate_pixels(self):
        """Returns the prior coordinates of the pixels in the patch."""
        return add_offset_set_coordinates(
            generate_pixels(self.pixel_shape), self._coordinate_offset
        )

    def __getitem__(self, key):
        """Access data for pixels in the patch. Indexed in patch coordinates."""
        return self._data[key]


