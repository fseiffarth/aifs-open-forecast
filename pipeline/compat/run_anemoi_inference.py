"""Launch `anemoi-inference` with compatibility patches for the AIFS Single v2.0 pins.

Patch 1 (anemoi-plugins-ecmwf-inference 0.2.1 vs anemoi-transform 0.1.16.post2):
    The plugin's InferenceOrography declares the newer filter keys (orography/geopotential)
    and passes orog="gh", but the pinned anemoi-transform Orography reads self.orog/self.z,
    so constructing the open-data input fails with KeyError: 'z'. Supplying both key sets
    makes the filter work with either anemoi-transform API.
"""

import sys

from anemoi.plugins.ecmwf.inference.opendata import geopotential_height as _gh

_gh.InferenceOrography.optional_inputs = {
    "orography": "gh",
    "geopotential": "z",
    "orog": "gh",
    "z": "z",
}

from anemoi.inference.__main__ import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
