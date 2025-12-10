from datetime import datetime, timedelta
import numpy as np
import pygrib
from io import BytesIO
import requests

from . import ml_ci


def download_rap_file_aws(dt: datetime, fh: int = 0):
    year = dt.year
    month = dt.month
    day = dt.day
    hour = dt.hour

    url = f"https://noaa-rap-pds.s3.amazonaws.com/rap.{year:04d}{month:02d}{day:02d}/rap.t{hour:02d}z.awp130pgrbf{fh:02d}.grib2"

    file = requests.get(url, stream=True)
    file.raise_for_status()
    # return BytesIO(file.content)
    return file.content


def get_rap_files(dt: datetime, fh: int = 0):
    dt = dt - timedelta(hours=fh)
    dt_1hr = dt - timedelta(hours=1)
    dt_12hr = dt - timedelta(hours=12)

    rap = download_rap_file_aws(dt, fh)
    rap_1hr = download_rap_file_aws(dt_1hr, fh)
    rap_12hr = download_rap_file_aws(dt_12hr, fh)

    return rap, rap_1hr, rap_12hr


RAP_GRID_SPACING: float = 13545.087 # meters


def get_ci_probs(dt: datetime, fh: int = 0, return_latlons: bool = False):
    rap, rap_1hr, rap_12hr = get_rap_files(dt, fh)

    ci_probs = ml_ci.compute_probabilities(
        rap, rap_1hr, rap_12hr, RAP_GRID_SPACING, return_latlons
    )

    return ci_probs


if __name__ == "__main__":
    testing_dt = datetime(2025, 5, 18, 18, 0)

    ci_probs = get_ci_probs(testing_dt, 1)

    print(np.min(ci_probs))
    print(np.mean(ci_probs))
    print(np.max(ci_probs))

    print(type(ci_probs))
    print(ci_probs)
