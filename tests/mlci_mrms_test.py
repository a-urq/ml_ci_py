import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from datetime import datetime, timedelta
import gzip
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import pygrib
import requests
import tempfile

from ml_ci_py import ml_ci_rap


def open_bytebuffer_in_pygrib(byte_buffer: bytes):
    tmp = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False)
    tmp.write(byte_buffer)
    tmp.flush()
    tmp.close()

    return pygrib.open(tmp.name)


def download_mrms_file_aws(dt: datetime):
    year = dt.year
    month = dt.month
    day = dt.day
    hour = dt.hour
    minute = dt.minute

    url = f"https://noaa-mrms-pds.s3.amazonaws.com/CONUS/SeamlessHSR_00.00/{year:04d}{month:02d}{day:02d}/MRMS_SeamlessHSR_00.00_{year:04d}{month:02d}{day:02d}-{hour:02d}{minute:02d}00.grib2.gz"

    file = requests.get(url, stream=True)
    file.raise_for_status()

    with gzip.open(file.raw) as f:
        file_content = f.read()
        return file_content


def get_mrms(dt: datetime):
    file = download_mrms_file_aws(dt)
    return open_bytebuffer_in_pygrib(file)


def make_mlci_mrms_plot(dt: datetime, img_name: str):
    print("Starting ML-CI/MRMS test plot using datetime", dt)
    print("Downloading MRMS radar data...")
    mrms = get_mrms(dt)

    refl_lats, refl_lons = mrms[1].latlons()
    refl_lons -= 360
    refl = mrms[1].values

    print("Downloading RAP Data and computing CI probabilities...")
    ci_probs, ci_lats, ci_lons = ml_ci_rap.get_ci_probs(dt, 1, return_latlons=True)

    colors = [
        (1, 1, 1),
        (1, 1, 1),
        (0.239, 0.733, 0.702),
        (0.024, 0.412, 0.039),
        (1, 0.75, 0),
        (1, 0.375, 0),
        (0.5, 0, 0),
        (1, 0.573, 1),
        (0.298, 0, 0.5),
    ]  # R -> G -> B
    n_bin = 1000  # Discretizes the interpolation into bins
    cmap = LinearSegmentedColormap.from_list("refl", colors, N=n_bin)

    print("Plotting data with matplotlib and cartopy...")
    proj = ccrs.LambertConformal(
        central_longitude=-95, central_latitude=35, standard_parallels=(25, 25)
    )

    fig = plt.figure(figsize=(11, 8.5))
    ax = plt.axes(projection=proj)
    ax.set_extent([-107.5, -88.5, 31.5, 42.5], crs=ccrs.PlateCarree())

    ax.set_title(
        f"{str(dt):s}Z - Reflectivity [dBZ] and 2-hour Convective Initiation Probability [20%, 40%, 60%, 80%]"
    )

    refl = ax.pcolormesh(
        refl_lons,
        refl_lats,
        refl,
        transform=ccrs.PlateCarree(),
        vmin=0,
        vmax=80,
        cmap=cmap,
    )
    cbar = plt.colorbar(refl, orientation="horizontal", pad=0.02, shrink=0.8)

    p20 = ax.contour(
        ci_lons,
        ci_lats,
        ci_probs,
        [0.2],
        transform=ccrs.PlateCarree(),
        colors="y",
        alpha=1,
    )
    p40 = ax.contour(
        ci_lons,
        ci_lats,
        ci_probs,
        [0.4],
        transform=ccrs.PlateCarree(),
        colors="#FF8800",
        alpha=1,
    )
    p60 = ax.contour(
        ci_lons,
        ci_lats,
        ci_probs,
        [0.6],
        transform=ccrs.PlateCarree(),
        colors="r",
        alpha=1,
    )
    p80 = ax.contour(
        ci_lons,
        ci_lats,
        ci_probs,
        [0.8],
        transform=ccrs.PlateCarree(),
        colors="m",
        alpha=1,
    )

    # reader = shpreader.Reader("County Shapefile/countyl010g.shp")

    # counties = list(reader.geometries())

    # COUNTIES = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())

    ax.add_feature(cfeature.BORDERS, linewidth=1, edgecolor="black")
    ax.add_feature(cfeature.COASTLINE, linewidth=1, edgecolor="black")
    # ax.add_feature(COUNTIES, facecolor="none", edgecolor="black", alpha=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor="black")

    plt.tight_layout()
    plt.savefig(img_name)


testing_dt = datetime(2025, 4, 17, 21, 00)

make_mlci_mrms_plot(testing_dt, f"tests/mlci-mrms-output.png")
