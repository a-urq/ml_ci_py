import numpy as np

from datetime import datetime, timedelta
import joblib
import pandas as pd
from pathlib import Path
import pygrib
from scipy.ndimage import gaussian_filter
from sklearn.ensemble import RandomForestClassifier
import tempfile

from packaging import version

NUMPY_V2 = version.parse(np.__version__).major >= 2

# Weird ugly compatibility stuff
if NUMPY_V2:
    import sys
    import numpy._core as core

    sys.modules["numpy.core"] = core
    sys.modules["numpy.core.multiarray"] = core.multiarray
else:
    import sys
    import numpy.core as _core

    sys.modules["numpy._core"] = _core
    sys.modules["numpy._core.multiarray"] = _core.multiarray


def calc_upslope_flow(zsfc, uwnd, vwnd, grid_resolution_meters):
    usf = np.zeros(zsfc.shape)

    grad_zsfc = np.gradient(zsfc)

    for i in range(len(uwnd)):
        for j in range(len(uwnd[0])):
            usf[i, j] = (grad_zsfc[1][i, j] / grid_resolution_meters) * uwnd[i, j] + (
                grad_zsfc[0][i, j] / grid_resolution_meters
            ) * vwnd[i, j]

    return usf


def calc_moisture_flux_convergence(spfh, uwnd, vwnd, grid_resolution_meters):
    mfc = np.zeros(spfh.shape)

    grad_spfh = np.gradient(spfh)
    grad_uwnd = np.gradient(uwnd)
    grad_vwnd = np.gradient(vwnd)

    for i in range(len(uwnd)):
        for j in range(len(uwnd[0])):
            advection_term = -uwnd[i, j] * (
                grad_spfh[1][i, j] / grid_resolution_meters
            ) - vwnd[i, j] * (grad_spfh[0][i, j] / grid_resolution_meters)
            convergence_term = -spfh[i, j] * (
                (grad_uwnd[1][i, j] / grid_resolution_meters)
                + (grad_vwnd[0][i, j] / grid_resolution_meters)
            )
            mfc[i, j] = advection_term + convergence_term

    return mfc


def calc_mass_flux_convergence(tmp2m, pres, spfh, uwnd, vwnd, grid_resolution_meters):
    rho = calc_density(tmp2m, pres, spfh)

    msc = np.zeros(rho.shape)

    grad_rho = np.gradient(rho)
    grad_uwnd = np.gradient(uwnd)
    grad_vwnd = np.gradient(vwnd)

    for i in range(len(uwnd)):
        for j in range(len(uwnd[0])):
            advection_term = -uwnd[i, j] * (
                grad_rho[1][i, j] / grid_resolution_meters
            ) - vwnd[i, j] * (grad_rho[0][i, j] / grid_resolution_meters)
            convergence_term = -rho[i, j] * (
                (grad_uwnd[1][i, j] / grid_resolution_meters)
                + (grad_vwnd[0][i, j] / grid_resolution_meters)
            )
            msc[i, j] = advection_term + convergence_term

    return msc


def calc_density(tmp2m, pres, spfh):
    rho = np.zeros(pres.shape)

    R_d = 287.04
    R_v = 461.5
    EPSILON = R_d / R_v

    for i in range(len(rho)):
        for j in range(len(rho[0])):
            pressure = pres[i, j]
            specific_humidity = spfh[i, j]
            mixing_ratio = specific_humidity / (1 - specific_humidity)
            vapor_pressure = mixing_ratio / (mixing_ratio + EPSILON) * pressure

            virtual_temperature = tmp2m[i, j] / (
                1 - vapor_pressure / pressure * (1 - EPSILON)
            )

            rho[i, j] = pressure / (R_d * virtual_temperature)

    return rho


def calc_vorticity_advection(vort, uwnd, vwnd, grid_resolution_meters):
    dzeta_dt = np.zeros(vort.shape)

    grad_zeta = np.gradient(vort)

    for i in range(len(uwnd)):
        for j in range(len(uwnd[0])):

            advection_term = -uwnd[i, j] * (
                grad_zeta[1][i, j] / grid_resolution_meters
            ) - vwnd[i, j] * (grad_zeta[0][i, j] / grid_resolution_meters)
            dzeta_dt[i, j] = advection_term

    return dzeta_dt


def get_model_input_data(rap_grbs, rap_1h_grbs, rap_12h_grbs, grid_resolution_meters):
    tmp2m_grb = rap_grbs.select(name="2 metre temperature")[0]
    psfc_grb = rap_grbs.select(name="Surface pressure")[0]
    mlcape_grb = rap_grbs.select(
        name="Convective available potential energy",
        typeOfLevel="pressureFromGroundLayer",
        topLevel=9000,
        bottomLevel=0,
    )[0]
    mlcin_grb = rap_grbs.select(
        name="Convective inhibition",
        typeOfLevel="pressureFromGroundLayer",
        topLevel=9000,
        bottomLevel=0,
    )[0]
    rh850mb_grb = rap_grbs.select(
        name="Relative humidity", typeOfLevel="isobaricInhPa", level=850
    )[0]
    rh700mb_grb = rap_grbs.select(
        name="Relative humidity", typeOfLevel="isobaricInhPa", level=700
    )[0]
    z700mb_grb = rap_grbs.select(
        name="Geopotential height", typeOfLevel="isobaricInhPa", level=700
    )[0]
    z500mb_grb = rap_grbs.select(
        name="Geopotential height", typeOfLevel="isobaricInhPa", level=500
    )[0]
    absv500mb_grb = rap_grbs.select(
        name="Absolute vorticity", typeOfLevel="isobaricInhPa", level=500
    )[0]
    spfh2m_grb = rap_grbs.select(name="2 metre specific humidity")[0]
    zsfc_grb = rap_grbs.select(name="Orography")[0]
    uwnd10m_grb = rap_grbs.select(name="10 metre U wind component")[0]
    vwnd10m_grb = rap_grbs.select(name="10 metre V wind component")[0]
    uwnd500mb_grb = rap_grbs.select(
        name="U component of wind", typeOfLevel="isobaricInhPa", level=500
    )[0]
    vwnd500mb_grb = rap_grbs.select(
        name="V component of wind", typeOfLevel="isobaricInhPa", level=500
    )[0]

    tmp2m_arr = tmp2m_grb.values
    psfc_arr = psfc_grb.values
    mlcape_arr = mlcape_grb.values
    mlcin_arr = mlcin_grb.values
    rh700mb_arr = rh700mb_grb.values
    rh850mb_arr = rh850mb_grb.values
    z700mb_arr = z700mb_grb.values
    z500mb_arr = z500mb_grb.values
    absv500mb_arr = absv500mb_grb.values
    spfh2m_arr = spfh2m_grb.values
    zsfc_arr = zsfc_grb.values
    uwnd10m_arr = uwnd10m_grb.values
    vwnd10m_arr = vwnd10m_grb.values
    uwnd500mb_arr = uwnd500mb_grb.values
    vwnd500mb_arr = vwnd500mb_grb.values

    upslope_flow = calc_upslope_flow(
        zsfc_arr, uwnd10m_arr, vwnd10m_arr, grid_resolution_meters
    )
    mass_flux_convergence = calc_mass_flux_convergence(
        tmp2m_arr,
        psfc_arr,
        spfh2m_arr,
        uwnd10m_arr,
        vwnd10m_arr,
        grid_resolution_meters,
    )
    moisture_flux_convergence = calc_moisture_flux_convergence(
        spfh2m_arr, uwnd10m_arr, vwnd10m_arr, grid_resolution_meters
    )
    vorticity_advection_500mb = calc_vorticity_advection(
        absv500mb_arr, uwnd500mb_arr, vwnd500mb_arr, grid_resolution_meters
    )

    z700mb_1h_grb = rap_1h_grbs.select(
        name="Geopotential height", typeOfLevel="isobaricInhPa", level=700
    )[0]
    z500mb_1h_grb = rap_1h_grbs.select(
        name="Geopotential height", typeOfLevel="isobaricInhPa", level=500
    )[0]
    z500mb_12h_grb = rap_12h_grbs.select(
        name="Geopotential height", typeOfLevel="isobaricInhPa", level=500
    )[0]

    z700mb_1h_arr = z700mb_1h_grb.values
    z500mb_1h_arr = z500mb_1h_grb.values
    z500mb_12h_arr = z500mb_12h_grb.values

    z700mb_1h_delta = z700mb_arr - z700mb_1h_arr
    z500mb_1h_delta = z500mb_arr - z500mb_1h_arr
    z500mb_12h_delta = z500mb_arr - z500mb_12h_arr

    return [
        np.array(mlcape_arr),
        np.array(mlcin_arr),
        np.array(rh850mb_arr),
        np.array(rh700mb_arr),
        np.array(upslope_flow),
        np.array(mass_flux_convergence),
        np.array(moisture_flux_convergence),
        np.array(vorticity_advection_500mb),
        np.array(z700mb_1h_delta),
        np.array(z500mb_1h_delta),
        np.array(z500mb_12h_delta),
    ]


def make_input_data_into_matrix(data):
    df = pd.DataFrame(
        {
            "MLCAPE": data[0].flatten(),
            "MLCIN": data[1].flatten(),
            "850mb RH": data[2].flatten(),
            "700mb RH": data[3].flatten(),
            "Upslope Flow": data[4].flatten(),
            "Mass Conv": data[5].flatten(),
            "MFC": data[6].flatten(),
            "500mb Vort Adv": data[7].flatten(),
            "1hr 700mb Height Change": data[8].flatten(),
            "1hr 500mb Height Change": data[9].flatten(),
            "12hr 500mb Height Change": data[10].flatten(),
        }
    )
    return df


def get_model_input_matrix(grbs, grbs_1hr, grbs_12hr, grid_resolution_meters):
    raw_data = get_model_input_data(grbs, grbs_1hr, grbs_12hr, grid_resolution_meters)
    return make_input_data_into_matrix(raw_data)


def open_bytebuffer_in_pygrib(byte_buffer: bytes):
    tmp = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False)
    tmp.write(byte_buffer)
    tmp.flush()
    tmp.close()

    return pygrib.open(tmp.name)


MODULE_DIR = str(Path(__file__).resolve().parent)
BLUR_SIGMA_METERS = (
    6 * 13545.087
)  # Originally, 6x the RAP grid spacing was used for calibration


def compute_probabilities(
    grbs, grbs_1hr, grbs_12hr, grid_resolution_meters, return_latlons: bool = False
) -> np.ndarray:
    # Ensures file is open in pygrib
    if type(grbs) is bytes:
        grbs_open = open_bytebuffer_in_pygrib(grbs)
        grbs = grbs_open
    if type(grbs_1hr) is bytes:
        grbs_open = open_bytebuffer_in_pygrib(grbs_1hr)
        grbs_1hr = grbs_open
    if type(grbs_12hr) is bytes:
        grbs_open = open_bytebuffer_in_pygrib(grbs_12hr)
        grbs_12hr = grbs_open

    lats, lons = grbs[1].latlons()

    input_matrix = get_model_input_matrix(
        grbs, grbs_1hr, grbs_12hr, grid_resolution_meters
    )

    # NEED to figure out a way to make this file path system portable
    with open(f"{MODULE_DIR:s}/model_files/uncalibrated_ci_model.joblib", "rb") as f:
        rf: RandomForestClassifier = joblib.load(f)

        spatial_calib = joblib.load(
            open(
                f"{MODULE_DIR:s}/model_files/spatial_awareness_calibration.joblib", "rb"
            )
        )
        isotonic_calib = joblib.load(
            open(f"{MODULE_DIR:s}/model_files/isotonic_calibration.joblib", "rb")
        )

        uncalib_output_matrix = rf.predict_proba(input_matrix)[:, 1]
        uncalib_output_arr = uncalib_output_matrix.reshape(337, 451)

        gaussian_probs = gaussian_filter(
            uncalib_output_arr, BLUR_SIGMA_METERS / grid_resolution_meters
        )

        diff_gauss = uncalib_output_arr - gaussian_probs

        part_calib_input_matrix = pd.DataFrame(
            {
                "Probability": uncalib_output_arr.flatten(),
                "Difference from Local Average": diff_gauss.flatten(),
            }
        )

        part_calib_output_matrix = spatial_calib.predict_proba(part_calib_input_matrix)[
            :, 1
        ]
        output_matrix = isotonic_calib.predict(part_calib_output_matrix)

        output_arr = output_matrix.reshape(337, 451)

        output_arr = gaussian_filter(output_arr, 1)

        if return_latlons:
            return output_arr, lats, lons
        else:
            return output_arr


RAP_DIRECTORY = "/media/nvme1/Capstone Research/RAP Data/"


def get_rap_file(dt: datetime):
    dt -= timedelta(hours=1)

    # /media/nvme1/Capstone Research/RAP Data/2023/09/30/rap.20230930.t22z.awp130pgrbf01.subset.grib2
    fname = f"{RAP_DIRECTORY:s}{dt.year:04d}/{dt.month:02d}/{dt.day:02d}/rap.{dt.year:04d}{dt.month:02d}{dt.day:02d}.t{dt.hour:02d}z.awp130pgrbf01.subset.grib2"

    print(fname)

    grbs = pygrib.open(fname)

    return grbs


def get_rap_file_1hr_prev(dt: datetime):
    dt -= timedelta(hours=1)

    return get_rap_file(dt)


def get_rap_file_12hr_prev(dt: datetime):
    dt -= timedelta(hours=12)

    return get_rap_file(dt)


if __name__ == "__main__":
    testing_dt = datetime(2025, 5, 18, 18, 0)

    grbs = get_rap_file(testing_dt)
    grbs_1hr = get_rap_file_1hr_prev(testing_dt)
    grbs_12hr = get_rap_file_12hr_prev(testing_dt)

    grid_spacing_meters = 13545.087

    ci_probs = compute_probabilities(grbs, grbs_1hr, grbs_12hr, grid_spacing_meters)

    print(np.min(ci_probs))
    print(np.mean(ci_probs))
    print(np.max(ci_probs))
