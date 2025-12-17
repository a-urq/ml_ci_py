# ml-ci-py
A machine learning model made to predict the probability of convective initiation within a 2-hour lead time model, using RAP environmental parameters for predictors. This Python package was created by Amelia Urquhart and Ricky Cavaliero as part of their OU Senior Capstone project. 

# Contributors
##### ***Amelia R H Urquhart**, University of Oklahoma School of Meteorology / Oklahoma Weather Lab / Storm Chase Archive*
##### ***Ricky Cavaliero**, University of Oklahoma School of Meteorology*
##### ***Dr. Eric Loken**, National Severe Storms Laboratory / Cooperative Institute for Severe and High-Impact Weather Research and Operations*

# Self-hosted Paper
A research paper detailing the creation and verification statistics of ML-CI version 1 has been uploaded to this repository under the filename "ML-CI Research Paper.pdf". It is available [here](https://github.com/a-urq/ml_ci_py/blob/main/ML-CI%20Research%20Paper.pdf). It is important to note that this paper is solely self-published and has not gone through peer review as of this time.

# How to Use
For initial testing, the best way to test this is to clone this repository and run the following terminal command in the repository's root folder.

`python3 -m tests.mlci_mrms_test`

A new image file will be created, and it should look like this:

<img width="1100" height="850" alt="image" src="https://github.com/user-attachments/assets/810f4097-6654-47c4-a4e9-42d6d664d52a" />

The UTC time used for computation and plotting is stored near the end of the `mlci_mrms_test.py` file, in the following variable.

testing_dt = datetime(2023, 4, 19, 22, 00)

This plots ML-CI probability contours and MRMS reflectivity for 2023-04-19 22:00 UTC. This can be changed to any date and time where MRMS and RAP data both exist in AWS's archives. Any date in 2022 should be safe, but 2021's RAP data coverage is inconsistent and fully absent in 2020 and before.

There are two ways to use this package in your own projects. The first approach uses the same RAP data source that was used to train the model.

```
from ml_ci_py import ml_ci_rap

dt = datetime(...)
forecast_hour = 1 # Controls which forecast hour in the relevant RAP run is used. 1 is suggested, which allows for real-time or simulated-real-time use

# (Recommended) returns numpy arrays containing probability, latitude, and longitude grids
ci_probs, ci_lats, ci_lons = ml_ci_rap.get_ci_probs(dt, forecast_hour, return_latlons=True)
# Only returns the probability grid
ci_probs = ml_ci_rap.get_ci_probs(dt, forecast_hour, return_latlons=False)
```

To supply your own model data, `ml_ci.compute_probabilities()` can be called directly. Be aware that this has only been tested with RAP data, and will likely work poorly with uneven grid spacings (GFS equirectangular grid, etc). RAP's grid spacing is not truly constant but stays very close to 13.5 km, allowing reasonable advection calculations to be made. This limitation may be addressed in future versions.

```
from ml_ci_py import ml_ci

# Relevant helper functions can be found in ml_ci_rap.py within the GitHub repository.
rap, rap_1hr, rap_12hr = get_rap_files(dt, fh)

# No model grid provided, must be supplied by the user for plotting
ci_probs = ml_ci.compute_probabilities(
    rap, rap_1hr, rap_12hr, RAP_GRID_SPACING, return_latlons
)
```
# Verification

<img width="1200" height="400" alt="multipanel-verif-2" src="https://github.com/user-attachments/assets/7a539e75-fca1-4754-a479-f94b7386a6d6" />

The model has a strong tendency to overestimate the probability of CI above 20%, and this must be accounted for in any operational use of this model. A modeled probability region of 100% will only produce a storm about 27% of the time. A general false alarm rate of 80% can be expected. However, despite these flaws, the model is able to skillfully distinguish between the conditions that prohibit convective initiation and the conditions that allow it. While false positives are a serious probalem with the current version of this model, false negatives are much rarer. Work is planned to continue in improving this model throughout the coming months.
