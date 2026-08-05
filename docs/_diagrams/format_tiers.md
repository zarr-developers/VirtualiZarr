| format                 | Non-cloud-optimizable   | Cloud-optimizable via virtualization | Cloud-optimizable upon write | Cloud-optimized by default | Cloud-Native (static) | Cloud-Native (transactional) |
| ---------------------- | ----------------------- | ------------- | ---------------------------- | -------------------------- | --------------------- | ---------------------------- |
| .tar*                  |                         | x             |                              |                            |                       |                              |
| .gz*                   | x                       |               |                              |                            |                       |                              |
| .zip (deflate)*        | x                       |               |                              |                            |                       |                              |
| .zip (no compression)* |                         | x             |                              |                            |                       |                              |
| netCDF3                |                         | x             |                              |                            |                       |                              |
| netCDF4 / HDF5         |                         | x             | x                            |                            |                       |                              |
| "Cloud-optimized HDF5" |                         | x             | x                            | x                          |                       |                              |
| HDF4                   |                         | x             |                              |                            |                       |                              |
| GRIB2                  |                         | x             |                              |                            |                       |                              |
| FITS                   |                         | x             |                              |                            |                       |                              |
| TIFF (untiled)         |                         | x             |                              |                            |                       |                              |
| GeoTIFF                |                         | x             | x                            |                            |                       |                              |
| COG                    |                         | x             | x                            | x                          |                       |                              |
| Native Zarr (v2 or v3) |                         | x             | x                            | x                          | x                     |                              |
| Zipped Native Zarr     |                         | x             | x                            | x                          |                       |                              |
| Icechunk               |                         | x             | x                            | x                          | x                     | x                            |
