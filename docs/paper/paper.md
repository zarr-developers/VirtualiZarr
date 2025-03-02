---
title: 'VirtualiZarr: Cloud-Optimize Archival Data as a Virtual Zarr Store, Using Xarray Syntax'
tags:
    - cloud computing
    - data access
    - climate data
    - python
    - xarray
    - zarr
    - icechunk
    - serverless
authors:
  - family-names: "Nicholas"
    given-names: "Thomas"
    orcid: "https://orcid.org/0000-0002-2176-0530"
    affiliation: "1, 2"
  - family-names: "Hagen"
    given-names: "Raphael"
    orcid: "https://orcid.org/0000-0003-1994-1153"
    affiliation: 3
  - name: Sean Harkins
    orcid: 0000-0000-0000-0000
    affiliation: 4
  - name: Aimee Barciauskas
    orcid: https://orcid.org/0000-0002-3158-9554
    affiliation: 4
  - name: Max Jones
    orcid: https://orcid.org/0000-0003-0180-8928
    affiliation: 4
  - name: Julia Signell
    orcid: https://orcid.org/0000-0002-4120-3192
    affiliation: 5
  - name: Ayush Nag
    orcid: 0000-0000-0000-0000
    affiliation: "6, 7"
  - name: Gustavo Hidalgo
    orcid: 0000-0000-0000-0000
    affiliation: 7
  - name: Tom Augspurger
    orcid: https://orcid.org/0000-0002-8136-7087
    affiliation: 7
  - name: Ryan Abernathey
    orcid: https://orcid.org/0000-0001-5999-4917
    affiliation: 1
affiliations:
  - name: Earthmover
    index: 1
  - name: "[C]Worthy"
    index: 2
  - name: "CarbonPlan"
    index: 3
  - name: "DevelopmentSeed"
    index: 4
  - name: "Element84"
    index: 5
  - name: "Paul G. Allen School of Computer Science and Engineering, University of Washington, Seattle, WA, USA"
    index: 6
  - name: "Microsoft"
    index: 7
date: 2 March 2025
bibliography: paper.bib

---

# Summary

Cloud storage is a great way to make very large datasets available to scientists and the public[@Abernathey:2021]. Unfortunately it is very slow to access data in old file formats placed in cloud object storage[@Rocklin:2020], as the formats were not designed in a “cloud-optimized” way. VirtualiZarr is a domain-agnostic tool which takes data saved in such pre-cloud file formats (e.g. netCDF, TIFF) and allows it to be accessed efficiently, as if it had instead been saved in the cloud-optimized array format Zarr[@zarr-developers:2024]. VirtualiZarr emphasises ease-of-use by re-using the widely-used and familiar user interface of Xarray[@Hoyer:2017]. It also integrates with the Icechunk transactional storage engine[@Icechunk], allowing archival data to be version-controlled and incrementally updated without disrupting user access. Together these tools allow scientists to easily and efficiently access vast quantities of data without running any server themselves, without the data provider organizations having to duplicate all their data.

# Statement of need

Massive quantities of public data are being moved to cloud storage, but are often contrained to stay in pre-cloud archival formats such as netCDF for archiving, provenance, or compatibility reasons.

VirtualiZarr is a python tool for creating “virtual” Zarr datacubes, enabling cloud-optimized access to multi-file datasets in a range of archival file formats (e.g. netCDF and TIFF) without copying the original data. Data is accessed either via the Kerchunk [@Kerchunk] references format, or via the Icechunk cloud-native transactional storage engine[@Icechunk]. Both store “virtual Zarr chunks” in the form of references to byte ranges in other pre-existing objects, and both allow users to access arbitrarily large datacubes using the familiar Xarray [@Hoyer:2017] interface. Although some previous tools to do this existed (particularly Kerchunk[@Kerchunk]), VirtualiZarr is significantly easier to use, more reliable, more scalable, and is extensible to cloud-optimizing a variety of custom file formats.

# How does it work?

VirtualiZarr works by creating a metadata-only representation of files in legacy formats, including references to byte ranges inside specific chunks of data on disk. VirtualiZarr is similar to the Kerchunk package which inspired it, except that it uses an array-level representation of the underlying data, stored in “chunk manifests”. Metadata-only references to data are saved to disk either via the Kerchunk on-disk reference file format, or using the Icechunk transactional storage engine, which facilitates later cloud-optimized access using Zarr-Python v3 and Xarray.

This approach has three advantages:

1. An array-level abstraction means users of VirtualiZarr do not need to learn a new interface, as they can use Xarray to manipulate virtual representations of their data to arrange the files comprising their datacube.

2. “Chunk manifests” enable writing the virtualized arrays out as valid Zarr stores directly (using Icechunk), meaning Zarr API implementations in any language can read the archival data directly. Zarr as a “universal reader” will allow data providers to serve all their archival multidimensional data via a common interface, regardless of the actual underlying file formats.

3. The integration with Icechunk allows “virtual” and “native” chunks to be treated interchangeably, so that an initial version of a datacube pointing at archival file formats can be gradually updated with new icechunk-native chunks with the safety of ACID transactions without the data users needing to make any distinction.

# Serverless generation and access

VirtualiZarr demonstrates the power of a serverless computing paradigm for science in two distinct ways.

First, when generating references for a large number of archival files, VirtualiZarr is able to parallelize the reference generation using any python `Executor`, which can perform the tasks in parallel across serverless functions-as-a-service platforms such as AWS Lambda. The problem is a good fit for serverless execution as it is an embarrassingly-parallel map step followed by a single reduce step, and the latter can be performed on the client. Here the advantage of the serverless paradigm is that the user does not need to decide how many machines to deploy, as the optimal level of container-level parallelism can be automatically chosen and deployed for them.

Second, once the virtual references have been deposited into Icechunk, the resultant cloud data store can be accessed by an arbitrary number of concurrent users without any server running atop S3. Even better, Icechunk's design allows users with write access (i.e. the data providers) to make arbitrary updates to the data even whilst other users are currently reading the data, whilst guaranteeing safety through ACID transactions. This serverless data sharing paradigm is a powerful one as it means data provider organisations no longer need to maintain constantly-running and scalable data portals in order to provide continuous access to all of their potential users[@Abernathey:2021].

# Acknowledgements

No direct financial support was given to work on this project, but the main developer (Thomas Nicholas) was initially partly supported to work on this project at [C]Worthy LLC. before being fully supported to continue work on this project at Earthmover PBC.

Thank you to all of the VirtualiZarr contributors and users, as well as the contributors to the Xarray, Zarr and Icechunk pprojects on which it depends. Special thanks to Martin Durant for his work on Kerchunk, which was a direct inspiration for VirtualiZarr.

# References
