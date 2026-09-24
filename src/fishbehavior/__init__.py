"""fishbehavior: automated zebrafish behavior-state labeling pipeline.

Pipeline (`all` runs every step; `live` runs one video through them with a local page):
    workbook + videos -> match & check -> scene setup -> tracking -> features
    -> behavior labels (5 ethogram states) -> calibration -> datasets & plots
"""

__version__ = "0.1.0"
