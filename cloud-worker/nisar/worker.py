"""
Standalone Cloud Worker for NISAR L-band AWS S3 (`us-west-2`) Windowed Processing (Section 7 & 17).

Can be executed in an AWS `us-west-2` compute environment or locally to run
windowed HDF5/NetCDF4 subsetting and polarimetric covariance analysis for a job payload.
"""

import argparse
import json
import sys
from pathlib import Path

# Add backend directory to sys.path when invoked from CLI
BACKEND_DIR = Path(__file__).resolve().parents[2] / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.regions import validate_and_resolve_region  # noqa: E402
from app.processing.nisar.change_analysis import run_nisar_analysis_pipeline  # noqa: E402


def run_worker(region_id: str, start_date: str, end_date: str, product_type: str = "GCOV") -> dict:
    region = validate_and_resolve_region(region_id=region_id)
    res = run_nisar_analysis_pipeline(
        region=region,
        start_date=start_date,
        end_date=end_date,
        product_type=product_type,
    )
    return {
        "coverage": res["coverage"].model_dump(),
        "statistics": res["statistics"].model_dump(),
        "layer_id": res["layer"].layer_id,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Amazon Sentinel NISAR L-Band Cloud Worker")
    parser.add_argument("--region-id", default="madeira")
    parser.add_argument("--start-date", default="2024-03-01")
    parser.add_argument("--end-date", default="2024-04-15")
    parser.add_argument("--product-type", default="GCOV")
    args = parser.parse_args()
    out = run_worker(args.region_id, args.start_date, args.end_date, args.product_type)
    print(json.dumps(out, indent=2, default=str))
