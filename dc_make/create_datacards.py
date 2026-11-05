import os
import sys
import json

if __name__ == "__main__":
    file_dir = os.path.dirname(os.path.abspath(__file__))
    pkg_dir = os.path.dirname(file_dir)
    base_dir = os.path.dirname(pkg_dir)
    pkg_dir_name = os.path.split(pkg_dir)[1]
    if base_dir not in sys.path:
        sys.path.append(base_dir)
    __package__ = pkg_dir_name

from StatInference.dc_make.maker import DatacardMaker

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create datacards.")
    parser.add_argument("--input", required=True, type=str, help="input directory")
    parser.add_argument("--output", required=True, type=str, help="output directory")
    parser.add_argument("--config", required=True, type=str, help="configuration file")
    parser.add_argument(
        "--hist-bins",
        required=False,
        type=str,
        default=None,
        help="bin edges to rebin histograms",
    )
    parser.add_argument(
        "--param_values",
        required=False,
        type=str,
        default=None,
        help="parameter values to run only certain masses",
    )
    for param in DatacardMaker.customizeble_parameters:
        parser.add_argument(
            f"--{param}",
            required=False,
            type=str,
            default=None,
            help=f"custom value for {param}",
        )

    args = parser.parse_args()

    if args.hist_bins is not None:
        if args.hist_bins.endswith(".json"):
            hist_bins = args.hist_bins
        else:
            hist_bins = [float(x) for x in args.hist_bins.split(",")]
    else:
        hist_bins = None

    if args.param_values is not None:
        param_values = [int(x) for x in args.param_values.split(",")]
    else:
        param_values = None

    kwargs = {
        param: getattr(args, param) for param in DatacardMaker.customizeble_parameters
    }

    maker = DatacardMaker(
        args.config,
        args.input,
        hist_bins=hist_bins,
        param_values=param_values,
        **kwargs,
    )
    maker.createDatacards(args.output)
