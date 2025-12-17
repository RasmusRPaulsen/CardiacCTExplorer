from pathlib import Path
from typing import Union
import cardiacctexplorer.general_utils as gu
from cardiacctexplorer.totalsegmentator_utils import compute_totalsegmentator_segmentations
from cardiacctexplorer.cardiac_utils import cardiac_analysis
# from cardiacctexplorer.aorta_utils import aorta_analysis
from cardiacctexplorer.fileconverter_utils import convert_input_files
# from cardiacctexplorer.measurement_utils import process_measurements


def get_default_parameters():
    default_parms = {
        "nudf_model_url": "https://shapeml.compute.dtu.dk/NUDF/",
        "nudf_checkpoint": "NUDF_checkpoint_epoch_24_17-12-2025.tar",
        "nudf_model_folder": "",
        "num_proc_total_segmentator": 1,
        "num_proc_general": 1,
        "recurse_subfolders": False,
        "out_of_reconstruction_value": -2048,
        "forced_aorta_min_hu_value": None,
        "forced_aorta_max_hu_value": None,
        "aorta_min_hu_value": 80,
        "aorta_min_max_hu_value": 400,
        "aorta_calcification_min_hu_value": 400,
        "aorta_calcification_max_hu_value": 1500,
        "aorta_calcification_std_multiplier": 3,
        "hounsfield_unit_offset": 0,
        "compute_centerline_from_ts_segmentation": True,
        "compare_with_totalsegmentator": False,
        "rendering_window_size": [1920, 1080],
        "device" : "gpu",
        "verbose": True,
        "quiet": False,
        "write_log_file": True
    }
    return default_parms


def cardiacctexplorer(in_name: Union[str, Path], output: Union[str, Path], cardiacct_parameters) -> bool:
    """
    Run AortaExplorer from within Python.

    For explanation of the arguments see description of command line
    arguments in bin/cardiacctexplorer.

    Return: success or not
    """
    quiet = cardiacct_parameters.get("quiet", False)
    verbose = cardiacct_parameters.get("verbose", False)
    write_log_file = cardiacct_parameters.get("write_log_file", False)

    # TODO: Need real article link
    if not quiet:
        print("\nIf you use this tool please cite TBD article\n")

    # ts_nr_proc = cardiacct_parameters.get("num_proc_total_segmentator", 1)
    # tg_nr_proc = cardiacct_parameters.get("num_proc_general", 1)
    recurse_subfolder = cardiacct_parameters.get("recurse_subfolders", False)

    output = str(output)
    Path(output).mkdir(parents=True, exist_ok=True)

    in_files, msg = gu.gather_input_files_from_input(in_name=in_name, recurse_subfolders=recurse_subfolder)
    if len(in_files) < 1:
        if write_log_file:
            gu.write_message_to_log_file(base_dir=output, message=msg, level="error")
        if not quiet:
            print(msg)
        return False
    if verbose:
        print(f"Found {len(in_files)} input files")

    # TODO: add later
    # in_files = convert_input_files(in_files=in_files, output_folder=output, params=aorta_parameters, nr_tg=tg_nr_proc,
    #                                verbose=verbose, quiet=quiet, write_log_file=write_log_file)
    #
    compute_totalsegmentator_segmentations(in_files=in_files, output_folder=output, params=cardiacct_parameters)

    cardiac_analysis(in_files=in_files, output_folder=output, params=cardiacct_parameters)

    # aorta_analysis(
    #     in_files=in_files,
    #     output_folder=output,
    #     params=aorta_parameters,
    #     nr_tg=tg_nr_proc,
    #     verbose=verbose,
    #     quiet=quiet,
    #     write_log_file=write_log_file,
    # )
    #
    # process_measurements(
    #     in_files=in_files, output_folder=output, verbose=verbose, quiet=quiet
    # )
    return True
