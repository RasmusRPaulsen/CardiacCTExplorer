import os
from shutil import copyfile

from acvl_utils.cropping_and_padding.bounding_boxes import crop_to_bbox
from scipy.ndimage import measurements
from pathlib import Path
import time
import multiprocessing as mp
import cardiacctexplorer.general_utils as gu
import cardiacctexplorer.io_utils as io_utils
import cardiacctexplorer.segmentation_utils as su
from cardiacctexplorer.nudf_predictor import NUDFPredictor
import SimpleITK as sitk
import numpy as np
from skimage.morphology import skeletonize, binary_dilation, binary_closing, ball
from skimage.measure import label
from skimage.segmentation import find_boundaries
import edt
import json
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import imageio
import skimage.io
from skimage import color
from skimage.util import img_as_ubyte
import importlib.metadata
from scipy.ndimage import distance_transform_edt
import torch
from torch.utils.model_zoo import load_url
from totalsegmentator.python_api import select_device


def extract_laa_roi(params):
    """
    Extract the region around the left atrial appendage
    """
    input_file = params["input_file"]
    ts_folder = params["ts_folder"]
    output_folder = params["output_folder"]
    # scan_id = params["pure_id"]
    crops_folder = params["crops_folder"]
    quiet = params["quiet"]
    verbose = params["verbose"]
    write_log_file = params["write_log_file"]

    segm_name_total = f"{ts_folder}total.nii.gz"
    crop_out_name = f"{crops_folder}laa_roi.nii.gz"
    # crop_out_name_np = f"{crops_folder}laa_roi.npy"
    laa_label_id = 61

    if os.path.exists(crop_out_name):
        if not quiet:
            print(f"{crop_out_name} already exists - skipping")
        return True

    if verbose:
        print(f"Computing {crop_out_name}")

    if not os.path.exists(segm_name_total):
        msg = f"TotalSegmentator segmentation {segm_name_total} not found. Can not extract LAA ROI."
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
        return False
    label_img_total = io_utils.read_nifti_with_logging_cached(segm_name_total, verbose, quiet, write_log_file, output_folder)
    if label_img_total is None:
        return False

    ct_img = io_utils.read_nifti_with_logging_cached(input_file, verbose, quiet, write_log_file, output_folder)
    if ct_img is None:
        return False

    ct_np = sitk.GetArrayFromImage(ct_img)
    label_img_total_np = sitk.GetArrayFromImage(label_img_total)
    laa_mask = label_img_total_np == laa_label_id

    if np.sum(laa_mask) == 0:
        msg = f"Could not find LAA label in TotalSegmentator segmentation {segm_name_total}."
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
        return False

    spacing = label_img_total.GetSpacing()
    # We want at least 1 cubic centimeter
    min_comp_size_mm3 = 1000
    min_comp_size_vox = int(min_comp_size_mm3 / (spacing[0] * spacing[1] * spacing[2]))
    if verbose:
        print(f"Finding LAA components with min_comp_size: {min_comp_size_vox} voxels = {min_comp_size_mm3:.1f} mm^3")

    components = su.get_components_over_certain_size_as_individual_volumes(laa_mask, min_comp_size_vox, 1)
    if components is None or len(components) == 0:
        msg = f"Could not find any components of size > {min_comp_size_mm3} mm3 in {segm_in_name}"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
        return False

    laa_mask = components[0]

    com_np = measurements.center_of_mass(laa_mask)
    # Do the transpose of the coordinates (SimpleITK vs. numpy)
    com_np = [com_np[2], com_np[1], com_np[0]]
    com_phys = label_img_total.TransformIndexToPhysicalPoint([int(com_np[0]), int(com_np[1]), int(com_np[2])])

    side_length_mm = 70
    nvox = 128

    # Create the sampled image with same direction
    direction = ct_img.GetDirection()

    # Desired voxel spacing for new image
    new_spacing = [side_length_mm / nvox, side_length_mm / nvox, side_length_mm / nvox]

    com = com_phys
    dir_x = direction[0]
    dir_y = direction[4]
    dir_z = direction[8]
    # print(f"Directions: {dir_x} {dir_y} {dir_z}")
    new_origin_x = com[0] - dir_x * side_length_mm / 2
    new_origin_y = com[1] - dir_y * side_length_mm  / 2
    new_origin_z = com[2] - dir_z * side_length_mm / 2
    # print(f"New origin: {new_origin_x} {new_origin_y} {new_origin_z}")

    new_size = [nvox, nvox, nvox]
    new_image = sitk.Image(new_size, ct_img.GetPixelIDValue())
    new_image.SetOrigin([new_origin_x, new_origin_y, new_origin_z])
    new_image.SetSpacing(new_spacing)
    new_image.SetDirection(direction)

    # Make translation with no offset, since sitk.Resample needs this arg.
    translation = sitk.TranslationTransform(3)
    translation.SetOffset((0, 0, 0))

    default_value = -2048.0
    interpolator = sitk.sitkLinear
    # Create final resampled image
    resampled_image = sitk.Resample(ct_img, new_image, translation, interpolator, default_value)
    sitk.WriteImage(resampled_image, crop_out_name)

    # occ_np = sitk.GetArrayViewFromImage(resampled_image)
    # occ_np = np.flip(occ_np, 2)
    # np.save(crop_out_name_np, occ_np)

    return True


def predict_laa_with_nudf(params):
    """
    Predict the labelmap of the left atrial appendage using the NUDF model
    """
    input_file = params["input_file"]
    ts_folder = params["ts_folder"]
    output_folder = params["output_folder"]
    # scan_id = params["pure_id"]
    crops_folder = params["crops_folder"]
    segm_folder = params["segm_folder"]
    quiet = params["quiet"]
    verbose = params["verbose"]
    write_log_file = params["write_log_file"]
    device_in = params["device"]
    min_hu = params.get("minimum_laa_hu_value", 200)
    crop_name_in = f"{crops_folder}laa_roi.nii.gz"
    sdf_out_name = f"{segm_folder}laa_nudf_sdf.nii.gz"
    crop_label_out_name = f"{segm_folder}laa_nudf_label_crop.nii.gz"
    laa_label_out_name = f"{segm_folder}laa_nudf_label.nii.gz"
    segm_name_total = f"{ts_folder}total.nii.gz"

    debug = False

    if verbose:
        print(f"Computing LAA prediction with NUDF for {input_file}")
    if os .path.exists(laa_label_out_name):
        if verbose:
            print(f"{laa_label_out_name} already exists - skipping")
        return True

    device = select_device(device_in)

    crop_img = io_utils.read_nifti_with_logging_cached(crop_name_in, verbose, quiet, write_log_file, output_folder)
    if crop_img is None:
        return False

    crop_img_np = sitk.GetArrayFromImage(crop_img)
    # Flip to make compatible with NUDF input
    crop_img_np = np.flip(crop_img_np, 2)
    # HU normalization
    crop_normalized = np.clip(crop_img_np, -1000, 1000) / 1000
    # Reshape to add channel dimension
    input = np.expand_dims(crop_normalized, axis=0)

    # TODO: Get from config
    resolution = 128
    batch_points = 100000
    checkpoint_file = params["nudf_checkpoint_file"]

    nudf_predictor = NUDFPredictor(checkpoint_file, device=device, resolution=resolution, batch_points=batch_points)

    if verbose:
        print("Setting up NUDF model")
    if not nudf_predictor.setup_model():
        msg = f"Could not set up NUDF model for LAA prediction."
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
        return False

    data = {'inputs': np.array(input, dtype=np.float32)}

    if verbose:
        print("Running NUDF prediction")
    logits, _ = nudf_predictor.predict(data)

    # Export SDF
    dim = (np.round(len(logits)**(1/3)).astype(int),)*3
    numpy_3d_sdf_tensor = np.reshape(logits,dim)

    # Flip to get correct orientation
    numpy_3d_sdf_tensor = np.flip(numpy_3d_sdf_tensor, 2)
    if debug:
        img_sdf = sitk.GetImageFromArray(numpy_3d_sdf_tensor)
        img_sdf.CopyInformation(crop_img)
        sitk.WriteImage(img_sdf, sdf_out_name)

    if verbose:
        print("Generating LAA label from SDF")

    # Generate label volume
    laa_label_np = (numpy_3d_sdf_tensor < 0).astype(np.uint8)

    # Flip crop image back to original orientation
    crop_img_np = np.flip(crop_img_np, 2)
    # Should have a minimum HU to be considered part of LAA
    laa_label_np[crop_img_np <= min_hu] = 0

    spacing = crop_img.GetSpacing()
    # We want at least 1 cubic centimeter
    min_comp_size_mm3 = 1000
    min_comp_size_vox = int(min_comp_size_mm3 / (spacing[0] * spacing[1] * spacing[2]))
    if verbose:
        print(f"Finding LAA components with min_comp_size: {min_comp_size_vox} voxels = {min_comp_size_mm3:.1f} mm^3")

    components = su.get_components_over_certain_size_as_individual_volumes(laa_label_np, min_comp_size_vox, 1)
    if components is None or len(components) == 0:
        msg = f"Could not find any components of size > {min_comp_size_mm3} mm3 in {segm_in_name}"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
        return False

    laa_mask = components[0]
    laa_mask_sitk = sitk.GetImageFromArray(laa_mask.astype(np.uint8))
    laa_mask_sitk.CopyInformation(crop_img)
    if debug:
        sitk.WriteImage(laa_mask_sitk, crop_label_out_name)

    # Read total segmentation to place LAA label in correct location
    label_img_total = io_utils.read_nifti_with_logging_cached(segm_name_total, verbose, quiet, write_log_file, output_folder)
    if label_img_total is None:
        return False

    # Resample LAA label to original image size and spacing
    original_size = label_img_total.GetSize()
    original_spacing = label_img_total.GetSpacing()
    original_origin = label_img_total.GetOrigin()
    original_direction = label_img_total.GetDirection()
    new_image = sitk.Image(original_size, laa_mask_sitk.GetPixelIDValue())
    new_image.SetOrigin(original_origin)
    new_image.SetSpacing(original_spacing)
    new_image.SetDirection(original_direction)
    # Make translation with no offset, since sitk.Resample needs this arg.
    translation = sitk.TranslationTransform(3)
    translation.SetOffset((0, 0, 0))
    default_value = 0
    interpolator = sitk.sitkNearestNeighbor
    # Create final resampled image
    resampled_image = sitk.Resample(laa_mask_sitk, new_image, translation, interpolator, default_value)
    sitk.WriteImage(resampled_image, laa_label_out_name)

    return True


def do_cardiac_analysis(params):
    """
    Compute cardiac data
    """
    input_file = params["input_file"]
    ts_folder = params["ts_folder"]
    output_folder = params["output_folder"]
    scan_id = params["pure_id"]
    quiet = params["quiet"]
    verbose = params["verbose"]
    write_log_file = params["write_log_file"]
    total_in_name = f"{ts_folder}total.nii.gz"

    # Do not inherit any previous error message
    gu.clear_last_error_message()
    gu.setup_vtk_error_handling(output_folder)

    if not os.path.exists(input_file):
        msg = f"Could not find {input_file} for cardiac analysis"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
        return False

    ts_total_exists = os.path.exists(total_in_name)
    # ts_hc_exists = os.path.exists(hc_in_name)
    if not ts_total_exists:
        msg = f"Could not find TotalSegmentator segmentations {total_in_name} for cardiac analysis"
        if not quiet:
            print(msg)
        if write_log_file:
            write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
        return False

    # TODO: add exception handling later. Now we should see where the error occurs
    # try:
    success = True
    success = extract_laa_roi(params)
    if not success:
        return False
    success = predict_laa_with_nudf(params)
    if not success:
        return False
    # except Exception as e:
    #     msg = "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
    #     msg += f"Exception during cardiac analysis of {scan_id}: {str(e)}\n"
    #     msg += "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
    #     if not quiet:
    #         print(msg)
    #     if write_log_file:
    #         gu.write_message_to_log_file(base_dir=output_folder, message=msg, level="error")
    #     success = False

    if verbose:
        image_reader_cache_info = io_utils.read_nifti_with_logging_cached.cache_info()
        print(f"Image reader cache info: {image_reader_cache_info}")
    io_utils.read_nifti_with_logging_cached.cache_clear()

    return True


def computer_process(params, output_folder, process_queue, process_id):
    verbose = params["verbose"]

    while not process_queue.empty():
        q_size = process_queue.qsize()
        input_file = process_queue.get()
        if verbose:
            print(f"Process {process_id} running cardiac analysis on: {input_file} - {q_size} left")
        local_start_time = time.time()
        params = gu.set_and_create_folders(input_file, output_folder, params)
        do_cardiac_analysis(params)
        n_proc = params["num_proc_general"]
        elapsed_time = time.time() - local_start_time
        q_size = process_queue.qsize()
        est_time_left = q_size * elapsed_time / n_proc
        time_left_str = gu.display_time(int(est_time_left))
        time_elapsed_str = gu.display_time(int(elapsed_time))
        if verbose:
            print(f"=================================================================================================")
            print(f"Process {process_id} done with {input_file} - took {time_elapsed_str}.\n"
                  f"Time left {time_left_str} for {q_size} scans (if {n_proc} processes alive)")
            print(f"=================================================================================================")
        stats_folder = params["stats_folder"]
        time_stats_out = f"{stats_folder}cardiac_proc_time.txt"
        with open(time_stats_out, "w") as f:
            f.write(f"{elapsed_time}\n")

def setup_nudf_folder_and_get_checkpoint(params):
    """
    Setup NUDF model folder and get checkpoint file
    """
    verbose = params["verbose"]

    nudf_model_url = params["nudf_model_url"]
    nudf_checkpoint = params["nudf_checkpoint"]
    nudf_model_folder = params.get("nudf_model_folder", "")
    if nudf_model_folder == "":
        # Find home directory
        home_dir = str(Path.home())
        nudf_model_folder = os.path.join(home_dir, ".cardiacctexplorer", "nudf_model")
        if verbose:
            print(f"NUDF model folder not specified - setting it to default: {nudf_model_folder}")
    params["nudf_model_folder"] = nudf_model_folder

    checkpoint_file = os.path.join(nudf_model_folder, nudf_checkpoint)
    params["nudf_checkpoint_file"] = checkpoint_file
    if os.path.exists(checkpoint_file):
        if verbose:
            print(f"NUDF checkpoint file already exists: {checkpoint_file}")
        return params
    if not(os.path.exists(nudf_model_folder)):
        os.makedirs(nudf_model_folder)
    if verbose:
        print(f"Downloading NUDF model checkpoint from {nudf_model_url} to {checkpoint_file}")
    load_url(f"{nudf_model_url}{nudf_checkpoint}", model_dir=nudf_model_folder, file_name=nudf_checkpoint,
             progress=verbose)
    if not os.path.exists(checkpoint_file):
        msg = f"Could not download NUDF model checkpoint to {checkpoint_file}"
        if not params["quiet"]:
            print(msg)
        if params["write_log_file"]:
            gu.write_message_to_log_file(base_dir=params["output_folder"], message=msg, level="error")
        return False
    return params



def cardiac_analysis(in_files, output_folder, params):
    verbose = params["verbose"]
    params = setup_nudf_folder_and_get_checkpoint(params)

    num_processes = params.get("num_proc_general", 1)

    if verbose:
        print(f"Computing cardiac data with max {num_processes} processes on {len(in_files)} files. Output to {output_folder}")

    # no need to spawn more processes than files
    num_processes = min(num_processes, len(in_files))

    files_to_process = []
    for fname in in_files:
        params = gu.set_and_create_folders(fname, output_folder, params)
        stats_folder = params["stats_folder"]
        stats_file = f"{stats_folder}/cardiac_statistics.json"
        # vis_file = f"{vis_folder}aorta_visualization.png"
        # if not os.path.exists(stats_file) or not os.path.exists(vis_file):
        #     files_to_process.append(fname)
        files_to_process.append(fname)
    if verbose:
        print(f"Found {len(files_to_process)} files to compute cardiac analysis out of {len(in_files)} files")

    in_files = files_to_process
    if len(in_files) == 0:
        if verbose:
            print("No files to compute cardiac analysis on  - all done!")
        return

    # no need to do multiprocessing for one file
    if len(in_files) == 1:
        input_file = in_files[0].strip()
        if verbose:
            print(f"Running cardiac analysis on: {input_file}")
        local_start_time = time.time()
        params = gu.set_and_create_folders(input_file, output_folder, params)
        do_cardiac_analysis(params)
        elapsed_time = time.time() - local_start_time
        elapsed_time_str = gu.display_time(int(elapsed_time))
        if verbose:
            print(f"Done with {input_file} - took {elapsed_time_str}")
        stats_folder = params["stats_folder"]
        time_stats_out = f"{stats_folder}cardiac_proc_time.txt"
        with open(time_stats_out, "w") as f:
            f.write(f"{elapsed_time}\n")
    else:
        process_queue = mp.Queue()
        for idx in in_files:
            input_file = idx.strip()
            process_queue.put(input_file)

        if verbose:
            print(f"Starting {num_processes} processes")

        processes = []
        for i in range(num_processes):
            p = mp.Process(target=computer_process, args=(params, output_folder, process_queue, i + 1))
            p.start()
            processes.append(p)

        if verbose:
            print("Waiting for processes to finish")
        for p in processes:
            p.join()
