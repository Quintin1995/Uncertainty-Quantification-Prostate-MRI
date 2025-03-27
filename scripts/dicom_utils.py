
import os
import glob
from pathlib import Path
from typing import Tuple
from pydicom import dcmread
from sqlite3 import connect


def get_shapes_from_dicom(dicom_dir: Path) -> Tuple:
    """Get the zero-pad shape and image space crop shape from the first dicom file in the directory."""

    # We'll be working with str instead of Path
    dicom_dir = str(dicom_dir)

    assert os.path.isdir(dicom_dir), f"dicom_dir must be a directory, got: {dicom_dir}"

    pattern = os.path.join(dicom_dir, '*')
    print(f"\tLooking for dicom files in pattern: {pattern}")

    dcm_fpaths = glob.glob(pattern)
    print(f"\tNumber of detected dicom files in: {len(dcm_fpaths)}")

    first_slice_fpath = dcm_fpaths[0]

    ds = dcmread(first_slice_fpath)

    zero_pad_shape = (ds.Rows*2, ds.Columns*2)
    image_space_crop = (ds.Rows, ds.Columns)
    print(f"\tCalculated ZERO-PAD shape: {zero_pad_shape}, based on rows, cols of the DICOM: {image_space_crop}")

    return zero_pad_shape, image_space_crop


def find_t2_tra_dir_in_study_dir(study_dir: Path) -> Path:
    """
    Description: Find the T2 TSE TRA directory in the study directory.
    Args:
        study_dir (Path): The study directory.
    Returns:
        Path: The T2 TSE TRA directory.
    """
    for seq_dir in study_dir.iterdir():
        if "tse2d1" in seq_dir.name.lower():  # Using lower() for case-insensitive match
            # List the files in the seq_dir and take the first and read with pydicom
            dcm_files = list(seq_dir.glob('*'))
            if dcm_files:  # Ensure there's at least one file to read
                dcm = dcmread(dcm_files[0])
                # If ProtocolName contains T2, TSE, and TRA, case insensitive, then we return the directory
                protocol_name = dcm.ProtocolName.lower()  # Make comparison case-insensitive
                if "t2" in protocol_name and "tse" in protocol_name and "tra" in protocol_name:
                    return seq_dir  # Returning the directory, not the DICOM object
    return None  # Return None if no matching directory is found


def find_respective_dicom_dir(pat_id: str, source_dir: Path = None, db_fpath: Path = None) -> Tuple[Path, Path]:
    """
    Description: Find the respective DICOM directory for the given patient ID based on the kspace acquisition date.
    Args:
        pat_id (str): The patient ID.
        source_dir (Path): The source directory where the data is stored.
        db_fpath (Path): The path to the database file.
    Returns:
        Tuple[Path, Path]: The DICOM directory and the NIFTI directory.
    """
    
    anon_id = pat_id.split('_')[-1]
    print(f"Db patient ID: {anon_id}")
    
    conn = connect(str(db_fpath))
    cursor = conn.cursor()
    try:
        # Query to retrieve all MRI dates for the given patient ID
        query = "SELECT mri_date FROM kspace_dset_info WHERE anon_id = ? ORDER BY mri_date"
        cursor.execute(query, (anon_id,))
        results = cursor.fetchall()
        print(f"\tResults from the query: {results}")
                
        if results:
            for result in results:        # loops over each study date found in the database and checks if it there is a matching dicom directory
                mri_date = str(result[0]) 
                mri_date_str = "{}-{}-{}".format(mri_date[:4], mri_date[4:6], mri_date[6:]) # Convert YYYY-MM-DD
                study_dir_path_dcms = source_dir / 'data' / pat_id / 'dicoms' / mri_date_str  # Construct expected DICOM dir path
                study_dir_path_niftis = source_dir / 'data' / pat_id / 'niftis' / mri_date_str  # Construct expected NIFTI dir path
                print(f"\tChecking for study dir: {study_dir_path_dcms}")
                print(f"\tChecking for study dir: {study_dir_path_niftis}")
                
                if study_dir_path_dcms.exists() and study_dir_path_niftis.exists():
                    print(f"\t\tMatching dicom dir found for patient ID based on kspace acquisition date {pat_id} in {source_dir / 'data' / pat_id / 'dicoms'}")
                    print(f"\t\tMatching nifti dir found for patient ID based on kspace acquisition date {pat_id} in {source_dir / 'data' / pat_id / 'niftis'}")
                    t2_tra_dcm_dir = find_t2_tra_dir_in_study_dir(study_dir_path_dcms)

                    # We have found the correct dicom link that immediatly to the to correct nifti and return that file too.
                    t2_tra_nif_fpath = Path(str(t2_tra_dcm_dir).replace('dicoms', 'niftis') + '.nii.gz')
                    
                    return t2_tra_dcm_dir, t2_tra_nif_fpath
                else:
                    print(f"\tWarning: No matching study directory found for patient ID {pat_id} in {source_dir / 'data' / pat_id / 'dicoms'}")
                
            # If no matching directory is found after checking all dates
            print(f"\tWarning: No matching study directory found for patient ID {pat_id}")
            raise Exception(f"No matching study directory found for patient ID {pat_id} in {source_dir / 'data' / pat_id / 'dicoms'}")
        else:
            print(f"\tWarning: No MRI date found for patient ID {pat_id}")
            raise Exception(f"No MRI date found for patient ID {pat_id}")
    finally:
        conn.close()