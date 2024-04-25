from monai.transforms import LoadImage
import torch
import os
import time
from multiprocessing import Process, Queue
import nibabel as nib
import glob

def read_data(filename,load_root,save_root,subj_name,count,queue=None,scaling_method=None, fill_zeroback=False):
    print("processing: " + filename, flush=True)
    path = os.path.join(load_root, filename,'MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_hp2000_clean.nii.gz')
    data, metadata = LoadImage()(path) #torch.Tensor(nib.load(path).get_fdata()) #LoadImage()(path)
    
    #change this line according to your file names
    save_dir = os.path.join(save_root,subj_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # change this line according to your dataset
    data = data[7:85, 7:103, 0:84, :]
    # width, height, depth, time
    # Inspect the fMRI file first using your visualization tool. 
    # Limit the ranges of width, height, and depth to be under 96. Crop the background, not the brain regions. 
    # Each dimension of fMRI registered to MNI space (2mm) is expected to be around 100.
    # You can do this when you load each volume at the Dataset class, including padding backgrounds to fill dimensions under 96.
   
    background = (data <=0) # change this because filtered UKB data has minus values
    
    valid_voxels = data[~background].numel()
    global_mean = data[~background].mean()
    global_std = data[~background].std()
    global_max = data[~background].max()
    # global min should be zero

    data[background] = 0

    # save volumes one-by-one in fp16 format.
    data_global = data.type(torch.float16)
    data_global_split = torch.split(data_global, 1, 3)
    for i, TR in enumerate(data_global_split):
        torch.save(TR.clone(), os.path.join(save_dir,"frame_"+str(i)+".pt"))
    
    # save global stat of fMRI volumes
    checkpoint = {
    'valid_voxels': valid_voxels,
    'global_mean': global_mean,
    'global_std': global_std,
    'global_max': global_max
    }
    torch.save(checkpoint, os.path.join(save_dir,"global_stats.pt"))

def main():
    # change two lines below according to your dataset
    dataset_name = 'HCP'
    load_root = '/global/cfs/cdirs/m4244/download_HCP_WM/HCP_download/HCP_filtered_rest_LR' # This folder should have fMRI files in nifti format with subject names. Ex) sub-01.nii.gz 
    save_root = f'/global/cfs/cdirs/m4244/download_HCP_WM/HCP_download/{dataset_name}_filtered_MNI_to_TRs_minmax'
    scaling_method = 'minmax' # choose either 'z-norm'(default) or 'minmax'.

    # make result folders
    filenames = os.listdir(load_root)
    os.makedirs(os.path.join(save_root,'img'), exist_ok = True)
    os.makedirs(os.path.join(save_root,'metadata'), exist_ok = True) # locate your metadata file at this folder 
    save_root = os.path.join(save_root,'img')
    
    finished_samples = os.listdir(save_root)
    queue = Queue() 
    count = 0
    for filename in sorted(filenames):
        subj_name = filename
        # extract subject name from nifti file. [:-7] rules out '.nii.gz'
        # we recommend you use subj_name that aligns with the subject key in a metadata file.

        expected_seq_length = 1201 # Specify the expected sequence length of fMRI for the case your preprocessing stopped unexpectedly and you try to resume the preprocessing.
        
        # change the line below according to your folder structure
        if (len(glob.glob(os.path.join(save_root,subj_name,'*.pt'))) < expected_seq_length): # preprocess if the subject folder does not exist, or the number of pth files is lower than expected sequence length. 
            try:
                count+=1
                p = Process(target=read_data, args=(filename,load_root,save_root,subj_name,count,queue,scaling_method))
                p.start()
                if count % 32 == 0: # requires more than 32 cpu cores for parallel processing
                    p.join()
            except Exception:
                print('encountered problem with'+filename)
                print(Exception)

if __name__=='__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print('\nTotal', round((end_time - start_time) / 60), 'minutes elapsed.')    
