import os
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from .data_preprocess_and_load.datasets import S1200, ABCD, UKB, Dummy
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from .parser import str2bool

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

class fMRIDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # generate splits folder
        split_dir_path = f'./data/splits/{self.hparams.dataset_name}'

        if self.hparams.downstream_task == 'tfMRI_3D':
            split_dir_path += '/tfMRI_3D'

            for prefix in ['nback', 'SST', 'MID']:
                if self.hparams.cope.startswith(prefix):
                    split_dir_path += '/'+prefix
                    break
        else:
            split_dir_path += '/metadata'

        os.makedirs(split_dir_path, exist_ok=True)
        self.split_file_path = os.path.join(split_dir_path, f"split_fixed_{self.hparams.dataset_split_num}.txt")
        
        self.setup()

        #pl.seed_everything(seed=self.hparams.data_seed)

    def get_dataset(self):
        if self.hparams.dataset_name == "Dummy":
            return Dummy
        elif self.hparams.dataset_name == "S1200":
            return S1200
        elif self.hparams.dataset_name == "ABCD":
            return ABCD
        elif 'UKB' in self.hparams.dataset_name:
            return UKB
        else:
            raise NotImplementedError

    def convert_subject_list_to_idx_list(self, train_names, val_names, test_names, subj_list):
        #subj_idx = np.array([str(x[0]) for x in subj_list])
        subj_idx = np.array([str(x[1]) for x in subj_list])
        S = np.unique([x[1] for x in subj_list])
        # print(S)
        # print('unique subjects:',len(S))  
        train_idx = np.where(np.in1d(subj_idx, train_names))[0].tolist()
        val_idx = np.where(np.in1d(subj_idx, val_names))[0].tolist()
        test_idx = np.where(np.in1d(subj_idx, test_names))[0].tolist()
        return train_idx, val_idx, test_idx
    
    def save_split(self, sets_dict):
        with open(self.split_file_path, "w+") as f:
            for name, subj_list in sets_dict.items():
                f.write(name + "\n")
                for subj_name in subj_list:
                    f.write(str(subj_name) + "\n")
    
    def determine_split_stratified(self, S, idx):
        print('making stratified split')
        #S = np.unique([x[1] for x in index_l]) #len(np.unique([x[1] for x in index_l]))
        site_dict = {x:S[x][idx] for x in S} # index 2: site_id, idex 3: data type (ABIDE1/ABIDE2)
        site_ids = np.array(list(site_dict.values()))
        #print('site_ids:',site_ids)
        #print('S:',S)
        #subjects = list(S.keys())
        
        #remove sites that has only one valid samples
        one_value_sites=[]
        values, counts = np.unique(site_ids, return_counts=True)
        # Print the value counts
        for value, count in zip(values, counts):
            # print(f"{value}: {count}") # 20,40 has one level
            if count == 1:
                one_value_sites.append(value)
                
        filtered_site_dict = {subj:site for subj,site in site_dict.items() if site not in one_value_sites}
        filtered_subjects = np.array(list(filtered_site_dict.keys()))
        filtered_site_ids = np.array(list(filtered_site_dict.values()))
        
        strat_split = StratifiedShuffleSplit(n_splits=1, test_size=1-self.hparams.train_split-self.hparams.val_split, random_state=self.hparams.dataset_split_num)
        trainval_indices, test_indices = next(strat_split.split(filtered_subjects, filtered_site_ids)) # 0.
        
        strat_split = StratifiedShuffleSplit(n_splits=1, test_size=self.hparams.val_split, random_state=self.hparams.dataset_split_num)
        train_indices, valid_indices = next(strat_split.split(filtered_subjects[trainval_indices], filtered_site_ids[trainval_indices]))
        S_train, S_val, S_test = filtered_subjects[trainval_indices][train_indices], filtered_subjects[trainval_indices][valid_indices], filtered_subjects[test_indices]
        
        self.save_split({"train_subjects": S_train, "val_subjects": S_val, "test_subjects": S_test})
        return S_train, S_val, S_test
    
    def determine_split_randomly(self, S):
        S = list(S.keys())
        S_train = int(len(S) * self.hparams.train_split)
        S_val = int(len(S) * self.hparams.val_split)
        S_train = np.random.choice(S, S_train, replace=False)
        remaining = np.setdiff1d(S, S_train) # np.setdiff1d(np.arange(S), S_train)
        S_val = np.random.choice(remaining, S_val, replace=False)
        S_test = np.setdiff1d(S, np.concatenate([S_train, S_val])) # np.setdiff1d(np.arange(S), np.concatenate([S_train, S_val]))
        # train_idx, val_idx, test_idx = self.convert_subject_list_to_idx_list(S_train, S_val, S_test, self.subject_list)
        self.save_split({"train_subjects": S_train, "val_subjects": S_val, "test_subjects": S_test})
        return S_train, S_val, S_test
    
    def load_split(self):
        subject_order = open(self.split_file_path, "r").readlines()
        subject_order = [x[:-1] for x in subject_order]
        train_index = np.argmax(["train" in line for line in subject_order])
        val_index = np.argmax(["val" in line for line in subject_order])
        test_index = np.argmax(["test" in line for line in subject_order])
        train_names = subject_order[train_index + 1 : val_index]
        val_names = subject_order[val_index + 1 : test_index]
        test_names = subject_order[test_index + 1 :]
        return train_names, val_names, test_names

    def prepare_data(self):
        # This function is only called at global rank==0
        return
    
    # filter subjects with metadata and pair subject names with their target values (+ sex)
    def make_subject_dict(self):
        # output: {'subj1':[target1,target2],'subj2':[target1,target2]...}
        img_root = os.path.join(self.hparams.image_path, 'img')
        final_dict = dict()
        if self.hparams.dataset_name == "S1200":
            subject_list = os.listdir(img_root)
            meta_data = pd.read_csv(os.path.join(self.hparams.image_path, "metadata", "HCP_1200_gender.csv"))
            meta_data_residual = pd.read_csv(os.path.join(self.hparams.image_path, "metadata", "HCP_1200_precise_age.csv"))
            meta_data_all = pd.read_csv(os.path.join(self.hparams.image_path, "metadata", "HCP_1200_all.csv"))
            if self.hparams.downstream_task == 'sex': task_name = 'Gender'
            elif self.hparams.downstream_task == 'age': task_name = 'age'
            elif self.hparams.downstream_task == 'int_total': task_name = 'CogTotalComp_AgeAdj'
            elif self.hparams.downstream_task == 'tfMRI' : task_name = 'tfMRI'
            elif self.hparams.downstream_task == 'tfMRI_3D' : task_name = 'tfMRI_3D'
            else: raise NotImplementedError()
            
            if 'tfMRI' in task_name: 
                beta_map_list = os.listdir(task_path)

                for subject in subject_list:
                    subject_name = subject
                    # subject_name : XXXXXX (6 digits)
                    if subject_name in beta_map_list:
                        if task_name == 'tfMRI_3D':
                            if self.hparams.contrast_motion_corrected:
                                motion_type = '1st_lev_with_motion_conf'
                            else:
                                motion_type = '1st_lev_event_only'
                            target_path =  os.path.join(task_path, subject_name,f'{subject_name}_{os.path.basename(task_path)}_{motion_type}_{cope}.nii.gz')
                            sex=0 # dummy variable, since we do not need this variable
                            if not os.path.exists(target_path):
                                continue
                            final_dict[subject]=[sex,target_path]
                        elif task_name == 'tfMRI':
                            raise NotImplementedError
            else:
                if self.hparams.downstream_task == 'sex':
                    meta_task = meta_data[['Subject',task_name]].dropna()
                elif self.hparams.downstream_task == 'age':
                    meta_task = meta_data_residual[['subject',task_name,'sex']].dropna()
                    #rename column subject to Subject
                    meta_task = meta_task.rename(columns={'subject': 'Subject'})
                elif self.hparams.downstream_task == 'int_total':
                    meta_task = meta_data[['Subject',task_name,'Gender']].dropna()  

                for subject in subject_list:
                    if int(subject) in meta_task['Subject'].values:
                        if self.hparams.downstream_task == 'sex':
                            target = meta_task[meta_task["Subject"]==int(subject)][task_name].values[0]
                            target = 1 if target == "M" else 0
                            sex = target
                        elif self.hparams.downstream_task == 'age':
                            target = meta_task[meta_task["Subject"]==int(subject)][task_name].values[0]
                            sex = meta_task[meta_task["Subject"]==int(subject)]["sex"].values[0]
                            sex = 1 if sex == "M" else 0
                        elif self.hparams.downstream_task == 'int_total':
                            target = meta_task[meta_task["Subject"]==int(subject)][task_name].values[0]
                            sex = meta_task[meta_task["Subject"]==int(subject)]["Gender"].values[0]
                            sex = 1 if sex == "M" else 0
                        elif self.hparams.downstream_task.startswith('WM_'):
                            target = meta_task[meta_task["Subject"]==int(subject)][task_name].values[0]
                            sex = meta_task[meta_task["Subject"]==int(subject)]["Gender"].values[0]
                            sex = 1 if sex == "M" else 0
                        final_dict[subject]=[sex,target]

        elif self.hparams.dataset_name == "ABCD":
            subject_list = [subj for subj in os.listdir(img_root)]
            
            meta_data = pd.read_csv(os.path.join(self.hparams.image_path, "metadata", "ABCD_phenotype_total.csv"))
            if self.hparams.downstream_task == 'sex': task_name = 'sex'
            elif self.hparams.downstream_task == 'age': task_name = 'age'
            elif self.hparams.downstream_task == 'int_total': task_name = 'nihtbx_totalcomp_uncorrected'
            elif self.hparams.downstream_task == 'int_fluid': task_name = 'nihtbx_fluidcomp_uncorrected'
            elif self.hparams.downstream_task == 'ASD': task_name = 'ASD_label'
            elif self.hparams.downstream_task == 'ADHD': task_name = 'ADHD_label_robust' # This is new label # 'ADHD_label' is KSAD parent 
            elif self.hparams.downstream_task == 'tfMRI' : task_name = 'tfMRI'
            elif self.hparams.downstream_task == 'tfMRI_3D' : task_name = 'tfMRI_3D'
            else: raise ValueError('downstream task not supported')

            if 'tfMRI' in task_name: 
                beta_map_list = os.listdir(self.hparams.task_path)

                for subject in subject_list:
                    subject_name = 'sub-'+subject
                    # subject_name : sub-XXXXX
                    if subject_name in beta_map_list:
                        if task_name == 'tfMRI_3D':
                            target_path =  os.path.join(self.hparams.task_path, subject_name,f'{subject_name}_{self.hparams.cope}.nii.gz')
                            sex=0 # dummy variable, since we do not need this variable
                            
                            if not os.path.exists(target_path):
                                continue
                            final_dict[subject]=[sex,target_path]
                        elif task_name == 'tfMRI':
                            raise NotImplementedError

            else:
                if self.hparams.downstream_task == 'sex':
                    meta_task = meta_data[['subjectkey',task_name]].dropna()
                else:
                    meta_task = meta_data[['subjectkey',task_name,'sex']].dropna()
                
                for subject in subject_list:
                    if subject in meta_task['subjectkey'].values:
                        target = meta_task[meta_task["subjectkey"]==subject][task_name].values[0]
                        sex = meta_task[meta_task["subjectkey"]==subject]["sex"].values[0]
                        final_dict[subject]=[sex,target]
            
        elif "UKB" in self.hparams.dataset_name:
            if self.hparams.downstream_task == 'sex': task_name = 'sex'
            elif self.hparams.downstream_task == 'age': task_name = 'age'
            elif self.hparams.downstream_task == 'income' : task_name = 'income'
            elif self.hparams.downstream_task == 'int_fluid' : task_name = 'fluid'
            elif self.hparams.downstream_task == 'tfMRI' : task_name = 'tfMRI'
            elif self.hparams.downstream_task == 'tfMRI_3D' : task_name = 'tfMRI_3D'
            else: raise ValueError('downstream task not supported')

            if 'tfMRI' in task_name:    
                beta_map_list = [subj[:7] for subj in os.listdir(self.hparams.task_path) if subj.endswith('20249_2_0')]
                subject_list = [subj for subj in os.listdir(img_root) if subj.endswith('20227_2_0')]
                for subject in subject_list:
                    # subject : 1000246_20227_2_0 or 1000246_20227_3_0
                    subject_name = subject[:7]
                    # subject_name = 1000246
                    if subject_name in beta_map_list:
                        if task_name == 'tfMRI_3D':
                            target_path =  os.path.join(self.hparams.task_path, subject_name+'_20249_2_0',f'zstat{self.hparams.cope}_MNI_space.nii.gz') # 3D
                            sex=0 # dummy variable, since we do not need this variable 
                        final_dict[subject_name]=[sex,target_path]
            else:
                meta_data = pd.read_csv(os.path.join(self.hparams.image_path, "metadata", "UKB_phenotype_gps_fluidint.csv"))
                if task_name == 'sex':
                    meta_task = meta_data[['eid',task_name]].dropna()
                else:
                    meta_task = meta_data[['eid',task_name,'sex']].dropna()

                for subject in os.listdir(img_root):
                    if subject.endswith('20227_2_0') and (int(subject[:7]) in meta_task['eid'].values):
                        target = meta_task[meta_task["eid"]==int(subject[:7])][task_name].values[0]
                        sex = meta_task[meta_task["eid"]==int(subject[:7])].values[0]
                        final_dict[str(subject[:7])] = [sex,target]
                    else:
                        continue 
        return final_dict

    def setup(self, stage=None):
        # this function will be called at each devices
        Dataset = self.get_dataset()
        params = {
                "root": self.hparams.image_path,
                "sequence_length": self.hparams.sequence_length,
                "stride_between_seq": self.hparams.stride_between_seq,
                "stride_within_seq": self.hparams.stride_within_seq,
                "with_voxel_norm": self.hparams.with_voxel_norm,
                "downstream_task": self.hparams.downstream_task,
                "shuffle_time_sequence": self.hparams.shuffle_time_sequence,
                "input_type": self.hparams.input_type,
                "input_scaling_method" : self.hparams.input_scaling_method,
                "label_scaling_method" : self.hparams.label_scaling_method,
                "dtype":'float16', 
                "time_as_channel": self.hparams.time_as_channel,
                "use_ic": self.hparams.use_ic,
                "input_features_path": self.hparams.input_features_path,
                "input_mask_path": self.hparams.input_mask_path} 

        subject_dict = self.make_subject_dict()
        if os.path.exists(self.split_file_path):
            train_names, val_names, test_names = self.load_split()
        else:
            train_names, val_names, test_names = self.determine_split_randomly(subject_dict)
        
        # if self.hparams.bad_subj_path:
        #     bad_subjects = open(self.hparams.bad_subj_path, "r").readlines()
        #     for bad_subj in bad_subjects:
        #         bad_subj = bad_subj.strip()
        #         if bad_subj in list(subject_dict.keys()):
        #             print(f'removing bad subject: {bad_subj}')
        #             del subject_dict[bad_subj]
        
        if self.hparams.limit_training_samples:
            train_names = np.random.choice(train_names, size=self.hparams.limit_training_samples, replace=False, p=None)
        train_dict = {key: subject_dict[key] for key in train_names if key in subject_dict}
        val_dict = {key: subject_dict[key] for key in val_names if key in subject_dict}
        test_dict = {key: subject_dict[key] for key in test_names if key in subject_dict}
        
        self.train_dataset = Dataset(**params,subject_dict=train_dict, train=True) # use_augmentations=False,
        # load train mean/std of target labels to val/test dataloader
        self.val_dataset = Dataset(**params,subject_dict=val_dict, train=False) 
        self.test_dataset = Dataset(**params,subject_dict=test_dict, train=False) 
        
        print("number of train_subj:", len(train_dict))
        print("number of val_subj:", len(val_dict))
        print("number of test_subj:", len(test_dict))
        print("length of train_idx:", len(self.train_dataset.data))
        print("length of val_idx:", len(self.val_dataset.data))  
        print("length of test_idx:", len(self.test_dataset.data))
        
        # DistributedSampler is internally called in pl.Trainer
        def get_params(train):
            return {
                "batch_size": self.hparams.batch_size if train else self.hparams.eval_batch_size,
                "num_workers": self.hparams.num_workers,
                "drop_last": True,
                "pin_memory": False,
                "persistent_workers": False if self.hparams.dataset_name == 'Dummy' else (train and (self.hparams.strategy == 'ddp')),
                "shuffle": train
            }
        self.train_loader = DataLoader(self.train_dataset, **get_params(train=True))
        self.val_loader = DataLoader(self.val_dataset, **get_params(train=False))
        self.test_loader = DataLoader(self.test_dataset, **get_params(train=False))
        

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        # return self.val_loader
        # currently returns validation and test set to track them during training
        return [self.val_loader, self.test_loader]

    def test_dataloader(self):
        return self.test_loader

    def predict_dataloader(self):
        return self.test_dataloader()

    @classmethod
    def add_data_specific_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=True, formatter_class=ArgumentDefaultsHelpFormatter)
        group = parser.add_argument_group("DataModule arguments")
        group.add_argument("--dataset_split_num", type=int, default=1) # dataset split, choose from 1, 2, or 3
        group.add_argument("--label_scaling_method", default="standardization", choices=["minmax","standardization"], help="normalization strategy for a regression task (mean and std are automatically calculated using train set)")
        group.add_argument("--input_scaling_method", type=str, default='minmax',choices=["minmax","znorm_zeroback","znorm_minback",'none'],
                          help="normalization strategy for input sub-sequences (added in SwiFT v2), specify none if your preprocessing pipeline is based on v1")
        group.add_argument("--image_path", default="default")
        group.add_argument("--task_path", default="default")
        group.add_argument("--mask_filename", default="")
        group.add_argument("--mask_input", action='store_true')
        group.add_argument("--cope", type=str,help='1,2,5 for UKB, nback_run_1_2_back_vs_0_back_zmap for ABCD')
        group.add_argument("--input_type", default="rest",choices=['rest','task'],help='refer to datasets.py')
        group.add_argument("--train_split", default=0.7, type=float)
        group.add_argument("--val_split", default=0.15, type=float)
        group.add_argument("--batch_size", type=int, default=4)
        group.add_argument("--sequence_length", type=int, default=30)
        group.add_argument("--eval_batch_size", type=int, default=16)
        group.add_argument("--img_size", nargs="+", default=[96, 96, 96, 20], type=int, help="image size (adjust the fourth dimension according to your --sequence_length argument)")
        group.add_argument("--stride_between_seq", type=float, default=1, help="skip some fMRI volumes between fMRI sub-sequences")
        group.add_argument("--stride_within_seq", type=float, default=1, help="skip some fMRI volumes within fMRI sub-sequences")
        group.add_argument("--num_workers", type=int, default=8)
        group.add_argument("--with_voxel_norm", type=str2bool, default=False)
        group.add_argument("--from_jeongbo", action='store_true')
        group.add_argument("--shuffle_time_sequence", action='store_true')
        group.add_argument("--time_as_channel", action='store_true')
        group.add_argument("--limit_training_samples", type=int, default=None, help="use if you want to limit training samples")
        
        group.add_argument("--use_ic", action='store_true')
        group.add_argument("--input_features_path", type=str, default="default")
        group.add_argument("--input_mask_path", type=str, default="default")
        return parser
