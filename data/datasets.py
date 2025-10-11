import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import random
import numpy as np
import PIL
import pandas as pd
import rasterio
from utils.normalise_s2 import normalise_s2
import torchvision.transforms as transforms


# surpress no georef warnings
import warnings
from rasterio.errors import NotGeoreferencedWarning

# Suppress only NotGeoreferencedWarnings
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)



class cv_dataset(Dataset):
    def __init__(self, img_dir,phase="train"):
        import rasterio
        assert phase in ["train","val"]

        self.img_dir = img_dir
        self.img_names = os.listdir(img_dir)
        self.img_names = sorted(self.img_names)
        
        # clip train/test
        if phase=="train":
            self.img_names = self.img_names[:-200]
            random.shuffle(self.img_names)
        if phase=="val":
            self.img_names = self.img_names[-200:]

        
        self.transform_original = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
        ])

        self.transform_small = transforms.Compose([
            #transforms.CenterCrop((300, 300)),
            transforms.Resize((75, 75)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        # get image
        # crop to lowest dimension squared
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        img = Image.open(img_path).convert('RGB')
        width, height = img.size
        lowest_dimension = min(width, height)
        transform_preprocess = transforms.Compose([
            transforms.CenterCrop((lowest_dimension, lowest_dimension)),
            ])
        
        # turn to tnesor
        img_original = transform_preprocess(img)

        # resize to 300X300 and turn to tensor
        img_original = self.transform_original(img)

        # bring to -1..+1
        img_original = (img_original*2)-1
        
        # if not at least 300*300, interpolate
        if img_original.shape[1]<300 or img_original.shape[2]<300:
            img_original = img_original.unsqueeze(0)
            img_original = torch.nn.functional.interpolate(img_original, size=(300,300), mode='bilinear')
            img_original = img_original.squeeze(0)

        # create LR version
        img_small =  torch.nn.functional.interpolate(img_original.unsqueeze(0), size=(75,75), mode='bilinear').squeeze(0)

        # return Images
        return img_small,img_original
    
# Define torch dataset Class
class SPOT6(Dataset):
    def __init__(self,dataset_root,phase="train",spectral_matching="histogram",
                     spatial_matching="identity",sen2_amount=1,return_type="interpolated"):
        """  Inputs:
                - path to root of x,y,.pkl
                - phase: str either train or test
                - spectral_matching: str either ["histogram","moment","normalize"]
                - spatial_matching: str either ["grid","shiftnet","none"]
                - sen2_amount: int deciding how many sen2 are returned
                - return_type: either ["interpolated","cross_sensor"]
        """
        
        # set args as class props
        self.dataset_root = dataset_root
        self.spectral_matching = spectral_matching
        self.spatial_matching = spatial_matching
        self.phase = phase
        self.sen2_amount = sen2_amount
        self.return_type = return_type
        # check if all args are valid
        assert self.spectral_matching in ["histogram","moment","identity"]
        assert self.spatial_matching in ["grid","shiftnet","identity"]
        assert self.phase in ["train","test","val"]
        assert self.return_type in ["interpolated","cross_sensor","interpolated_matched"]
        
        # load dataset file
        dataset = pd.read_pickle(os.path.join(self.dataset_root,"dataset_stratified.pkl"))
        
        # first get rif of time deltas that are too far away
        dataset = self.filter_by_time_delta(dataset,lowest_delta=10)
        
        # randomly set amount to train/test/val
        dataset = self.change_random_type(dataset,new_type=self.phase,fraction=0.2)

        # filter no acquisiions:
        dataset = dataset[dataset['sen2_acquisitions'].apply(lambda x: bool(x))]
        
        # filter for phase
        self.dataset = dataset[dataset["type"] == self.phase]
        
        # load spectral matching from file
        import utils.dataloader_utils as dataloader_utils
        self.spectral_matching = getattr(dataloader_utils,self.spectral_matching)
        # load spatial_matching from file
        self.spatial_matching = getattr(dataloader_utils,self.spatial_matching)

    def filter_by_time_delta(self,df,lowest_delta=None):
        """
        Filters a DataFrame based on a 'lowest_delta' threshold, getting rid of rows where time delta is too big.
        Parameters:
        - df (pd.DataFrame): Input DataFrame with 'sen2_acquisitions' column.
        - lowest_delta (int, optional): Threshold for filtering.

        Returns:
        - pd.DataFrame: Filtered DataFrame.
        """
        # Function to extract the lowest integer key from each dictionary   
        def get_lowest_key(row):
                keys = row['sen2_acquisitions'].keys()
                return int(min(map(int, keys))) if keys else None
        # Apply the function and create the new column
        df['lowest_delta'] = df.apply(get_lowest_key, axis=1)

        if lowest_delta!=None:
            df = df[df['lowest_delta'] <= lowest_delta]

        return(df)
        

    def change_random_type(self,df,new_type="val",fraction=0.15):
        """
        Randomly changes the 'type' field of a DataFrame's subset to a new value.

        Parameters:
        - df (pandas.DataFrame): Input DataFrame with a 'type' field.
        - new_type (str, optional): New type to assign; default is 'val'.
        - fraction (float, optional): Fraction of rows to change; default is 0.15.

        Returns:
        - pandas.DataFrame: Modified DataFrame.
        """
        df = df.reset_index()
        # Calculate the number of rows to change
        num_rows = len(df)
        num_to_change = int(fraction * num_rows)

        # Randomly select index positions to change
        import random
        random.seed(42)
        indices_to_change = random.sample(range(num_rows), num_to_change)

        # Change the 'type' value at those indices to 'val'
        df.loc[indices_to_change, 'type'] = new_type
        return df

    def __len__(self):
        return(len(self.dataset))
    
    def __getitem__(self,idx):
        
        # get row from dataset file
        dataset_row = self.dataset.iloc[idx]
        
        # get HR image path
        hr_path = os.path.join(self.dataset_root,dataset_row["spot6_image"])
        # get LR image paths by adding dataset root path infront of dict values
        lr_paths  = {key: os.path.join(self.dataset_root,value) for key, value in dataset_row["sen2_acquisitions"].items()}
        
        # read HR image
        hr = torch.Tensor(rasterio.open(hr_path).read())
        hr = (hr/255.).float()

        # This is MISR stuff
        # read lr image(s) into list
        lr = [] # empty list to hold images
        lr_days_delta = list(lr_paths.keys()) # get sorted list of images by time
        for count,value in enumerate(lr_days_delta):
            # stop loop when wanted amount of images reached
            if count>=self.sen2_amount: 
                break
            #read lr image from path and append to list
            lr_im = torch.Tensor(rasterio.open(lr_paths[value]).read())
            lr.append(lr_im)
        # if there are not as many sen2 images as requested, repeat last entry in the list
        while len(lr) < self.sen2_amount: lr.append(lr[-1])


        # return LR version if interpoaltion is wanted
        if self.return_type=="interpolated":
            lr = torch.nn.functional.interpolate(hr.unsqueeze(0),size=(75,75),mode="bilinear",antialias=True).squeeze(0)
            lr = normalise_s2(lr,stage="norm")
            hr = normalise_s2(hr,stage="norm")
            if self.sen2_amount>1:
                # stack tensor x times to simulate MISR
                lr = torch.stack([lr]*self.sen2_amount)
                #lr = lr.view(self.sen2_amount*lr.shape[1], lr.shape[2], lr.shape[3]) # goes from 4,3,75,75 to 12,75,75
                assert lr.shape[0]==self.sen2_amount*3
            return lr,hr
        
                # return LR version if interpoaltion is wanted
        if self.return_type=="interpolated_matched":
            # hist match both SPOT5 HR and LR to Sen2 LR to be in S2 domain
            lr_sen2_int = lr[0]
            lr_sen2_int = lr_sen2_int/10000
            lr_sen2_int = torch.nn.functional.interpolate(lr_sen2_int.unsqueeze(0),size=(300,300),mode="bilinear",antialias=True).squeeze(0)
            # create LR version from HR SPOT6
            lr = torch.nn.functional.interpolate(hr.unsqueeze(0),size=(75,75),mode="bilinear",antialias=True).squeeze(0)
            # match both SPOT6 LR and HR to Sen2 spectral charachteristics
            lr,hr = self.spectral_matching(lr_sen2_int,lr),self.spectral_matching(lr_sen2_int,hr)
            # normalze, both are in S2 spectral range already
            lr = normalise_s2(lr,stage="norm")
            hr = normalise_s2(hr,stage="norm")
            # apply gaussian blur to lr version
            apply_blur=False
            if apply_blur:
                sigma = random.uniform(0.4,0.6)
                gaussian_blur = transforms.GaussianBlur(3, sigma=sigma)
                lr = gaussian_blur(lr)

            # if more wanted, stack fake LR images
            if self.sen2_amount>1:
                # stack tensor x times to simulate MISR
                lr = torch.stack([lr]*self.sen2_amount)
                # add noise to tensor
                # Generate Gaussian noise
                noise = torch.randn_like(lr)
                std = 0.075
                scaled_noise = std * noise
                lr = lr + scaled_noise
                lr = torch.clamp(lr, -1, 1)

            return lr,hr
        

        if self.return_type=="cross_sensor": 
            # perform standard preprocessing
            lr = [(tensor/10000.).float() for tensor in lr]
            
            # perform spectral matching to closest sen2 image in time
            hr = self.spectral_matching(lr[0],hr)
            
            # perform spatial matching to match each LR to HR - DEACTIVATED FOR NOW
            #lr = [self.spatial_matching(lr_im,hr) for lr_im in lr]

            # last check to ensure dimensions are correct
            if hr.shape!= torch.Size([3, 300, 300]):
                hr = torch.nn.functional.interpolate(hr.unsqueeze(0), size=(300, 300), mode='bilinear', align_corners=True,antialias=True).squeeze(0)
            for v,lr_ in enumerate(lr):
                if lr_.shape!= torch.Size([3, 75, 75]):
                    lr_ = torch.nn.functional.interpolate(lr_.unsqueeze(0), size=(75, 75), mode='bilinear', align_corners=True,antialias=True).squeeze(0)
                    lr[v] = lr_
            
            # stack tensors
            lr = torch.stack(lr)

            # perform normalization
            lr = normalise_s2(lr,stage="norm")
            hr = normalise_s2(hr,stage="norm")
            
            return lr,hr
        
        # return None if no valid selection has been made
        return None,None
    

def dataset_selector(config):
    if config.Data.dataset_type=="cv":
        # train
        ds_train = cv_dataset(config.Data.data_dir,phase="train")
        dl_train = DataLoader(ds_train, batch_size=config.Data.train_batch_size, shuffle=True,
                              num_workers=config.Data.num_workers,prefetch_factor=config.Data.prefetch_factor,
                              drop_last=True,sen2_amount=config.Data.sen2_amount)
        # val
        ds_val = cv_dataset(config.Data.data_dir,phase="val")
        dl_val = DataLoader(ds_val, batch_size=config.Data.val_batch_size, shuffle=False,
                            num_workers=config.Data.num_workers,prefetch_factor=config.Data.prefetch_factor,
                            drop_last=True,sen2_amount=config.Data.sen2_amount)
        # get datamodule
        pl_datamodule = create_pl_datamodule(dl_train, dl_val)
        return(pl_datamodule)
    
    if config.Data.dataset_type=="SPOT6":
        # train
        ds_train = SPOT6(config.Data.data_dir,phase="train",sen2_amount=config.Data.sen2_amount,return_type=config.Data.return_type,spectral_matching=config.Data.spectral_matching)
        dl_train = DataLoader(ds_train, batch_size=config.Data.train_batch_size, shuffle=True,num_workers=config.Data.num_workers,prefetch_factor=config.Data.prefetch_factor,drop_last=True)
        # val
        ds_val = SPOT6(config.Data.data_dir,phase="val",sen2_amount=config.Data.sen2_amount,return_type=config.Data.return_type,spectral_matching=config.Data.spectral_matching)
        dl_val = DataLoader(ds_val, batch_size=config.Data.val_batch_size, shuffle=False,num_workers=config.Data.num_workers,prefetch_factor=config.Data.prefetch_factor,drop_last=True)
        # get datamodule
        pl_datamodule = create_pl_datamodule(dl_train, dl_val)
        return(pl_datamodule)
    
    if config.Data.dataset_type=="S2_6b":
        desired_20m_order = ["B05_20m","B06_20m","B07_20m","B8A_20m","B11_20m","B12_20m"]
        sr_factor = getattr(config.Generator, "scaling_factor", 4)         # e.g., set to 8 for ×8
        hr_size   = (512,512)
        from utils.S2_6b_ds import S2SAFEDataset
        ds_train = S2SAFEDataset(
            phase="train",
            manifest_json="/data3/S2_20m/s2_safe_manifest_20m.json",
            group_by="granule",
            group_regex=r".*?/GRANULE/([^/]+)/IMG_DATA/.*",
            bands_keep=desired_20m_order,
            band_order=desired_20m_order,
            dtype="float32",
            hr_size=hr_size,
            sr_factor=sr_factor,
        )
        dl_train = DataLoader(ds_train, batch_size=config.Data.train_batch_size, shuffle=True
                                ,num_workers=config.Data.num_workers,prefetch_factor=config.Data.prefetch_factor
                                ,drop_last=True)
        ds_val =  S2SAFEDataset(
            phase="val",
            manifest_json="/data3/S2_20m/s2_safe_manifest_20m.json",
            group_by="granule",
            group_regex=r".*?/GRANULE/([^/]+)/IMG_DATA/.*",
            bands_keep=desired_20m_order,
            band_order=desired_20m_order,
            dtype="float32",
            hr_size=hr_size,
            sr_factor=sr_factor,
        )
        dl_val = DataLoader(ds_val, batch_size=config.Data.val_batch_size, shuffle=False
                                ,num_workers=config.Data.num_workers,prefetch_factor=config.Data.prefetch_factor
                                ,drop_last=True)
        pl_datamodule = create_pl_datamodule(dl_train, dl_val)
        return(pl_datamodule)


def create_pl_datamodule(train_loader,val_loader):
    import pytorch_lightning as pl
    class pl_datamodule(pl.LightningDataModule):
        def __init__(self, train_loader, val_loader):
            super().__init__()
            self.train_loader = train_loader
            self.val_loader = val_loader
        def train_dataloader(self):
            return self.train_loader
        def val_dataloader(self):
            return self.val_loader
        def prepare_data(self):
            pass
        def setup(self, stage=None):
            pass
        

    datamodule = pl_datamodule(train_loader,val_loader)
    return(datamodule)
