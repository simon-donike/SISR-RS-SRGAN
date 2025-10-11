# Package Imports
import torch
from omegaconf import OmegaConf
from multiprocessing import freeze_support

# local imports
from model.SRGAN import SRGAN_model

# set torch device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Run Main Function
if __name__ == '__main__':
    # required for Multprocessing on Windows
    freeze_support()

    # General
    torch.set_float32_matmul_precision('medium')
    # load config
    config_file_path = "config_predict.yaml"
    config = OmegaConf.load(config_file_path)

    #############################################################################################################
    """ LOAD MODEL """
    #############################################################################################################
    # load rpetrained or instanciate new
    model = SRGAN_model(config_file_path=config_file_path)
    # load pretrained weights
    model.load_state_dict(torch.load(config.Model.ckpt_path)["state_dict"],strict=False)
    # model settings
    model = model.eval()
    model = model.to(device)

    #############################################################################################################
    """ Set up SR-S2 utils - start SR """
    #############################################################################################################
    # get SR2 utils and create
    from utils.sr_s2_utils.main import windowed_SR_and_saving
    folder_path = "data/sen2_tiles/S2A_MSIL2A_20230729T100031_N0509_R122_T33TUG_20230729T134559.SAFE/"
    # create SR object
    sr_obj = windowed_SR_and_saving(folder_path, window_size=(75, 75),keep_lr_stack=True)

    # start SR of RGB bands
    sr_obj.start_super_resolution(band_selection="10m",model=model,forward_call="predict")



