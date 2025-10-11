
# For this to work, you need tacoreader and rasterio installed
# !pip install tacoreader rasterio --upgrade --no-deps
from huggingface_hub import hf_hub_download
import tacoreader

# DL from HF hub (you need to have git-lfs installed for this to work) - select out dir!
dataset = hf_hub_download("tacofoundation/SEN2NAIPv2", "sen2naipv2-histmatch", repo_type="dataset", local_dir=".")