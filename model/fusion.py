from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import OmegaConf
from utils.logging_helpers import plot_fusion
import wandb

class RecursiveNet(nn.Module):
# FROM HRResNet to fuse temporal images
    def __init__(self,in_channels=3,num_layers=4,alpha_residual=False):
        '''
        Args:
            config : dict, configuration file
        '''
        config= {
              "alpha_residual": False,
              "in_channels": 3,
              "num_layers" : 4,
              "kernel_size": 3}
        
        super(RecursiveNet, self).__init__()

        self.input_channels = config["in_channels"]
        self.num_layers = config["num_layers"]
        self.alpha_residual = config["alpha_residual"]
        kernel_size = config["kernel_size"]
        padding = kernel_size // 2

        self.fuse = nn.Sequential(
            ResidualBlock(2 * self.input_channels, kernel_size),
            nn.Conv2d(in_channels=2 * self.input_channels, out_channels=self.input_channels,
                      kernel_size=kernel_size, padding=padding),
            nn.PReLU())

    def forward(self, x, alphas=None):

        
        batch_size, nviews, channels, width, heigth = x.shape
        parity = nviews % 2
        half_len = nviews // 2
        
        while half_len > 0:
            alice = x[:, :half_len] # first half hidden states (B, L/2, C, W, H)
            bob = x[:, half_len:nviews - parity] # second half hidden states (B, L/2, C, W, H)
            bob = torch.flip(bob, [1])

            alice_and_bob = torch.cat([alice, bob], 2)  # concat hidden states accross channels (B, L/2, 2*C, W, H)
            alice_and_bob = alice_and_bob.view(-1, 2 * channels, width, heigth)
            #print(alice_and_bob.shape)
            x = self.fuse(alice_and_bob)
            x = x.view(batch_size, half_len, channels, width, heigth)  # new hidden states (B, L/2, C, W, H)

            if self.alpha_residual: # skip connect padded views (alphas_bob = 0)
                alphas_alice = alphas[:, :half_len]
                alphas_bob = alphas[:, half_len:nviews - parity]
                alphas_bob = torch.flip(alphas_bob, [1])
                x = alice + alphas_bob * x
                alphas = alphas_alice

            nviews = half_len
            parity = nviews % 2
            half_len = nviews // 2

        return torch.mean(x, 1)
    

    
class RecursiveNet_pl(pl.LightningModule):
    def __init__(self):
        super(RecursiveNet_pl, self).__init__()
        # Define the loss function (customize as needed)
        self.model = RecursiveNet()
        self.loss_function = torch.nn.MSELoss()  # Example loss function
        self.config = OmegaConf.load("config.yaml")

    def forward(self,batch,batch_idx=0):
        return(self.model.forward(batch))
    
    def training_step(self, batch, batch_idx):
        lr, _ = batch  # get data
        fused = self(lr) # fuse images
        lr_mean = torch.mean(lr,dim=1,keepdim=True).squeeze()
        loss = self.loss_function(fused, lr_mean)  # define self.loss_function as appropriate
        self.log('train/L2', loss) # log loss to wandb
        return(loss) # get loss

    def configure_optimizers(self):
        # Here you would define your optimizer (and optionally, your scheduler)
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def validation_step(self,batch,batch_idx):
        lr, hr = batch  # get data
        fused = self(lr) # fuse images
        lr_mean = torch.mean(torch.clone(lr),dim=1,keepdim=True).squeeze()
        loss = self.loss_function(fused, lr_mean)  # define self.loss_function as appropriate
        self.log('val/L2', loss)
        if batch_idx == 0:
            lr = lr[:,0,:,:,:]
            plot_fusion_img = plot_fusion(torch.clone(lr),torch.clone(fused),torch.clone(hr))
            self.logger.experiment.log({"Fusion":  wandb.Image(plot_fusion_img)})
        return(loss) # get loss
        
class ResidualBlock(nn.Module):
    """
    A residual block, comprising two convolutional blocks with a residual connection across them.
    """

    def __init__(self, n_channels=64,kernel_size=3 ):
        """
        :param kernel_size: kernel size
        :param n_channels: number of input and output channels (same because the input must be added to the output)
        """
        super(ResidualBlock, self).__init__()

        # The first convolutional block
        self.conv_block1 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                              batch_norm=True, activation='PReLu')

        # The second convolutional block
        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                              batch_norm=True, activation=None)

    def forward(self, input):
        """
        Forward propagation.

        :param input: input images, a tensor of size (N, n_channels, w, h)
        :return: output images, a tensor of size (N, n_channels, w, h)
        """
        residual = input  # (N, n_channels, w, h)
        #if input.shape[0] !=8:
        #    return input
        output = self.conv_block1(input)  # (N, n_channels, w, h)
        output = self.conv_block2(output)  # (N, n_channels, w, h)
        output = output + residual  # (N, n_channels, w, h)

        return output
    


class ConvolutionalBlock(nn.Module):
    """
    A convolutional block, comprising convolutional, BN, activation layers.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, batch_norm=False, activation=None):
        """
        :param in_channels: number of input channels
        :param out_channels: number of output channe;s
        :param kernel_size: kernel size
        :param stride: stride
        :param batch_norm: include a BN layer?
        :param activation: Type of activation; None if none
        """
        super(ConvolutionalBlock, self).__init__()

        if activation is not None:
            activation = activation.lower()
            assert activation in {'prelu', 'leakyrelu', 'tanh'}

        # A container that will hold the layers in this convolutional block
        layers = list()

        # A convolutional layer
        layers.append(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2))

        # A batch normalization (BN) layer, if wanted
        if batch_norm is True:
            layers.append(nn.BatchNorm2d(num_features=out_channels))

        # An activation layer, if wanted
        if activation == 'prelu':
            layers.append(nn.PReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.2))
        elif activation == 'tanh':
            layers.append(nn.Tanh())

        # Put together the convolutional block as a sequence of the layers in this container
        self.conv_block = nn.Sequential(*layers)

    def forward(self, input):
        """
        Forward propagation.

        :param input: input images, a tensor of size (N, in_channels, w, h)
        :return: output images, a tensor of size (N, out_channels, w, h)
        """
        #if input.shape[0]!=8:
        #    return(input)
        output = self.conv_block(input)  # (N, out_channels, w, h)

        return output