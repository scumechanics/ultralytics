# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Ultralytics neural network modules.

This module provides access to various neural network components used in Ultralytics models, including convolution
blocks, attention mechanisms, transformer components, and detection/segmentation heads.

Examples:
    Visualize a module with Netron
    >>> from ultralytics.nn.modules import Conv
    >>> import torch
    >>> import subprocess
    >>> x = torch.ones(1, 128, 40, 40)
    >>> m = Conv(128, 128)
    >>> f = f"{m._get_name()}.onnx"
    >>> torch.onnx.export(m, x, f)
    >>> subprocess.run(f"onnxslim {f} {f} && open {f}", shell=True, check=True)  # pip install onnxslim
"""

from .block import (
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    CIB,
    DFL,
    ELAN1,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    A2C2f,
    AConv,
    ADown,
    Attention,
    BNContrastiveHead,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    ContrastiveHead,
    GhostBottleneck,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    MaxSigmoidAttnBlock,
    Proto,
    RepC3,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    SCDown,
    TorchVision,
)
from .conv import (
    CBAM,
    ChannelAttention,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostConv,
    Index,
    LightConv,
    RepConv,
    SpatialAttention,
)
from .head import (
    OBB,
    Classify,
    Detect,
    LRPCHead,
    Pose,
    RTDETRDecoder,
    Segment,
    WorldDetect,
    YOLOEDetect,
    YOLOESegment,
    v10Detect,
)
from .transformer import (
    AIFI,
    MLP,
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
    LayerNorm2d,
    MLPBlock,
    MSDeformAttn,
    TransformerBlock,
    TransformerEncoderLayer,
    TransformerLayer,
)
from .ess import ESSamp
from .eucb import EUCB
from .dcafe import DCAFE
from .dcafe2 import DCAFE_V2,DCAFE_Lite,DCAFE_Hybrid
from .converse import Converse2D
from .DFF import *
from .BiFPN import Bi_FPN, BiFPN_Layer, ScalarWeightedFusion, ChannelAttentionFusion
from .iEMA import *
from .SD_FAPB    import C2PSA_SDFAPB
from .SG_CAFusion import SG_CAFusion, SaliencyAuxLoss
from .DSAM import *
from .CLCA import *
from .SECA import *
from .LGAM import *
from .DSAttention import *
from .DSLAM import *
from .SELA import *
from .DEMAttention import *
from .MLAttention import *
from .LCAttention import *
from .EPSA import *
from .defect_focus import *
from .ImprovedLGAM import *
from .BinaryAttention import *
from .EfficientViM import *
from .RCSOSA import *
from .rcsosa_v2 import *
from .BFAM import *
from .GhostModule import *
from .EUCB2 import *
from .MANet import *
# from .GCBlock import *
from .RepNCSPELAN4 import RepNCSPELAN4_high,RepNCSPELAN4_low
from .RepNCSPELAN42 import RepNCSPELAN4
# from .GELAN import RepNCSPELAN4, SPPELAN
from .Slimneck import GSConv, VoVGSCSP
from .C3k2_ARConv import C3k2_ARConv
from .RFB import BasicRFB 
from .RefConv import  RefConv 
from .MultiOrderGatedAggregation import MultiOrderGatedAggregation  
from .MogaSubBlock import MogaSubBlock  
from .C3k2_DeepDBB import C3k2_DeepDBB 
from .LRSA import LRSA
from .C3K2_CBSA import C3K2_CBSA
__all__ = (
    "C3k2_DFF_1",
    "C3k2_DFF_2",
    "Converse2D",
    "DCAFE",
    "DCAFE_V2",
    "DCAFE_Lite",
    "DCAFE_Hybrid",
    "EUCB",
    "ESSamp",
    "AIFI",
    "C1",
    "C2",
    "C2PSA",
    "C3",
    "C3TR",
    "CBAM",
    "CIB",
    "DFL",
    "ELAN1",
    "MLP",
    "OBB",
    "PSA",
    "SPP",
    "SPPELAN",
    "SPPF",
    "A2C2f",
    "AConv",
    "ADown",
    "Attention",
    "BNContrastiveHead",
    "Bottleneck",
    "BottleneckCSP",
    "C2f",
    "C2fAttn",
    "C2fCIB",
    "C2fPSA",
    "C3Ghost",
    "C3k2",
    "C3x",
    "CBFuse",
    "CBLinear",
    "ChannelAttention",
    "Classify",
    "Concat",
    "ContrastiveHead",
    "Conv",
    "Conv2",
    "ConvTranspose",
    "DWConv",
    "DWConvTranspose2d",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "Detect",
    "Focus",
    "GhostBottleneck",
    "GhostConv",
    "HGBlock",
    "HGStem",
    "ImagePoolingAttn",
    "Index",
    "LRPCHead",
    "LayerNorm2d",
    "LightConv",
    "MLPBlock",
    "MSDeformAttn",
    "MaxSigmoidAttnBlock",
    "Pose",
    "Proto",
    "RTDETRDecoder",
    "RepC3",
    "RepConv",
    "RepNCSPELAN4",
    "RepVGGDW",
    "ResNetLayer",
    "SCDown",
    "Segment",
    "SpatialAttention",
    "TorchVision",
    "TransformerBlock",
    "TransformerEncoderLayer",
    "TransformerLayer",
    "WorldDetect",
    "YOLOEDetect",
    "YOLOESegment",
    "v10Detect",
    'Bi_FPN',
    'BiFPN_Layer',
    'ScalarWeightedFusion',
    'ChannelAttentionFusion',
    'C2PSA_SDFAPB', 'SG_CAFusion', 'SaliencyAuxLoss'
)
