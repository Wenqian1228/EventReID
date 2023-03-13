from __future__ import absolute_import

from models.PSTA import PSTA
from models.PSTA_img_event_cat import PSTA_img_event_cat
from models.PSTA_img_event_deform1 import PSTA_img_event_deform1

from models.PSTA_img_event_deform1128 import PSTA_img_event_deform1128
from models.HiCMD_img import HiCMD_Net
from models.OSNet_img import OSNet
from models.SRS_img import SRS_Net
from models.STMN_img import STMN_Net
from models.TransReID_img import TransReID_Net
from models.CCReID_img import CC_Net
from models.MCNL_img import MCNL

######################################################################
from models.HiCMD_img_event_deform1 import HiCMD_Net_deform1
from models.OSNet_img_event_deform1 import OSNet_deform1
from models.STMN_img_event_deform1 import STMN_Net_deform1
from models.TransReID_img_event_deform1 import TransReID_Net_deform1
from models.SRS_img_event_deform1 import SRS_Net_deform1
from models.CCReID_img_event_deform1 import CC_Net_deform1


#############################################################################
from models.HiCMD_img_event_deform2 import HiCMD_Net_deform2
from models.OSNet_img_event_deform2 import OSNet_deform2
from models.STMN_img_event_deform2 import STMN_Net_deform2
from models.TransReID_img_event_deform2 import TransReID_Net_deform2
from models.SRS_img_event_deform2 import SRS_Net_deform2
from models.CCReID_img_event_deform2 import CC_Net_deform2
from models.PSTA_img_event_deform2 import PSTA_img_event_deform2
#############################################################################
from models.HiCMD_img_event_deform3 import HiCMD_Net_deform3
from models.OSNet_img_event_deform3 import OSNet_deform3
from models.STMN_img_event_deform3 import STMN_Net_deform3
from models.TransReID_img_event_deform3 import TransReID_Net_deform3
from models.SRS_img_event_deform3 import SRS_Net_deform3
from models.CCReID_img_event_deform3 import CC_Net_deform3
from models.PSTA_img_event_deform3 import PSTA_img_event_deform3
from models.MCNL_img_event_deform3 import MCNL_deform3
#############################################################################123
from models.OSNet_SNN_event import OSNet_SNN
from models.OSNet_SNN_event1 import OSNet_SNN1
from models.PSTA_SNN1 import PSTA_SNN1
from models.PSTA_SNN2 import PSTA_SNN2
from models.PSTA_SNN3 import PSTA_SNN3
from models.slayersnn_exam import NetworkBasic
from models.OSNet_SNN_event2 import OSNet_SNN2
from models.PSTA_SNN_deform1 import PSTA_SNN_deform1
from models.PSTA_SNN_deform2 import PSTA_SNN_deform2
from models.PSTA_SNN_deform3 import PSTA_SNN_deform3
from models.PSTA_SNN2 import PSTA_SNN2
from models.PSTA_SNN3 import PSTA_SNN3
from models.PSTA_SNN4 import PSTA_SNN4
from models.GRL_img import GRL_img
from models.STGCN_img import STGCN_img
from models.SINet_img import SINet
from models.BiCnet_img import BiCNet
from models.TCLNet_img import TCLNet
from models.AP3D_img import AP3D_img
from models.PSTA_visual import PSTA_visual
from models.OSNet_visual import OSNet_visual
from models.OSNet_img_event_visual import OSNet_img_event_visual


__factory = {
    'PSTA' : PSTA,
    'HiCMD_img':HiCMD_Net,
    'OSNet_img':OSNet,
    'SRS_img':SRS_Net,
    'STMN_img':STMN_Net,
    'TransReID_img':TransReID_Net,
    'CCReID_img':CC_Net,
    'MCNL_img':MCNL,
    ######################################################
    'GRL_img':GRL_img,
    'STGCN_img':STGCN_img,
    'SINet':SINet,
    'BiCNet':BiCNet,
    # 'TCLNet':TCLNet,
    'AP3D_img':AP3D_img,
############################################################################
    'PSTA_img_event_cat':PSTA_img_event_cat,
    'PSTA_img_event_deform1':PSTA_img_event_deform1,
    'PSTA_img_event_deform2':PSTA_img_event_deform2,
    'PSTA_img_event_deform1128':PSTA_img_event_deform1128,

    'HiCMD_Net_deform1':HiCMD_Net_deform1,
    'OSNet_deform1':OSNet_deform1,
    'STMN_Net_deform1':STMN_Net_deform1,
    'TransReID_Net_deform1':TransReID_Net_deform1,
    'SRS_Net_deform1':SRS_Net_deform1,
    'CC_Net_deform1':CC_Net_deform1,
############################################################################
    'HiCMD_Net_deform2':HiCMD_Net_deform2,
    'OSNet_deform2':OSNet_deform2,
    'STMN_Net_deform2':STMN_Net_deform2,
    'TransReID_Net_deform2':TransReID_Net_deform2,
    'SRS_Net_deform2':SRS_Net_deform2,
    'CC_Net_deform2':CC_Net_deform2,

############################################################################
    'HiCMD_Net_deform3':HiCMD_Net_deform3,
    'OSNet_deform3':OSNet_deform3,
    'STMN_Net_deform3':STMN_Net_deform3,
    'TransReID_Net_deform3':TransReID_Net_deform3,
    'SRS_Net_deform3':SRS_Net_deform3,
    'CC_Net_deform3':CC_Net_deform3,
    'PSTA_img_event_deform3':PSTA_img_event_deform3,
    'MCNL_deform3':MCNL_deform3,

############################################################################
    'OSNet_SNN':OSNet_SNN,

############################################################################
    'OSNet_SNN1':OSNet_SNN1,
    'PSTA_SNN1':PSTA_SNN1,
    'PSTA_SNN2':PSTA_SNN2,
    'PSTA_SNN3':PSTA_SNN3,
    'SNN_exam':NetworkBasic,
    'PSTA_SNN_deform1':PSTA_SNN_deform1,
    'PSTA_SNN_deform2':PSTA_SNN_deform2,
    'PSTA_SNN_deform4':PSTA_SNN_deform3,
    'PSTA_SNN2':PSTA_SNN2,
    'PSTA_SNN3':PSTA_SNN3,
    'PSTA_SNN4':PSTA_SNN4,
############################################################################  
    'OSNet_SNN2':OSNet_SNN2,
############################################################################
    'PSTA_visual':PSTA_visual,
    'OSNet_visual':OSNet_visual,
    'OSNet_img_event_visual':OSNet_img_event_visual,

}

def get_names():
    return __factory.keys()


def init_model(name,*args,**kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model :{}".format(name))
    return __factory[name](*args,**kwargs)
