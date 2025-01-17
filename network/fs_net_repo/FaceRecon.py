# follow FS-Net
import torch.nn as nn
import network.fs_net_repo.gcn3d as gcn3d

import network.fs_net_repo.PointTransformerV2 as ptv2
import torch
import torch.nn.functional as F
from absl import app
import absl.flags as flags

FLAGS = flags.FLAGS

# global feature num : the channels of feature from rgb and depth
# grid_num : the volume resolution

class FaceRecon(nn.Module):
    def __init__(self):
        super(FaceRecon, self).__init__()
        self.neighbor_num = FLAGS.gcn_n_num
        self.support_num = FLAGS.gcn_sup_num

        # PointTrnasformer V2

        self.recon_num = 3
        self.face_recon_num = FLAGS.face_recon_c
        dim_fuse = sum([128, 128, 256, 256, 512, FLAGS.obj_c])
        #3D Geometric Feature Extractor : PointTransformerV2
        self.ptv2 = ptv2.PointTransformerV2Network()

        # 16: total 6 categories, 256 is global feature
        self.conv1d_block = nn.Sequential(
            nn.Conv1d(dim_fuse, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        self.recon_head = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, self.recon_num, 1),
        )

        self.face_head = nn.Sequential(
            nn.Conv1d(FLAGS.feat_face + 3, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, self.face_recon_num, 1),  # Relu or not?
        )

    def forward(self,
                vertices: "tensor (bs, vetice_num, 3)",
                cat_id: "tensor (bs, 1)",
                ):
        obj_num = 1
        # Input
        data_dict = {
            'coord': vertices,
            'offset': torch.tensor([[1028]]).cuda()
        }
        global_point_feat, point_feat = self.ptv2(obj_num,data_dict)
        #global_point_feat = (1028,1282) , point_feat = (1028,512)
        #  concate feature
        bs, vertice_num, _ = vertices.size()
        fm_4 = point_feat.unsqueeze(dim=0)
        f_global = fm_4.max(1)[0]  # (bs, f)
        feat = global_point_feat.unsqueeze(dim=0)
        feat_face_re = f_global.view(bs, 1, f_global.shape[1]).repeat(1, feat.shape[1], 1).permute(0, 2, 1)
        # feat is the extracted per pixel level feature
        conv1d_input = feat.permute(0, 2, 1)  # (bs, fuse_ch, vertice_num)
        conv1d_out = self.conv1d_block(conv1d_input)
        recon = self.recon_head(conv1d_out)
        # average pooling for face prediction
        feat_face_in = torch.cat([feat_face_re, conv1d_out, vertices.permute(0, 2, 1)], dim=1)
        face = self.face_head(feat_face_in)
        return recon.permute(0, 2, 1), face.permute(0, 2, 1), feat


def main(argv):
    classifier_seg3D = FaceRecon()

    points = torch.rand(2, 1000, 3)
    import numpy as np
    obj_idh = torch.ones((2, 1))
    obj_idh[1, 0] = 5
    '''
    if obj_idh.shape[0] == 1:
        obj_idh = obj_idh.view(-1, 1).repeat(points.shape[0], 1)
    else:
        obj_idh = obj_idh.view(-1, 1)

    one_hot = torch.zeros(points.shape[0], 6).scatter_(1, obj_idh.cpu().long(), 1)
    '''
    recon, face, feat = classifier_seg3D(points, obj_idh)
    face = face.squeeze(0)
    t = 1



if __name__ == "__main__":
    print(1)
    from config.config import *
    app.run(main)


