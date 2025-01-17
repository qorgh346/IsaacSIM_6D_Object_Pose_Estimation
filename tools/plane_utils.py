import torch

def get_plane(pc, pc_w):
    # min least square
    n = pc.shape[0]
    A = torch.cat([pc[:, :2], torch.ones([n, 1], device=pc.device)], dim=-1)
    b = pc[:, 2].view(-1, 1)
    W = torch.diag(pc_w)
    WA = torch.mm(W, A)
    ATWA = torch.mm(A.permute(1, 0), WA)
    # ATWA_1 = torch.inverse(ATWA)
    # edit by song
    # print('fuxk u', ATWA)
    try:
        ATWA_1 = torch.linalg.inv(ATWA)
    except:
        ATWA[0][0] = ATWA[0][0] + 0.1
        ATWA[1][1] = ATWA[1][1] + 0.1
        ATWA[2][2] = ATWA[2][2] + 0.1
        ATWA_1 = torch.linalg.inv(ATWA)
    Wb = torch.mm(W, b)
    ATWb = torch.mm(A.permute(1, 0), Wb)
    X = torch.mm(ATWA_1, ATWb)
    # return dn
    dn_up = torch.cat([X[0] * X[2], X[1] * X[2], -X[2]], dim=0),
    dn_norm = X[0] * X[0] + X[1] * X[1] + 1.0
    dn = torch.nan_to_num(dn_up[0] / dn_norm)

    normal_n = torch.nan_to_num(dn / torch.norm(dn))
    for_p2plane = X[2] / torch.sqrt(dn_norm)

    # normal_n_shape = normal_n.shape
    # normal_n = normal_n[~torch.any(normal_n.isnan(), dim=-1)]
    # normal_n = normal_n.reshape(normal_n.shape[0], *normal_n_shape[1:])
    #
    # dn_shape = dn.shape
    # dn = dn[~torch.any(dn.isnan(), dim=1)]
    # dn = dn.reshape(dn.shape[0], *dn_shape[1:])
    #
    # for_p2plane_shape = for_p2plane.shape
    # for_p2plane = for_p2plane[~torch.any(for_p2plane.isnan(), dim=1)]
    # for_p2plane = for_p2plane.reshape(for_p2plane.shape[0], *for_p2plane_shape[1:])
    # print('normal_n is', normal_n)
    # print('dn is', dn)
    # print('for_p2plane is', for_p2plane)

    return normal_n, dn, for_p2plane

def get_plane_parameter(pc, pc_w):
    # min least square
    n = pc.shape[0]
    A = torch.cat([pc[:, :2], torch.ones([n, 1], device=pc.device)], dim=-1)
    b = pc[:, 2].view(-1, 1)
    W = torch.diag(pc_w)
    WA = torch.mm(W, A)
    ATWA = torch.mm(A.permute(1, 0), WA)
    ATWA_1 = torch.inverse(ATWA)
    Wb = torch.mm(W, b)
    ATWb = torch.mm(A.permute(1, 0), Wb)
    X = torch.mm(ATWA_1, ATWb)
    return X