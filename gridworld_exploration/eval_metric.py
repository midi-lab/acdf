import torch


def get_dsm_sss_errors (z0, z1, s0, s1):
    ## Type 1: DSM (different states merged)
    ## Type 2: SSS (same state separated)
    type1_err=0
    type2_err=0
    abs_acc=0
    abs_err=0

    # s0 = torch.tensor(s0)
    # s1 = torch.tensor(s1)
    for i in range(z0.shape[0]):

        Z = z0[i] == z1[i]
        Z = Z.long()
        # Z_comp = Z[0] * Z[1] * Z[2] * Z[3]
        Z_comp = torch.prod(Z)

        S = s0[i] == s1[i]
        S = S.long()
        # S_comp = S[0] * S[1]
        S_comp = torch.prod(S)

        if Z_comp and (1-S_comp):
            # Error 1: Merging states which should not be merged
            type1_err += 1

        if (1-Z_comp) and S_comp :
            #Error 2: Did not merge states which should be merged
            type2_err += 1

        if Z_comp and S_comp : 
            abs_acc += 1

        if (1 - Z_comp) and (1 - S_comp):
            abs_err += 1

    type1_err = type1_err/z0.shape[0] * 100
    type2_err = type2_err/z0.shape[0] * 100

    abs_acc = abs_acc/z0.shape[0] * 100

    abs_err = abs_acc/z0.shape[0] * 100

    return type1_err, type2_err, abs_acc, abs_err