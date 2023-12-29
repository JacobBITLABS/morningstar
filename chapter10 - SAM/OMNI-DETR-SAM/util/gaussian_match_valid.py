import torch
from gaussians import NWD

A = torch.tensor([[0.9923, 0.1426, 0.0154, 0.0455],
                  [0.8890, 0.2108, 0.0659, 0.0455],
                  [0.9253, 0.2677, 0.0659, 0.0487],
                 ], dtype=torch.float32)

B = torch.tensor([
                  [1, 2, 3, 4],
                  [1, 2, 3, 4],
                ], dtype=torch.float32)


#@torch.jit.script
def gaussian_loss_matcher_v2(src_boxes, permuted_target_list, constant=16):
    # print("Gaussian Loss V2")
    source = src_boxes # predicted boxes
    target = permuted_target_list # list of expected predicted boxes

    # # Can be eempty
    # if torch.numel(permuted_target_list) == 0:
    #     t_target = []
          
    #     for i, x in enumerate(source.cpu().numpy()):
    #         # print(x)
    #         print("adding penalty")
    #         t_target.append(20) 

    #     t = torch.tensor(t_target)
    #     res = torch.reshape(t, (len(t_target), 1))
    #     div_f = len(src_boxes)
    #     res.div(div_f)

    #     # print("************************")
    #     # print(res)
    #     # print("************************")
    #     res1 = torch.tensor(res).to('cuda')
    #     return res1

    # # Build target matrix
    # t_target = []
    # for pred_class in target:
    #     # DETR no obj hack
    #     if pred_class != 0:
    #         class_gaussian = list(filter(lambda cat: cat['id'] == pred_class, CLASS_BOX_GAUSSIANS))
    #         xb_mean = class_gaussian[0]['xs_mean'] 
    #         yb_mean = class_gaussian[0]['ys_mean'] 
    #         wb_std = class_gaussian[0]['width_mean']
    #         hb_std = class_gaussian[0]['height_mean']
            
    #         t_target.append([xb_mean, yb_mean, wb_std, hb_std])
    #     else:
    #         t_target.append([0.02, 0.02, 0.02, 0.02])

    # # construct tensor and send to GPUs/device
    # t_target = torch.tensor(t_target).to('cuda')

    t_target = B
    # print("[T_TARGET]", t_target)    

    #print("t_target: ", t_target)
    # Split each component [[x, y, w, h], [x1, y1, w1, h1]] -> ([x, x1], [y, y1], [w, w1], [h, h1])
    source_x, source_y, source_w, source_h = torch.split(source, 1, dim=1) # unpack
    target_x, target_y, target_w, target_h = torch.split(t_target, 1, dim=1) # unpack

    #print("source_x")
    #print(source_x)
    #print("target_x")
    #print(target_x)

    # d1
    # print("Subtract Rows Pairwise ")
    # X
    source_x_sub_target_x = source_x.unsqueeze(0) - target_x.unsqueeze(1) # subtract each N[i][j] with all M[i][j]
    # print(source_x_sub_target_x)
    # Y
    source_y_sub_target_y = source_y.unsqueeze(0) - target_y.unsqueeze(1) # subtract each N[i][j] with all M[i][j]
    # print(source_y_sub_target_y)
    # X, Y -> [[X, Y]]
    source_y_sub_target_y_cancat = torch.cat((source_x_sub_target_x, source_y_sub_target_y), 2)
    # print(source_y_sub_target_y_cancat)
    # calculate norm
    norm_source_target_subtracted = torch.norm(source_y_sub_target_y_cancat, dim=2, p=2)
    # print(norm_source_target_subtracted)
    d1 = norm_source_target_subtracted.pow(2) # should be pairwise added with d2

    # print("d1: ")
    # print(d1)

    # d2
    #print()
    #print("d2:")
    # print(source_split_wh)
    # print(target_split_wh)
    # W
    source_w_add_target_w = source_w.unsqueeze(0) + target_w.unsqueeze(1)
    # Divided by 2
    source_w_add_target_w_div = torch.div(source_w_add_target_w, 2.0)
    # H 
    source_h_add_target_h = source_h.unsqueeze(0) + target_h.unsqueeze(1)
    # Divided by 2
    source_h_add_target_h_div = torch.div(source_h_add_target_h, 2.0)
    # Add (h1+h2)/2 + (w1+w2)/2
    # print(source_w_add_target_w_div)
    # print(source_h_add_target_h_div)
    A_add_B = (source_w_add_target_w_div + source_h_add_target_h_div)
    # sqrt(h1*h2)
    source_h_mul_source_h = (source_h.unsqueeze(0) * target_h.unsqueeze(1)).sqrt_()
    # sqrt(w1*w2)
    source_h_mul_target_h = (source_w.unsqueeze(0) * target_w.unsqueeze(1)).sqrt_()
    # sqrt(h1*h2) + sqrt(w1*w2)s
    source_h_mul_target_h_add_source_h_mul_target_h = source_h_mul_source_h + source_h_mul_target_h
    #print(source_h_mul_target_h_add_source_h_mul_target_h)
    # times 2
    source_h_mul_target_h_add_source_h_mul_target_h_mul_two = source_h_mul_target_h_add_source_h_mul_target_h.mul(2)
    # print(source_h_mul_target_h_add_source_h_mul_target_h_mul_two)
    d2 = A_add_B - source_h_mul_target_h_add_source_h_mul_target_h_mul_two
    # reshape d2
    d2 = torch.reshape(d2, (len(permuted_target_list), -1))
    
    # print("d2")
    # print(torch.reshape(d2, (len(permuted_target_list), -1)))

    #  mapping = math.exp(-(np.sqrt(d1.item() + d2.item()) / constant))
    d1_add_d2 = d1 + d2
    # distance: absolute value 
    d1_add_d2_abs = torch.abs(d1_add_d2)
    # sqrt
    d1_add_d2_sqrt = d1_add_d2_abs.sqrt_()
    # div with constant
    d1_add_d2_sqrt_div = d1_add_d2_sqrt.div(constant)
    # exp
    d1_add_d2_sqrt_div_exp = torch.exp(d1_add_d2_sqrt_div * -1)

    return d1_add_d2_sqrt_div_exp.T



def main():
    print("Validator")
    
    gLoss = gaussian_loss_matcher_v2(A, B)
    print("Results Matcher")
    print(gLoss)

    print("NWD")
    a = A.numpy()
    b = B.numpy()
    print("[0][0]")
    m1_x, m1_y, w1, h1 = a[0]
    m2_x, m2_y, w2, h2 = b[0]
    nwd = NWD(m1_x, m1_y, w1, h1, m2_x, m2_y, w2, h2, C=16)
    print(a[0], ", ", b[0], "NWD: ", nwd)
    print("[0][1]")
    m1_x, m1_y, w1, h1 = a[0]
    m2_x, m2_y, w2, h2 = b[1]
    nwd = NWD(m1_x, m1_y, w1, h1, m2_x, m2_y, w2, h2, C=16)
    print(a[0], ", ", b[1], "NWD: ", nwd)
    print()
    print("[1][0]")
    m1_x, m1_y, w1, h1 = a[1]
    m2_x, m2_y, w2, h2 = b[0]
    nwd = NWD(m1_x, m1_y, w1, h1, m2_x, m2_y, w2, h2, C=16)
    print(a[1], ", ", b[0], "NWD: ", nwd)
    print("[1][1]")
    m1_x, m1_y, w1, h1 = a[1]
    m2_x, m2_y, w2, h2 = b[1]
    nwd = NWD(m1_x, m1_y, w1, h1, m2_x, m2_y, w2, h2, C=16)
    print(a[1], ", ", b[1], "NWD: ", nwd)
    print()
    print("[2][0]")
    m1_x, m1_y, w1, h1 = a[2]
    m2_x, m2_y, w2, h2 = b[0]
    nwd = NWD(m1_x, m1_y, w1, h1, m2_x, m2_y, w2, h2, C=16)
    print(a[2], ", ", b[0], "NWD: ", nwd)
    print("[2][1]")
    m1_x, m1_y, w1, h1 = a[2]
    m2_x, m2_y, w2, h2 = b[1]
    nwd = NWD(m1_x, m1_y, w1, h1, m2_x, m2_y, w2, h2, C=16)
    print(a[2], ", ", b[1], "NWD: ", nwd)


main()