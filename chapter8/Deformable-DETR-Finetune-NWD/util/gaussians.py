from imp import source_from_cache
from turtle import width
from numpy import std
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
# import numpy as np
import cupy as np
from numpy import linalg as LA
import scipy.stats as stats 
import statistics
import math
from PIL import Image
import torch

"""
Compute BBox gaussians from the labelled data available to use as box-loss
"""
# list containng a dict for each class's 2D Gaussian for boxes
# we expose this one publically for the loss-computation in Criterition 
CLASS_BOX_GAUSSIANS = []
training_img_num = 0

def get_cat_ids(coco: COCO):
    """ Get category ids from anno file - wrapper """
    # Category IDs.
    cat_ids = coco.getCatIds()
    print(f"Number of Unique Categories: {len(cat_ids)}")
    print("Category IDs:")
    print(cat_ids)  # The IDs are not necessarily consecutive.
    
    return cat_ids

def print_cats(categories):
    """ Print categories for debug purpose """
    nms=[cat['name'] for cat in categories]
    print('COCO categories: \n{}\n'.format(' '.join(nms)))

def NormalizeData(data):
    """Normalize data - distribution sum to 1 """
    return (data - np.min(data)) / (np.max(data) - np.min(data))

"""
OLD IMPLEMENTATION
"""
def OLD_Gaussian_Wasserstein_Distance(m1_x, m1_y, w1, h1, m2_x, m2_y, w2, h2):
    """
    2nd Wasserstein Distance
    """
    my_a = np.array([m1_x, m1_y])               # define my_1
    sigma_a = np.array([[w1/2, 0], [0, h1/2]])  # covariance matrix
    
    my_b = np.array([m2_x, m2_y])               # define my_2
    sigma_b = np.array([[w2/2, 0], [0, h2/2]])  # covariance
    
    # d1 - my
    my_a_subtract_my_b = np.subtract(my_a, my_b)                    # subtract my_1 from my_2
    norm_my_a_subtract_my_b = LA.norm(my_a_subtract_my_b)           # norm
    d1 = norm_my_a_subtract_my_b ** 2  # squared
    # print("d1: ", d1)
    
    # Sigma - Std. Dev
    d2 = ((h1 + h2) / 2) + ((w1 + w2) / 2) - 2 * (math.sqrt(h1 * h2) + math.sqrt(w1 * w2))
    # print("d2: ", d2)
  
    # 2nd Wasserstein Distance
    w_2 = abs(d1 + d2) # distance non-negative
    
    return w_2

def OLD_NWD(m1_x, m1_y, w1, h1, m2_x, m2_y, w2, h2, C=16):
    """
    Normalized 2nd Wasserstein Distance
    > Then, we found that when changing 𝐶 in a certain range (from 8 to 24), the value of AP waves marginally and is much higher than baseline
    """
    w_2 = OLD_Gaussian_Wasserstein_Distance(m1_x, m1_y, w1, h1, m2_x, m2_y, w2, h2)
    NWD = math.exp(-(np.sqrt(w_2)/C))

    if math.isnan(NWD): # this is only for inspecting -> change to an assert
            NWD = 1e5

    # print("NWD old: ", NWD)
    return NWD

def OLD_gaussian_loss(src_boxes, tgt_box):
    """ Takes the src boces and class list
        E.g.: src_box[0] boxes class with id indeces from permuted_target_list[0]
        Returns
    """
    training_img_num = 0
    loss_accum = 0 # loss accumulator
    assert(len(src_boxes) == len(tgt_box))
    for src_bbox, tgt_box in zip(src_boxes, tgt_box):
        src_bbox = src_bbox.tolist()
        tgt_bbox = tgt_box.tolist()
        
        src_bbox_x, src_bbox_y, src_bbox_width, src_bbox_height = src_bbox # unpack
        tgt_bbox_x, tgt_bbox_y, tgt_bbox_width, tgt_bbox_height = tgt_bbox # unpack

        # Gaussian from classs
        #class_gaussian = list(filter(lambda cat: cat['id'] == pred_class, CLASS_BOX_GAUSSIANS))
        # print("bbox: ",  bbox_x, bbox_y, bbox_height, bbox_width, " class gaussian: ", class_gaussian)
        # x_mean = class_gaussian[0]['xs_mean'] 
        # y_mean = class_gaussian[0]['ys_mean'] 
        # w_std = class_gaussian[0]['width_mean']
        # h_std = class_gaussian[0]['height_mean']

        # print("prior_box:", xb_mean, wb_std)
        # print("pred_box:", bbox_x, bbox_width, " class: ", pred_class)
        # # plot height
        # mu = yb_mean
        # variance = hb_std
        # sigma = math.sqrt(variance)
        # x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        # plot_label= str(pred_class) + "pror"
        # plt.plot(x, stats.norm.pdf(x, mu, sigma), label=plot_label)
        
        # mu1 = bbox_y
        # variance1 = bbox_height
        # sigma1 = math.sqrt(variance1)
        # plot_label1 = str(pred_class) + "pred"
        # plt.plot(x, stats.norm.pdf(x, mu1, sigma1), label=plot_label1)
        # plt.suptitle('Traning Plot')
        # plt.xlabel('z-score')
        # plt.ylabel('Probability Density')
        # plt.legend(loc="upper left")
        # plt.savefig('util/gaussian_test_output' + '/' + str(training_img_num) + '.jpg')
        # plt.clf()

        training_img_num += 1 
        # calculate NWD LOSS
        loss_accum += OLD_NWD(src_bbox_x, src_bbox_y, src_bbox_width, src_bbox_height, tgt_bbox_x, tgt_bbox_y, tgt_bbox_width, tgt_bbox_height)
    

    # print("accum loss: ", loss_accum) 
    return torch.tensor(loss_accum).to('cuda')

"""
NEW IMPLEMENTATION Faster
"""
def Gaussian_Wasserstein_Distance(m1_x, m1_y, w1, h1, m2_x, m2_y, w2, h2):
    # print("[Gaussian_Wasserstein_Distance]")
    # print("m1_x: ", m1_x)
    # print("m1_y: ", m1_y)

    # my_a = torch.tensor([m1_x, m1_y])
    my_a = torch.cat((m1_x, m1_y), dim=-1)         # [0.1992, 0.33450001]
    # sigma_a = torch.tensor([[w1/2, 0], [0, h1/2]])
    # print("my_a")
    # print(my_a)

    # my_b = torch.tensor([m2_x, m2_y])
    my_b = torch.cat((m2_x, m2_y), dim=-1)      
    # sigma_b = torch.tensor([[w2/2, 0], [0, h2/2]])
    # print("my_b")
    # print(my_b)

    my_a_subtract_my_b = my_a - my_b  # [ 0.0002     -0.00359997]
    # print()
    # print("my_a_subtract_my_b = my_a - my_b")
    # print(my_a_subtract_my_b)
    
    d1 = torch.norm(my_a_subtract_my_b) ** 2

    d2 = ((h1 + h2) / 2) + ((w1 + w2) / 2) - 2 * (torch.sqrt(h1 * h2) + torch.sqrt(w1 * w2))

    w_2 = torch.abs(d1 + d2)

    return w_2

def NWD(m1_x, m1_y, w1, h1, m2_x, m2_y, w2, h2, C=16):
    w_2 = Gaussian_Wasserstein_Distance(m1_x, m1_y, w1, h1, m2_x, m2_y, w2, h2)
    # print("w_2: ", w_2)
    NWD = torch.exp(-(torch.sqrt(w_2) / C))
    # print("NWD: ", NWD)
    NWD = torch.torch.nan_to_num(NWD, nan=1e5)

    # print("NWD New: ", NWD)
    return NWD

def gaussian_loss(src_boxes, tgt_box):
    assert(len(src_boxes) == len(tgt_box))
    loss_accum = torch.tensor(0.0)  # Initialize as float
    
    if len(tgt_box) == 0:
        loss_accum_lst = torch.full((len(src_boxes),), 2.0)
        loss_accum = len(src_boxes) * torch.tensor(2.0)  # Initialize as float

        return loss_accum, loss_accum_lst
    
    loss_accum_lst = []
    src_bbox_x, src_bbox_y, src_bbox_width, src_bbox_height = src_boxes.t()
    tgt_bbox_x, tgt_bbox_y, tgt_bbox_width, tgt_bbox_height = tgt_box.t()

    # Calculate NWD for all pairs of source and target boxes
    src_bbox_x = src_bbox_x.unsqueeze(1)
    src_bbox_y = src_bbox_y.unsqueeze(1)
    src_bbox_width = src_bbox_width.unsqueeze(1)
    src_bbox_height = src_bbox_height.unsqueeze(1)

    tgt_bbox_x = tgt_bbox_x.unsqueeze(1)
    tgt_bbox_y = tgt_bbox_y.unsqueeze(1)
    tgt_bbox_width = tgt_bbox_width.unsqueeze(1)
    tgt_bbox_height = tgt_bbox_height.unsqueeze(1)

    loss = NWD(src_bbox_x, src_bbox_y, src_bbox_width, src_bbox_height, tgt_bbox_x, tgt_bbox_y, tgt_bbox_width, tgt_bbox_height)
    loss_accum = loss.sum()

    assert(loss_accum == loss.squeeze().sum())

    return loss_accum.detach(), loss.squeeze().detach()


def gaussian_loss_matcher_v2(src_boxes, tgt_bbox, constant=16):
    # print("Gaussian Loss V2")
    source = src_boxes # predicted boxes
    target = tgt_bbox # list of expected predicted boxes

    # Can be eempty
    # if torch.numel(tgt_bbox) == 0:
    #     t_target = []
          
    #     for i, x in enumerate(source.cpu().numpy()):
    #         # print(x)
    #         print("adding penalty")
    #         t_target.append(2) 

    #     t = torch.tensor(t_target)
    #     res = torch.reshape(t, (len(t_target), 1))
    #     div_f = len(src_boxes)
    #     res.div(div_f)

    #     # print("************************")
    #     # print(res)
    #     # print("************************")
    #     res1 = torch.tensor(res).to('cuda')
    #     return res1

    """
    Below is only used when tgt_bbox is a list of ifs
    """
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
    # # print("[T_TARGET]", t_target)    



    #print("t_target: ", t_target)
    # Split each component [[x, y, w, h], [x1, y1, w1, h1]] -> ([x, x1], [y, y1], [w, w1], [h, h1])
    source_x, source_y, source_w, source_h = torch.split(source, 1, dim=1) # unpack
    target_x, target_y, target_w, target_h = torch.split(target, 1, dim=1) # unpack

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
    # sqrt(h1*h2) + sqrt(w1*w2)
    source_h_mul_target_h_add_source_h_mul_target_h = source_h_mul_source_h + source_h_mul_target_h
    #print(source_h_mul_target_h_add_source_h_mul_target_h)
    # times 2
    source_h_mul_target_h_add_source_h_mul_target_h_mul_two = source_h_mul_target_h_add_source_h_mul_target_h.mul(2)
    # print(source_h_mul_target_h_add_source_h_mul_target_h_mul_two)
    d2 = A_add_B - source_h_mul_target_h_add_source_h_mul_target_h_mul_two
    # reshape d2
    if len(tgt_bbox) > 0: # MAYBE NOT A GOOD FIX
        d2 = torch.reshape(d2, (len(tgt_bbox), -1))
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

    else:
        d1_abs = torch.abs(d1)
        # sqrt
        d1_abs_sqrt = d1_abs.sqrt_()
        # div with constant
        d1_abs_sqrt.div(constant)
        # exp
        d1_abs_sqrtdiv_exp = torch.exp(d1_abs_sqrt * -1)
        
        return d1_abs_sqrtdiv_exp.T










def variance(data, mean, ddof=0):
    """ Calcualte the variance with a dictated mean"""
    n = len(data)
    # mean = sum(data) / n
    return sum((x - mean) ** 2 for x in data) / (n - ddof)

def stdev(data, mean):
    var = variance(data, mean)
    std_dev = math.sqrt(var)
    return std_dev

def compute_gaussians(args, testing=False):
    print("*******************************")
    print("   Computing class Gaussians   ")
    print("*******************************")
    
    if testing:
        print("[TEST MODE]")
        BASE_DIR = '../../../dataset/visDrone2022_det'  
        anno_file = 'scaled_visDrone_label_train_w_indicator.json'
        anno_dir = BASE_DIR + '/annotations/'
        print("Labelled annotation file:", str(anno_file), " in dir: ", anno_dir)

    else:
        BASE_DIR = args.data_path
        anno_file = args.annotation_json_label
        anno_dir = args.data_path + '/annotations/'
        print("Labelled annotation file:", str(anno_file), " in dir: ", anno_dir)

    img_dir = BASE_DIR + '/scaled_label_train/'
    coco = COCO(annotation_file=anno_dir + anno_file) # create pycocotools objects
    cat_ids = get_cat_ids(coco)            # get categories

    # load categories
    cats = coco.loadCats(cat_ids)          # load categories 
    print_cats(cats)                       # print -> maybe pretty print later

    cat_num = 1
    for category in cats:
        print(cat_num, " of ", len(cats))
        cat_id = category['id']
        print("Cat_id: ", cat_id)
        cat_name = category['name']
        print("category_name: ", cat_name)

        xs = []
        ys = []
        widths = []
        heights = []
        # Get images for that ID
        for img_id in coco.getImgIds(catIds=[cat_id]):
            current_img = coco.loadImgs(ids=[img_id])
            # print("current_img: ", current_img)
            # print("image: ", current_img[0]['file_name'])
            img_path = img_dir + current_img[0]['file_name'] 
            # get image width and height without loading the image
            im = Image.open(img_path)
            current_img_width, current_img_height = im.size
            
            # print annotation from certain class and the specific image
            annotations_ids = coco.getAnnIds(catIds=[cat_id], imgIds=[img_id])
            # load annotations
            cat_annotations = coco.loadAnns(ids=annotations_ids)

            # collect data
            for anno in cat_annotations: 
                # print("No. Anno: ", len(cat_annotations))
                x_min = anno['bbox'][0]
                y_min = anno['bbox'][1]
                width_box = anno['bbox'][2]
                height_box = anno['bbox'][3]
                # We need to transfrom x,y to the true center, from upper left corner (COCO format)
                x_center = x_min + (width_box * 0.5) 
                y_center = y_min + (height_box * 0.5)

                # NORMALIZE
                x_center_norma = x_center / current_img_width
                y_center_norma = y_center / current_img_height 
                width_norma = width_box / current_img_width
                height_norma = height_box / current_img_height

                # print("width_norm: ", width_norm)
                # append to arrs
                xs.append(x_center_norma)    # center-cord x
                ys.append(y_center_norma)    # center-cord y
                widths.append(width_norma)   # width
                heights.append(height_norma) # height

        print("*****")

        # calculate means
        xs_mean = statistics.mean(xs)
        ys_mean = statistics.mean(ys)
        width_squared_var = stdev(widths, xs_mean) # statistics.stdev(widths) # ** 2 # squared std. dev
        height_squared_var = stdev(heights, ys_mean) #statistics.stdev(heights) # ** 2 # squared std. dev

        # create category distribution
        cat_distribution = {
            'id': cat_id,
            'name': cat_name,
            'xs_mean': xs_mean,
            'ys_mean': ys_mean,
            'width_mean': width_squared_var,
            'height_mean': height_squared_var,
        }

        CLASS_BOX_GAUSSIANS.append(cat_distribution) # append distribtion 
        # Reset
        xs = []
        ys = []
        widths = []
        heights = []
        cat_num += 1
    
    print(CLASS_BOX_GAUSSIANS)
    print("Done!")
    print()


if __name__ == "__main__":
    args = ''
    compute_gaussians(args, testing=True)

    for record in CLASS_BOX_GAUSSIANS:
        cat_name = record['name']
        mu_width = record['xs_mean']
        std_width = record['width_mean']

        print(cat_name, mu_width, std_width)

        # plot height
        mu = mu_width
        variance = std_width
        sigma = math.sqrt(variance)
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)

        plot_label= cat_name
        plt.plot(x, stats.norm.pdf(x, mu, sigma), label=plot_label)
        plt.suptitle('Gaussian BBox Widths')
        plt.xlabel('z-score')
        plt.ylabel('Probability Density')
        # plt.clf()

        widths = []
        heights = []

        plt.legend(loc="upper left")
        plt.savefig('xs_mean_width_mean.jpg')
































# def Gaussian_Wasserstein_Distance(xa, ya, wa, ha, xb, yb, wb, hb):
#     """
#     2nd Wasserstein Distance
#     """
#     my_a = np.array([xa, ya])
#     sigma_a = np.array([[wa/2, 0], [0, ha/2]])
    
#     my_b = np.array([xb, yb])
#     sigma_b = np.array([[wb/2, 0], [0, hb/2]])
    
#     # My - center    
#     my_a_subtract_my_b = np.absolute(np.subtract(my_a, my_b))
#     norm_my_a_subtract_my_b = LA.norm(my_a_subtract_my_b) # || m1 - m2 ||_2^2
#     norm_squared_my_a_subtract_my_b = norm_my_a_subtract_my_b ** 2
#     # print("My norm", norm_squared_my_a_subtract_my_b)

#     # Sigma - Std. Dev
#     sigma_a_subtract_sigma_b = np.absolute(np.subtract(sigma_a, sigma_b))
#     norm_sigma_a_subtract_sigma_b = (LA.norm(sigma_a_subtract_sigma_b, ord='fro')) **2 # Frobenium Norm
#     # print("sigma norm: ", norm_sigma_a_subtract_sigma_b)

#     # 2nd Wasserstein Distance
#     w_2 = norm_squared_my_a_subtract_my_b + norm_sigma_a_subtract_sigma_b
#     return w_2


# def NWD(xa, ya, wa, ha, xb, yb, wb, hb, C=12):
#     """
#     Normalized 2nd Wasserstein Distance
#     > Then, we found that when changing 𝐶 in a certain range (from 8 to 24), the value of AP waves marginally and is much higher than baseline
#     """
#     w_2 = Gaussian_Wasserstein_Distance(xa, ya, wa, ha, xb, yb, wb, hb)
#     NWD = np.exp(-(np.sqrt(w_2)/C))
#     #print("NWD: ", NWD)
#     return NWD
