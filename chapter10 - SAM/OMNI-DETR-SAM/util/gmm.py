import numpy as np
from sklearn.mixture import GaussianMixture
import torch
import torch.nn as nn
from collections import deque
import torch.nn.functional as F

class GMM():
    """
    Thresholding with gaussian mixture model
    """
    def __init__(self, num_score_samples=100, num_classes=10):
        self.scores = [deque( maxlen=num_score_samples) for _ in range(num_classes+1)] #[] # list of lists with scores
        self.num_score_samples = num_score_samples
        self.num_classes = num_classes
        self.thrs = [0.7] * (num_classes+1) # [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]

    def policy(self, gmm_assignment, gmm_scores, scores, default_gt_threshold=0.7, policy='high'):
        assert policy in ['middle', 'high']
        if policy == 'high':
            if (gmm_assignment == 1).any():
                gmm_scores[gmm_assignment == 0] = -np.inf
                indx = np.argmax(gmm_scores, axis=0) # Returns the indices of the maximum values along an axis.
                pos_indx = (gmm_assignment == 1) & (scores >= scores[indx]).squeeze()
                pos_thr = float(scores[pos_indx].min())
                # pos_thr = max(given_gt_thr, pos_thr)
            else:
                pos_thr = default_gt_threshold
        elif policy == 'middle':
            if (gmm_assignment == 1).any():
                pos_thr = float(scores[gmm_assignment == 1].min())
                # pos_thr = max(given_gt_thr, pos_thr)
            else:
                pos_thr = default_gt_threshold
        return pos_thr

    def compute_gaussian_mixture_model(self, label:int, scores):
        # print("SCORES: ", scores)
        if isinstance(scores, deque):
            scores = np.array(list(scores))
            # print("SCORES LISTED: ", scores)
        # print("scores: ", scores)
        if  len(scores) > 4:
            if len(scores.shape) == 1:
                scores = scores[:, np.newaxis]
            means_init = [[np.min(scores)], [np.max(scores)]]
            weights_init = [1 / 2, 1 / 2]
            precisions_init = [[[1.0]], [[1.0]]]
            #print("compute gaussian: ", label)
            # Fit a Gaussian Mixture Model with 2 components
            gmm = GaussianMixture(n_components=2,  
                                weights_init=weights_init,
                                means_init=means_init,
                                precisions_init=precisions_init) # positive and negative 
            gmm.fit(scores)
            # Get the assignments for each sample
            gmm_assignment = gmm.predict(scores) # Predict the labels for the data samples in X using trained model.
            # Obtain the scores based on the fitted GMM -> Compute the log-likelihood of each sample.
            score_samples = gmm.score_samples(scores)
            # compute threshold for label
            threshold = self.policy(gmm_assignment, score_samples, scores)
        else: 
            # if we already have a thrs that is not the defualt one, we then keep that.
            # print("THRES: ", self.thrs)
            # print("label: ", label)
            if self.thrs[label] != 0.7:
                threshold = self.thrs[label]
            else:
                # otherwise we keep and simply return it here for continuity.
                threshold = default_gt_threshold = 0.7
        return threshold

    def extract_from_raw(self, outputs):
        """
        Extract the values from the raw outputs of DETR. Returns the labels and scores
        """
        # pred_logits = outputs['pred_logits'] # [batch_size, num_queries, num_classes]
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        # topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]

        results = zip(torch.flatten(scores).tolist(), torch.flatten(labels).tolist())
        return results

    def gmm_update(self, outputs):
        results = self.extract_from_raw(outputs)

        for score, label in results:
            # must be in range of valid classes. We do not track the other onces 
            if label <= self.num_classes: 
                #print("label: ", label, " score: ", score)
                self.scores[label].append(score)

        # recompute the gaussian models and get the thresholds
        for label in range(1, self.num_classes+1):
            new_thrs = self.compute_gaussian_mixture_model(label, self.scores[label])
            # print("label: ", label, " threshold: ", new_thrs)
            self.thrs[label] = new_thrs

        # print("NEW THRS: ", self.thrs)
        return True

    def gmm_recompute(self):
        # recompute the gaussian models and get the thresholds
        for label in range(1, self.num_classes+1):
        #for label in range(0, self.num_classes):
            new_thrs = self.compute_gaussian_mixture_model(label, self.scores[label])
            # print("label: ", label, " threshold: ", new_thrs)
            self.thrs[label] = new_thrs

    def score_update(self, outputs):
        """ Iterate over each class id and add the scores to corresponding deque"""
        results = self.extract_from_raw(outputs)

        for score, label in results:
            # must be in range of valid classes. We do not track the other onces 
            if label <= self.num_classes: 
                #print("label: ", label, " score: ", score)
                self.scores[label].append(score)

        # # reompute 
        # self.gmm_recompute()

    def get_current_gmm_thresholds(self):
        """ Returnt the current thrs """
        return self.thrs

    def overwrite_thresholds(self, thrs):
        self.thrs = thrs

    def overwrite_scores(self, scores):
        """
        Takes topk=100 elements and  puts them into their respective dequre
        """
        for i, sublist in enumerate(scores):
            self.scores[i] = deque(sorted(sublist, reverse=True)[:100])

        # print("Printing Current deques..:")
        # for queue in self.scores:
        #     print(list(queue))

        self.gmm_recompute()
    