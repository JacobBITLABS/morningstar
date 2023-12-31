import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import numpy as np
from scipy.optimize import linear_sum_assignment
from models.segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
						   dice_loss, sigmoid_focal_loss)
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
					   accuracy, get_world_size, interpolate,
					   is_dist_avail_and_initialized, inverse_sigmoid)
import random
import wandb
from PIL import Image, ImageFont, ImageDraw
from util import box_ops
from view2view.self_attention import MultiheadAttentionSIM
import sys

# we matcher ground mod aerial. Både fordi det er mindst mod mange of fordi 
# we tager alle 900 proposals, omend det måske skulle sættes lidt ned overall 
class View2View(torch.nn.Module):
	""" View 2 View Model mapper"""
	def __init__(self, num_queries, patch_size, resize_mode, topk, multihead_att=True, num_heads=10, head_dim=384):
		super(View2View, self).__init__()
		self.num_queries = num_queries
		self.patch_size = patch_size
		self.resize_mode = resize_mode
		self.topk = topk
		self.mean = torch.tensor([0.485, 0.456, 0.406]) # to normalize images
		self.std = torch.tensor([0.229, 0.224, 0.225])
		self.to_pil = T.ToPILImage() # Transpose the tensor from CHW to HWC format
		self.multihead_att = multihead_att

		# self.multiheadattention = nn.MultiheadAttention(384, 8) # https://atcold.github.io/pytorch-Deep-Learning/en/week12/12-3/
		if not multihead_att:
			self.W_ground = nn.Linear(384, 384, bias=False)
			self.W_aerial = nn.Linear(384, 384, bias=False)
		else: 
			embed_dim = 3 * 900 
			self.multi_head_sim = MultiheadAttentionSIM(3 * 900, num_heads, topk)
			
		# positinonal encoding can be calulated onces
		batch_size = 2
		seq_length = topk
		embed_size = 3 * 900 # 384
		positional_encoding = torch.zeros(seq_length, embed_size)
		# Calculate the positional encodings
		position = torch.arange(0, seq_length, dtype=torch.float32).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, embed_size, 2, dtype=torch.float32) * (-np.log(10000.0) / embed_size))
		positional_encoding[:, 0::2] = torch.sin(position * div_term)
		positional_encoding[:, 1::2] = torch.cos(position * div_term)

		# Expand the positional encoding to match the batch size
		self.positional_encoding = positional_encoding.unsqueeze(0).expand(batch_size, -1, -1).to('cuda')


	def extract_patches_from_normalized_boxes(self, samples, output_bboxes, scores, debug=False, save_patches=True, save_samples=False):	
		if isinstance(samples, NestedTensor):
		# because NestedTensor is hated outside of the efficency-scope
			image_tensors = samples.tensors
		else:
			image_tensors = samples
			# image_tensors = image_tensors.unsqueeze(0) # ???

		# print("image_tensors.shape: ", image_tensors.shape)
		batch_size, channels, width, height = image_tensors.shape

		if debug:
			print("Image Tensors Size: ", image_tensors.shape)
			print("output_bboxes Boxes Size: ", output_bboxes.size)
			print("output_bboxes.size()", output_bboxes.size())
		
		# torch.Size([2, topK, 4])
		boxes = output_bboxes.clone()  # Make a copy to keep the original tensor intact
		scaled_boxes = torch.empty_like(boxes) # Create a tensor to store the scaled boxes with the same shape as normalized_boxes
		batch_size, num_boxes, _ = boxes.size() # torch.Size([2, topK, 4])

		for batch_idx in range(batch_size):
			channels, im_w, im_h = image_tensors[batch_idx].shape
			# change the size
			target_sizes = torch.tensor([[im_w, im_h]])
			target_sizes = target_sizes.cuda()
			img_h, img_w = target_sizes.unbind(1)
			scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
			scaled_boxes = boxes * scale_fct[:, None, :]
			# another solution to above
			# boxes[batch_idx, :, 0] *= im_w  # Normalize y_min
			# boxes[batch_idx, :, 1] *= im_h  # Normalize x_minj
			# boxes[batch_idx, :, 2] *= im_w  # Normalize y_max
			# boxes[batch_idx, :, 3] *= im_h  # Normalize x_max

			if save_samples:
				img = image_tensors[batch_idx]
				normalized_tensor = img.clone()  # Create a copy of the tensor to avoid modifying the original
				for t, m, s in zip(normalized_tensor, self.mean, self.std):
						t.mul_(s).add_(m)
				pil_image = self.to_pil(normalized_tensor.cpu()) # Make sure to move the tensor to CPU if it's on the GPU	
				pil_image.save(f"bbox_images/image_w_o_bboxes_{batch_idx}.png", "png")
				boxes_to_plot = scaled_boxes[batch_idx]
				draw = ImageDraw.Draw(pil_image)
				# Draw boxes			
				for x_min, y_min, x_max, y_max in boxes_to_plot.tolist():
					#xmin, ymin, xmax, ymax = box_ops.box_cxcywh_to_xyxy(out_bbox)
					# print("Plotting box: ", x_min, y_min, x_max, y_max)
					ANNO_COLOR = (221, 40, 252) 
					draw.rectangle(((x_min, y_min), (x_max, y_max)), outline=ANNO_COLOR)
					
				pil_image.save(f"bbox_images/image_with_boxes_{batch_idx}.png", "png")
			
		# Define the patch size
		# patch_size = (3, 1, 128) # patch_size with channels TODO: make this general
		patch_size = (3, 1, 30*30) 
		resized_roi_patches = torch.empty(batch_size, num_boxes, *patch_size).to('cuda') # Initialize resized_roi_patches with the predefined shape

		# Extract patches for each bounding box
		for i in range(batch_size):
			# get current image
			curr_image = image_tensors[i,:,:,:]
			# print("current image: ", curr_image.size())
			for j in range(num_boxes):
				# Convert tensors to numpy arrays before accessing values
				x_min = scaled_boxes[i, j, 0].cpu().detach().cpu().round().int() 
				y_min = scaled_boxes[i, j, 1].cpu().detach().cpu().round().int()  #.cpu().detach().numpy()
				x_max = scaled_boxes[i, j, 2].cpu().detach().cpu().round().int()  # .cpu().detach().numpy()
				y_max = scaled_boxes[i, j, 3].cpu().detach().cpu().round().int()  # .cpu().detach().numpy()

				# torch.Size([2, 3, 776, 640])
				patch = curr_image[:, y_min:y_max, x_min:x_max]
				patch_size = patch.shape
				
				# a prediction can have a 0 dimension, we need to handle this. We "penalize" it by giving it no information with 0 / black
				zero_dim = [dim for dim, size in enumerate(patch.size()) if size == 0]
				len_zero_dim = len(zero_dim)

				if len_zero_dim > 0:
					# there is an empty vector
					if len_zero_dim > 1:
						patch = torch.zeros(3, 2, 2)
					else:
						if zero_dim[0] == 2:
							patch = torch.zeros(3, patch_size[1], patch_size[1])
						else:
							patch = torch.zeros(3, patch_size[2], patch_size[2])
					
				# debug to the the patches:
				if save_patches: 
					normalized_tensor = patch.clone()  # Create a copy of the tensor to avoid modifying the original
					for t, m, s in zip(normalized_tensor, self.mean, self.std):
							t.mul_(s).add_(m)
					to_pil = T.ToPILImage() # Transpose the tensor from CHW to HWC format
					pil_image = to_pil(normalized_tensor.cpu()) # Make sure to move the tensor to CPU if it's on the GPU
					pil_image.save(f"patch_saver/{j}_patch_sample.png")

				patch = patch.unsqueeze(0) # add a mini batch to confrom to F.interpolate()
				resized_roi = F.interpolate(patch, size=self.patch_size, mode=self.resize_mode) # align_corners=False
				resized_roi = resized_roi.squeeze(0) # remove mini batch
				# flatten to [3, patch_size[0]* patch_size[1]]
				resized_roi = resized_roi.view(3, -1).unsqueeze(1)

				# if debug:
				# 	print("[New Patch Size (roi)]: ", resized_roi.size())

				resized_roi_patches[i, j] = resized_roi # i: batch, j: box

		if debug:
			print("[Resized_roi_patches Size]: ", resized_roi_patches.size())
		
		return resized_roi_patches # torch.Size([NUM BOXES, 3, 1, 128]) # channels, 1, 128 (128 = 64 x 64)

	def get_topk(self, out_logits, out_bbox, threshold=0.0, debug=True):
		# print("TOP K")
		# print("out_logits.shape: ", out_logits.shape)
		# print("out_logits.shape: ", out_bbox.shape)

		prob = out_logits.sigmoid()
		topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), self.topk, dim=1)
		scores = topk_values
		topk_select_index = topk_boxes = topk_indexes // out_logits.shape[2]
		labels = topk_indexes % out_logits.shape[2]
		boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
		boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

		# construct an output shaped like out_logicts, but only containing topK
		# topk_select_index = topk_indexes // out_logits.shape[2]
		# gather corresponding logit and boxes
		selected_logits = torch.gather(out_logits, 1, topk_select_index.unsqueeze(-1).repeat(1,1, 20)) # 20 = num_classes
		selected_boxes =  torch.gather(out_bbox, 1, topk_select_index.unsqueeze(-1).repeat(1,1,4))
		
		# construct src-like dictionary
		top_k_outputs = {
			'pred_logits': selected_logits,
			'pred_boxes': selected_boxes, 
		}	

		# # score thresholds 
		# keep = scores > threshold
		# print("KEEP; ", keep)
		# boxes = boxes[keep]
		# labels = labels[keep]
		# scores = scores[keep] # probabilities
		# Apply thresholding to each batch
		# for batch_idx in range(boxes.shape[0]):
		# 	batch_keep = scores[batch_idx] > threshold
		# 	boxes[batch_idx] = boxes[batch_idx][batch_keep]
		# 	labels[batch_idx] = labels[batch_idx][batch_keep]
		# 	scores[batch_idx] = scores[batch_idx][batch_keep]  # probabilities

		# im_h, im_w = [600, 337] #im.size
		# target_sizes = torch.tensor([[im_w,im_h]])
		# target_sizes = target_sizes.cuda()
		# img_h, img_w = target_sizes.unbind(1)
		# scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
		# boxes = boxes * scale_fct[:, None, :]

		# print("TOPK")
		# print(boxes)

		return scores, boxes, top_k_outputs
	
	
	def get_attention(self, ground, aerial, debug=False):
		""" Compute attention Score between ground and aerial
			Output: 
			In summary, the attention weight at (i, j) tells you how much focus or
			influence the element at position j in the "Aerial" input has on the element 
			at position i in the "Ground"  
		"""
		if debug:
			print("ATTENTION FUNCTION")
			print("ground.size(): ", ground.size())
			print("aerial.size(): ", aerial.size())

		ground_transformed = self.W_ground(ground)
		aerial_transformed = self.W_aerial(aerial)

		if debug:
			print("ground_transformed.size(): ", ground_transformed.size())
			print("aerial_transformed.size(): ", aerial_transformed.size())
			# print("ground_transformed: ", ground_transformed)
			# print("aerial_transformed : ", aerial_transformed)

		# Calculate the attention matrix using dot product, scaled by sqrt(d_k)
		d_k = 384  # Dimension of the keys
		similarity_matrices = torch.bmm(ground_transformed, aerial_transformed.transpose(1, 2)) / (d_k ** 0.5)
		similarity_matrices = F.softmax(similarity_matrices, dim=1)

		if debug:
			print("Attention Weights:")
			print(similarity_matrices)
			print("[Attention Weights Shape]: ", similarity_matrices.size())

		# # The resulting similarity_matrices list contains one similarity matrix for each batch, each of shape (100, 100)
		return similarity_matrices # batched		

	def get_y_area_cost(self, bboxes_1, bboxes_2):
		# Extract the y-values and areas from the bounding boxes
		y_values_1 = bboxes_1[:, :, 1]
		# print("y_values_1: ", y_values_1 )
		areas_2 = bboxes_2[:, :, 2] * bboxes_2[:, :, 3]
		
		# print("areas_2", areas_2)
		# Expand dimensions to prepare for broadcasting
		y_values_1 = y_values_1.unsqueeze(2)
		areas_2 = areas_2.unsqueeze(1)

		# Calculate the L1 distance between y-values in bboxes_1 and areas in bboxes_2
		cost_matrix = torch.abs(y_values_1 - areas_2)

		return cost_matrix


	def construct_cost_matrix_with_ratio(self, bboxes_1, bboxes_2):
		# Calculate the width-to-height ratio for each box in both sets
		ratio_1 = bboxes_1[:, :, 2] / bboxes_1[:, :, 3]
		ratio_2 = bboxes_2[:, :, 2] / bboxes_2[:, :, 3]

		# Expand dimensions to prepare for broadcasting
		ratio_1 = ratio_1.unsqueeze(2)
		ratio_2 = ratio_2.unsqueeze(1)

		# Calculate the absolute differences in ratio
		ratio_diff = torch.abs(ratio_1 - ratio_2)

		return ratio_diff

	def forward(self, g_samples, a_samples, g_outputs, a_outputs, debug=False):
		# 1. get each of the output boxes
			# we do only want the assigned boxes from the matcher actually. Therefore we want to take that out of the 
			# no because we do not want to be afhængig af matcheren som en del af processen direkte
		g_boxes = g_outputs['pred_boxes']
		a_boxes = a_outputs['pred_boxes']
		# 1.5 Get The topk outputs (Efficentcy)
		g_out_logits = g_outputs['pred_logits']
		a_out_logits = a_outputs['pred_logits']

		g_scores, g_boxes, g_top_k_outputs  = self.get_topk(g_out_logits, g_boxes)
		a_scores, a_boxes, a_top_k_outputs = self.get_topk(a_out_logits, a_boxes)

		# 2. get the patches for each output
		g_patches = self.extract_patches_from_normalized_boxes(g_samples, g_boxes, g_scores).squeeze()
		a_patches = self.extract_patches_from_normalized_boxes(a_samples, a_boxes, a_scores).squeeze()
		
		# print("[g_stacked_patches size]: ", g_patches.size()) # num_queries x embed/hiddendim
		# print("[a_stacked_patches size]: ", a_patches.size())

		if len(g_patches.size()) < 4:
			g_patches = g_patches.unsqueeze(0)
		if len(a_patches.size()) < 4:
			a_patches = a_patches.unsqueeze(0)

		# Define the mean and std for normalization
		# mean = torch.tensor([0.485, 0.456, 0.406])
		# std = torch.tensor([0.229, 0.224, 0.225])

		# Permute the tensor
		# permuted_tensor = g_patches.clone()
		# permuted_tensor = permuted_tensor.permute(0, 2, 1, 3)
		# print("permuted_tensor.size(): ", permuted_tensor.size())

		# # Loop through each batch and image, normalize, and save
		# for batch_idx in range(permuted_tensor.size(0)):
		# 	image = permuted_tensor[batch_idx, :, :, :]
		# 	print("captured image: ", image.size())
		# 	# Normalize the image
		# 	for t, m, s in zip(image, mean, std):
		# 		t.mul_(s).add_(m)

		# 	# Transpose the tensor from CHW to HWC format
		# 	image = image.permute(1, 2, 0)

		# 	# Convert the tensor to a PIL image
		# 	pil_image = Image.fromarray((image * 255).byte().cpu().numpy())

		# 	# Save the PIL image with a unique filename
		# 	filename = f"patches_images/output_image_batch_{batch_idx}_sample.png"
		# 	pil_image.save(filename)


		# 3. Matrix Tranformations using the learned parameters		
		# 4. Cross Attention
		# attention_scores = self.multiheadattention(g_stacked_patches, a_stacked_patches, a_stacked_patches) # IDK which should atten which
		batch_size_g = g_patches.size(0)
		batch_size_a = a_patches.size(0)

		# print("batch_size_a: ", batch_size_a)
		# print("batch_size_g: ", batch_size_g)

		assert batch_size_a == batch_size_a

		# Reshape tensors to the correct shape for multihead attention
		"""
		To reshape your tensor with size [2, 100, 3, 128] to be accepted by nn.MultiheadAttention, you need to transpose and reshape it to have the shape (sequence_length, batch_size, embed_dim), where sequence_length is 100, batch_size is 2, and embed_dim is 3 * 128 = 384 (due to the 3 channels of size 128). Here's how you can do it:
		"""
		# Reshape the tensor for nn.MultiheadAttention
		sequence_length = self.topk  # 100
		batch_size = batch_size_a # == batch_size_g
		embed_dim = 3 * 900 # 3 * 128  # Combine the 3 channels

		# Transpose and reshape
		tensor_g = g_patches.permute(1, 0, 2, 3).reshape(batch_size, sequence_length, embed_dim)
		tensor_a = a_patches.permute(1, 0, 2, 3).reshape(batch_size, sequence_length, embed_dim)
			
		# if debug:
		# 	print("[tensor_g size]: ", tensor_g.size()) # num_queries x embed/hiddendim
		# 	print("[tensor_a size]: ", tensor_a.size())

		# Add the positional encoding to the input tensor
		tensor_g = tensor_g + self.positional_encoding
		tensor_a = tensor_a + self.positional_encoding

		# # Sum along the third dimension to get the pairwise similarity tensor
		# pairwise_similarity = att_output.sum(dim=2).T # .T: I see, you want to reshape the tensor from [100, 2] to [2, 100]
		# print("[pairwise_similarity]: ", pairwise_similarity)
		if self.multihead_att:
			pairwise_similarity = self.multi_head_sim(tensor_g, tensor_a)
		else:
			pairwise_similarity = self.get_attention(tensor_g, tensor_a) # Shape: torch.Size([2, 100, 100])

		# print("[PAIRWISE SIMILARITY ]")
		# print("pairwise_similarity: ", pairwise_similarity)
		# print("pairwise_similarity size: ", pairwise_similarity.size())
		
		# Assuming pairwise_similarity is a torch tensor of shape (2, 100, 100)
		batch_size, num_rows, num_cols = pairwise_similarity.size()

		# Step 1: Compute the cost matrix
		# Find the maximum similarity score in each batch
		max_similarity = pairwise_similarity.max(dim=2, keepdim=True)[0]
		# Compute the cost matrix for the entire batch in one go
		similarity_cost_matrix = max_similarity - pairwise_similarity
			
		# print("[MAX SIMILARITY ]")
		# print("max_similarity: ", max_similarity)
		y_area_cost_matrix = self.get_y_area_cost(g_boxes, a_boxes)

		ratio_cost_matrix = self.construct_cost_matrix_with_ratio(g_boxes, a_boxes)

		# sum to cost-matrix ( equal weight for now)
		cost_matrix = similarity_cost_matrix + y_area_cost_matrix + ratio_cost_matrix


		with torch.no_grad():
			# Step 2: Use linear_sum_assignment to find near-optimal assignment
			# Initialize an empty tensor to store the assignments
			assignments = [] #torch.zeros(batch_size, num_rows, dtype=torch.int64, device=cost_matrix.device)

			# Loop through each batch
			for batch_idx in range(batch_size):
				# Move the cost matrix to CPU and convert the cost matrix for this batch to a NumPy array for linear_sum_assignment
				cost_matrix_np = cost_matrix[batch_idx].cpu().numpy() 
				# Use linear_sum_assignment to find the optimal assignment
				# indices = [linear_sum_assignment(cost_matrix_np.detach().numpy())] # CPU Bound
		
				row_indices, col_indices = linear_sum_assignment(cost_matrix_np)
				# Convert the indices to tensors and append them to the assignments list
				row_indices_tensor = torch.as_tensor(row_indices, dtype=torch.int64)
				col_indices_tensor = torch.as_tensor(col_indices, dtype=torch.int64)

				assignments.append((row_indices_tensor, col_indices_tensor))

				# assignments.append([(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices])

			# print("MAPPER ASSIGNMENTS:")
			# print(assignments)
				
		return assignments, g_top_k_outputs, a_top_k_outputs


class V2VCriterion(nn.Module): 

	def __init__(self, losses, num_classes, weight_dict, args):
		super().__init__()
		self.weight_dict = weight_dict  
		self.losses = losses
		self.focal_alpha = 0.25
		self.num_classes = num_classes
	
	# Function to convert pred_logits to target format
	def convert_pred_logits_to_target(self, top_k_outputs):
		"""
		This function takes the output from output['out_logits'] and construct a target format
   		"""
		# print("convert_pred_logits_to_target.top_k_outputs: ", top_k_outputs)
		pred_logits, out_bbox = top_k_outputs['pred_logits'], top_k_outputs['pred_boxes']
		# Initialize the list to store target dictionaries
		batch_tgt_lst = []

		# Get top 
		# prob = pred_logits.sigmoid()
		# topk_values, topk_indexes = torch.topk(prob.view(pred_logits.shape[0], -1), topK, dim=1)
		# all_indices = torch.arange(pred_logits.shape[2]).unsqueeze(0).repeat(pred_logits.shape[0], 1).to('cuda')
		# print("all_indices: ", all_indices)
		# topk_boxes = all_indices // pred_logits.shape[2]
		# labels = all_indices % pred_logits.shape[2]
		# boxes = torch.gather(out_bbox, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

		labels = pred_logits.argmax(-1) # Get the class index with the highest score. It is a prob. distribution
		scores = pred_logits.softmax(-1).max(-1).values  # Get the maximum softmax score
		boxes = out_bbox # Get the predicted bounding boxes
		# print("SELECTED BOXES: ", boxes)
		# print("SELECTED LABELS: ", labels)
		# sys.exit()

		# assert same batch size 
		assert len(labels) == len(boxes) # 2 == 2

		batch_size, num_queries, num_classes = pred_logits.shape
		# Iterate over each batch
		for batch_idx in range(batch_size):
			batch_prediction_dict = {} # dict for each prediction
			labels_to_add = labels[batch_idx]
			boxes_to_add = boxes[batch_idx]
			
			# Assign values to dict
			batch_prediction_dict['labels'] = labels_to_add
			batch_prediction_dict['boxes'] = boxes_to_add

			# Append the prediction dictionary to the target list
			batch_tgt_lst.append(batch_prediction_dict)

		return batch_tgt_lst


	def loss_labels(self, outputs, targets, assigments, num_boxes, log=True):
		"""Classification loss (NLL)
		targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
		"""
		# print("indices shape: ", indices.size())   # indices shape:  torch.Size([2, 100])
		# print("g_labels shape: ", g_labels.size()) # g_labels shape:  torch.Size([2, 100])
		# print("a_labels shape: ", a_labels.size()) # a_labels shape:  torch.Size([2, 100])
		assert 'pred_logits' in outputs
		# src_logits = outputs['pred_logits']
		# print("******* src_logits ********")
		# print(src_logits)

		# Get topK src_logits.
		# outputs -> g_outputs
		src_logits = outputs['pred_logits'] #self.top_k_logits(outputs, g_topk_indexes)['pred_logits']
		# print("Ground src_logits: ", src_logits)
		# print("TARGETS: ", targets)

		# print("Indices: ", indices)
		indices = assigments
		# print("Assingments: ", indices)
		idx = self._get_src_permutation_idx(indices)
		# print("IDX: ", idx)

		target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
		# print("target_classes_o: ", target_classes_o)
		target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
		target_classes[idx] = target_classes_o
		# print("target_classes[idx]: ", target_classes) 

		# # batch_size, num_queries, num_classes = src_logits.shape
		# target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
		# 									dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
		# print("CREATED TARGET_CLASSES_ONEHOT")
		# target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

		# target_classes_onehot = target_classes_onehot[:,:,:-1]
		# Create one-hot encoding of target classes
		target_classes_onehot = F.one_hot(target_classes, num_classes=self.num_classes)
		torch.set_printoptions(threshold=10_000)
		# print("target_classes_onehot: ", target_classes_onehot)

		#print("target_classes_onehot ", target_classes_onehot, " with size: ", target_classes_onehot.shape)
		#print("src_logits shape: ", src_logits.shape)
		# src_logits:   torch.Size([2, 900, 20])
		# target one hot:  torch.Size([2, 900, 20])
		loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot.float(), num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
		losses = {'loss_map_ce': loss_ce}

		# sys.exit()
		
		if log:
			# TODO this should probably be a separate loss, not hacked in this one here
			losses['map_class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
		return losses

	def _get_src_permutation_idx(self, indices):
		"""
		Modified from DETR
		"""
		# permute predictions following indices
		batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
		# src_idx = torch.cat([src for (src, _) in indices])
		src_idx = torch.cat([src for (_, src) in indices])
		return batch_idx, src_idx
	
	def get_loss(self, loss, outputs, targets, assigments, num_boxes, **kwargs):
		loss_map = {
			'labels_map': self.loss_labels,
			# 'position'
		}
		assert loss in loss_map, f'do you really want to compute {loss} loss?'
		return loss_map[loss](outputs, targets, assigments, num_boxes, **kwargs)

	def forward(self, g_outputs, a_outputs, assignments, g_top_k_outputs, a_top_k_outputs):
		""" Forward """
		# convert a_outputs to target format
		a_targets = self.convert_pred_logits_to_target(a_top_k_outputs) # convert a_outputs to target format

		# Compute the average number of target boxes accross all nodes, for normalization purposes
		# we get the number of boxes from the indices, as we want to normalize by the assigned onces
		# Here it it done with targets which is very fine
		num_boxes = sum(len(t["labels"]) for t in a_targets)
		num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(a_outputs.values())).device)
		if is_dist_avail_and_initialized():
			torch.distributed.all_reduce(num_boxes)
		num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

		# compute the losses
		losses = {}
		for loss in self.losses:
			kwargs = {}
			losses.update(self.get_loss(loss, g_top_k_outputs, a_targets, assignments, num_boxes, **kwargs))

		return losses

def build(args):
	"""
	Function responsible for constructino View2View model
	"""
	num_classes = 20 if args.dataset_file != 'coco' else 91

	model = View2View(
		num_queries=args.num_queries,
		patch_size=(30, 30), # we use the algorithm to rescale to this and then faltten. # (1, 128),
		resize_mode='area', #'bilinear', # is this the best way?
		topk=30
		)

	weight_dict = {'loss_map_ce': args.cls_v2v_loss_coef}
	losses = ['labels_map'] # , 'boxes', 'cardinality']

	criterion = V2VCriterion(losses, num_classes, weight_dict, args)

	return model, criterion





	# def top_k_logits(self, outputs, topk_indexes):
	# 	"""
	# 	Extract the indices in topk_indexes and returns that in a src dict format
   	# 	"""
	# 	torch.set_printoptions(profile="full")
	# 	out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
		
	# 	topk_select_index = topk_indexes // out_logits.shape[2]
	# 	# gather corresponding logit and boxes
	# 	selected_logits = torch.gather(out_logits, 1, topk_select_index.unsqueeze(-1).repeat(1,1, 20)) # 20 = num_classes

	# 	selected_boxes =  torch.gather(out_bbox, 1, topk_select_index.unsqueeze(-1).repeat(1,1,4))
		
	# 	# construct src-like dictionary
	# 	top_k_outputs = {
	# 		'pred_logits': selected_logits,
	# 		'pred_boxes': selected_boxes, 
	# 	}	
	# 	return top_k_outputs