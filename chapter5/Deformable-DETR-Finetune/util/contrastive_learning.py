"""
At first we are simply trying a technique involving the silhouette score. We want to enforce the predictions of the classes 

"""
import torch
import umap
import numpy as np
from sklearn.metrics import silhouette_samples
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

class Contrastive():
    def __init__(self):
        self.umap = umap.UMAP(n_components=2)

    def compute_umap(self, embeddings):
        # Fit and transform the data to 2D
        embeddings_2d = self.umap.fit_transform(embeddings)
        return embeddings_2d

    def get_silhouette(self, selected_embeddings, selected_labels, plot=False):
        # print("[get_silhouette]")
        # print("[selected_embeddings].size() ", selected_embeddings.size())
        # Reshape embeddings
        selected_embeddings = selected_embeddings.cpu().detach().numpy()
        #print("[selected_embeddings].size() ", selected_embeddings.shape)
        selected_labels = selected_labels.cpu().numpy()
        #print("selected_labels: ", selected_labels)
        unique_labels = np.unique(selected_labels)

        all_silhouette_scores = []
        for i in range(selected_embeddings.shape[0]):
            embedding = selected_embeddings[i]
            #print("[embedding].size() ", embedding.shape)
            embeddings_2d = self.compute_umap(embedding)  # get umap
            
            #print("embeddings_2d")
            #print(embeddings_2d)

            if plot:
                plt.figure(figsize=(8, 6))
                for i, label in enumerate(unique_labels):
                    mask = selected_labels == label
                    mask = mask.cpu()
                    classs = label.item()
                    plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], label=f'Class {classs}', color=label_color_dict[label])

                plt.title('UMAP Plot with Class Labels')
                plt.xlabel('UMAP Component 1')
                plt.ylabel('UMAP Component 2')
                plt.legend(loc='upper right')
                plt.savefig('current' + '_scatter.png')
                print("done.")

            # compute pairwise distances
            pairwise_distances_all = pairwise_distances(embeddings_2d)
            # Compute silhouette scores for all points
            silhouette_scores_all = silhouette_samples(pairwise_distances_all, labels=selected_labels, metric='precomputed')
            # Calculate the overall silhouette score
            overall_silhouette_score = np.mean(silhouette_scores_all)

            if overall_silhouette_score < 0.0:
                # negative
                all_silhouette_scores.append(0)
            else: 
                all_silhouette_scores.append(overall_silhouette_score)
        
        # print("all_silhouette_scores: ", all_silhouette_scores)
        return np.sum(all_silhouette_scores)



  # Group embeddings by their corresponding labels
        # grouped_embeddings = {label: [] for label in unique_labels}
        # for i, label in enumerate(labels_cpu):
        #     grouped_embeddings[label].append(embeddings_2d[i])

        # # Compute silhouette scores for each label
        # silhouette_scores = {}
        # for label, emb_list in grouped_embeddings.items():
        #     result = compute_silhouette(label, emb_list)
        #     if result[1] is not None:
        #         silhouette_scores[result[0]] = result[1]

        # # Print silhouette scores for each label
        # for label, score in silhouette_scores.items():
        #     print(f"Label {label}: Silhouette Score: {score}")
