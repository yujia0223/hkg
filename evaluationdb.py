# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 12:03:34 2021

@author: admin
"""
import torch
import numpy as np 
import pandas as pd

from sklearn import metrics

from collections import defaultdict

from numpy import genfromtxt


def cluster_scores(cluster_labels, labels_true):

        labels_pred = cluster_labels

        ARI = metrics.adjusted_rand_score(labels_true, labels_pred)
        print ("ARI:", round(ARI, 3))

        homogeneity_score = metrics.homogeneity_score(labels_true, labels_pred)
        print("homogeneity_score:", round(homogeneity_score, 3))

        completeness_score = metrics.completeness_score(labels_true, labels_pred)
        print("completeness_score:", round(completeness_score, 3))

        v_measure_score = metrics.v_measure_score(labels_true, labels_pred)
        print("v_measure_score:", round(v_measure_score, 3))

        NMI = metrics.normalized_mutual_info_score(labels_true, labels_pred)
        print("NMI:", round(NMI, 3))
        MI = metrics.mutual_info_score(labels_true, labels_pred)
        print("MI:", round(MI, 3))
        print("")
        return ARI, homogeneity_score, completeness_score,v_measure_score, NMI


def cluster_entities(cluster_index, level, new_reference_data):

        print("Clustering entities..")

        new_reference_data['category'] = pd.factorize(new_reference_data['level{0}'.format(level)])[0]
        #print ("Factorized class labels", diff_vectors_df.category)

        cluster_labels = cluster_index[:,level]

        #find the unique classes again here for cluster number k
        classes_df = new_reference_data.drop_duplicates(subset='level{0}'.format(level), keep="first")

        #print ("Unique classes", classes_df.shape[0])
        #print (classes_df.classes)
        class_count = classes_df.shape[0]    #find the unique combos of s-o pairs

        labels_true = new_reference_data.category.tolist() #the actual class labels frm the df, first itn is same, rest merged

        #TODO:  change the cluster_labels
        print("NCRP evaluation results")

        ARI, homogeneity_score, completeness_score,v_measure_score, NMI = cluster_scores(cluster_labels, labels_true)
        
        return ARI, homogeneity_score, completeness_score,v_measure_score, NMI

