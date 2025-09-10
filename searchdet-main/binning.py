
import os
import time
import numpy as np
import torch
from PIL import Image
import cv2
import faiss
import matplotlib .pyplot as plt
from sklearn .metrics .pairwise import cosine_similarity
from sklearn .preprocessing import MinMaxScaler

from heatmap_generation import build_dino ,DinoFeatureExtractor
from segment_anything_hq import sam_model_registry ,SamAutomaticMaskGenerator

def initialize_models ():
    total_init_start =time .time ()
    print ("🚀 Инициализация моделей для биннинга...")

    dino_start =time .time ()
    print ("🧠 Загрузка DINOv2 модели...")
    dino_model =build_dino ().eval ().cuda ()
    dino_time =time .time ()-dino_start
    print (f"   ⏱️ DINOv2 модель загружена за: {dino_time:.3f} сек")

    extractor_start =time .time ()
    print ("🔧 Инициализация feature extractor...")
    feature_extractor =DinoFeatureExtractor (dino_model ,resize_images =True ,crop_images =False )
    extractor_time =time .time ()-extractor_start
    print (f"   ⏱️ Feature extractor создан за: {extractor_time:.3f} сек")

    sam_start =time .time ()
    print ("🎯 Инициализация SAM-HQ Base модели...")
    sam_checkpoint ="sam-hq/pretrained_checkpoint/sam_hq_vit_b.pth"
    model_type ="vit_b"
    device ="cuda"
    sam =sam_model_registry [model_type ](checkpoint =sam_checkpoint )
    sam .to (device =device )
    sam_time =time .time ()-sam_start
    print (f"   ⏱️ SAM-HQ модель загружена за: {sam_time:.3f} сек")

    total_init_time =time .time ()-total_init_start
    print (f"🎉 Общее время инициализации биннинга: {total_init_time:.3f} сек")

    return feature_extractor ,sam

def get_dino_vector (image ,feature_extractor ):
    with torch .no_grad ():
        cls_token ,_ =feature_extractor ([image ])
    return cls_token .squeeze ().cpu ().numpy ()

def extract_features_from_masks (image ,masks ,feature_extractor ):
    features =[]
    for mask in masks :
        segmentation =mask ['segmentation']
        mask_image =np .full_like (image ,255 )
        mask_image [segmentation ]=image [segmentation ]
        pil_image =Image .fromarray (mask_image )
        features .append (get_dino_vector (pil_image ,feature_extractor ))
    return np .array (features )

def adjust_embedding (query_embedding ,positive_embeddings ,negative_embeddings ):

    positive_weights =cosine_similarity (query_embedding .reshape (1 ,-1 ),positive_embeddings ).flatten ()
    negative_weights =cosine_similarity (query_embedding .reshape (1 ,-1 ),negative_embeddings ).flatten ()

    positive_adjustment =np .sum (positive_weights [:,np .newaxis ]*positive_embeddings ,axis =0 )
    negative_adjustment =np .sum (negative_weights [:,np .newaxis ]*negative_embeddings ,axis =0 )

    combined_adjustment =positive_adjustment -negative_adjustment

    return combined_adjustment

def main ():
    total_binning_start =time .time ()
    print ("🎯 Запуск биннинг анализа...")

    pos_image_folder ='query_images_white 2012 mercedes c class front'
    neg_image_folder ='query_images_black 2011 ford fusion front'

    load_start =time .time ()
    print ("📁 Загрузка positive/negative изображений...")
    pos_image_files =[f for f in os .listdir (pos_image_folder )if f .endswith ('.jpg')or f .endswith ('.png')]
    neg_image_files =[f for f in os .listdir (neg_image_folder )if f .endswith ('.jpg')or f .endswith ('.png')]
    pos_image_files =pos_image_files [:4 ]
    neg_image_files =neg_image_files [:4 ]
    load_time =time .time ()-load_start
    print (f"   ⏱️ Загружено {len(pos_image_files)} позитивных и {len(neg_image_files)} негативных файлов за: {load_time:.3f} сек")

    feature_extractor ,sam =initialize_models ()

    pos_embed_start =time .time ()
    print ("🧠 Создание эмбеддингов для позитивных изображений...")
    pos_embeddings =[]
    for img_file in pos_image_files :
        img_path =os .path .join (pos_image_folder ,img_file )
        image =Image .open (img_path ).convert ('RGB')
        embedding =get_dino_vector (image ,feature_extractor )
        pos_embeddings .append (embedding )
    pos_embeddings =np .array (pos_embeddings )
    pos_embed_time =time .time ()-pos_embed_start
    print (f"   ⏱️ Позитивные эмбеддинги созданы за: {pos_embed_time:.3f} сек")

    neg_embed_start =time .time ()
    print ("🧠 Создание эмбеддингов для негативных изображений...")
    neg_embeddings =[]
    for img_file in neg_image_files :
        img_path =os .path .join (neg_image_folder ,img_file )
        image =Image .open (img_path ).convert ('RGB')
        embedding =get_dino_vector (image ,feature_extractor )
        neg_embeddings .append (embedding )
    neg_embeddings =np .array (neg_embeddings )
    neg_embed_time =time .time ()-neg_embed_start
    print (f"   ⏱️ Негативные эмбеддинги созданы за: {neg_embed_time:.3f} сек")

    adjust_start =time .time ()
    print ("⚖️ Корректировка эмбеддингов...")
    adjusted_embeddings =[]
    for pos_emb in pos_embeddings :
        adjusted_emb =adjust_embedding (pos_emb ,pos_embeddings ,neg_embeddings )
        adjusted_embeddings .append (adjusted_emb )
    adjusted_embeddings =np .array (adjusted_embeddings ).astype ('float32')

    adjusted_embeddings =adjusted_embeddings /np .linalg .norm (adjusted_embeddings ,axis =1 ,keepdims =True )
    adjust_time =time .time ()-adjust_start
    print (f"   ⏱️ Эмбеддинги скорректированы за: {adjust_time:.3f} сек")

    image_load_start =time .time ()
    print ("📸 Загрузка входного изображения...")
    input_image_path ='cars_multiple.jpg'
    input_image =Image .open (input_image_path ).convert ('RGB')
    input_image_np =np .array (input_image )
    image_load_time =time .time ()-image_load_start
    print (f"   ⏱️ Входное изображение загружено за: {image_load_time:.3f} сек")

    sam_gen_start =time .time ()
    print ("🔪 Генерация масок с помощью SAM-HQ...")
    mask_generator =SamAutomaticMaskGenerator (
    model =sam ,
    points_per_side =32 ,
    pred_iou_thresh =0.8 ,
    stability_score_thresh =0.9 ,
    crop_n_layers =1 ,
    crop_n_points_downscale_factor =2 ,
    min_mask_region_area =100 ,
    )
    masks =mask_generator .generate (input_image_np )
    sam_gen_time =time .time ()-sam_gen_start
    print (f"   ⏱️ SAM-HQ сгенерировал {len(masks)} масок за: {sam_gen_time:.3f} сек")

    mask_embed_start =time .time ()
    print ("🧠 Извлечение эмбеддингов для всех масок...")
    mask_embeddings =extract_features_from_masks (input_image_np ,masks ,feature_extractor )
    mask_embeddings =mask_embeddings /np .linalg .norm (mask_embeddings ,axis =1 ,keepdims =True )
    mask_embed_time =time .time ()-mask_embed_start
    print (f"   ⏱️ Эмбеддинги для {len(masks)} масок извлечены за: {mask_embed_time:.3f} сек ({mask_embed_time/len(masks)*1000:.1f} мс/маска)")

    distance_start =time .time ()
    print ("📏 Вычисление дистанций между эмбеддингами...")
    num_masks =len (mask_embeddings )
    all_distances =[]
    for i ,mask_emb in enumerate (mask_embeddings ):
        distances =[]
        for adj_emb in adjusted_embeddings :
            distance =np .linalg .norm (mask_emb -adj_emb )
            distances .append (distance )
        distances =np .array (distances )

        sorted_distances =np .sort (distances )
        all_distances .extend (sorted_distances )

        bins =np .linspace (np .min (sorted_distances ),np .max (sorted_distances ),num =5 )
        hist ,bin_edges =np .histogram (sorted_distances ,bins =bins )
        if np .any (hist >=4 ):
            print (f"Concept detected in mask {i} based on adjusted embeddings.")
    distance_time =time .time ()-distance_start
    print (f"   ⏱️ Дистанции вычислены за: {distance_time:.3f} сек")

    plt .figure (figsize =(10 ,6 ))
    plt .hist (all_distances ,bins =20 ,edgecolor ='black',alpha =0.7 )
    plt .title ('Distribution of Distances between Adjusted Embeddings and Mask Embeddings')
    plt .xlabel ('Euclidean Distance')
    plt .ylabel ('Frequency')
    plt .grid (True )
    plt .show ()

    binning1_start =time .time ()
    print ("📊 Биннинг алгоритм - Итерация 1...")

    sorted_all_distances =sorted (all_distances )
    total_distances =len (sorted_all_distances )

    num_bins =total_distances //5
    bins =[]
    for i in range (0 ,total_distances ,5 ):
        bin_distances =sorted_all_distances [i :i +5 ]
        bins .append (bin_distances )

    binning1_time =time .time ()-binning1_start
    print (f"   ⏱️ Биннинг итерация 1 завершена за: {binning1_time:.3f} сек ({num_bins} бинов создано)")

    num_bins =len (bins )
    bin_numbers =np .arange (1 ,num_bins +1 )

    bin_averages =[np .mean (bin_distances )for bin_distances in bins ]
    plt .figure (figsize =(10 ,6 ))
    plt .bar (bin_numbers ,bin_averages ,color ='skyblue')
    plt .title ('Average Distance per Bin (Iteration 1)')
    plt .xlabel ('Bin Number')
    plt .ylabel ('Average Euclidean Distance')
    plt .grid (True )
    plt .show ()

    distance_to_bin ={}
    for bin_idx ,bin_distances in enumerate (bins ):
        for distance in bin_distances :
            distance_to_bin [distance ]=bin_idx

    mask_distances =[]
    for i ,mask_emb in enumerate (mask_embeddings ):
        distances =[]
        for adj_emb in adjusted_embeddings :
            distance =np .linalg .norm (mask_emb -adj_emb )
            distances .append (distance )
        mask_distances .append ((i ,distances ))

    selected_masks =[]
    for mask_idx ,distances in mask_distances :
        bins_for_mask =[distance_to_bin [distance ]for distance in distances ]

        counts_in_bins_1_2 =sum (1 for bin_idx in bins_for_mask if bin_idx <=1 )
        if counts_in_bins_1_2 >=4 :
            print (f"Mask {mask_idx} selected (concept detected in bins 1 and 2) in Iteration 1.")
            selected_masks .append (mask_idx )
        else :
            print (f"Mask {mask_idx} not selected in Iteration 1.")

    input_image_cv =cv2 .cvtColor (input_image_np ,cv2 .COLOR_RGB2BGR )
    for mask_idx in selected_masks :
        mask =masks [mask_idx ]
        segmentation =mask ['segmentation']

        colored_mask =np .zeros_like (input_image_cv )
        colored_mask [segmentation ]=[0 ,255 ,0 ]

        overlaid_image =cv2 .addWeighted (input_image_cv ,0.7 ,colored_mask ,0.3 ,0 )

        coords =np .column_stack (np .where (segmentation ))
        y_min ,x_min =coords .min (axis =0 )
        y_max ,x_max =coords .max (axis =0 )

        cv2 .rectangle (overlaid_image ,(x_min ,y_min ),(x_max ,y_max ),(0 ,0 ,255 ),2 )

        overlaid_image_rgb =cv2 .cvtColor (overlaid_image ,cv2 .COLOR_BGR2RGB )

        plt .figure (figsize =(8 ,6 ))
        plt .imshow (overlaid_image_rgb )
        plt .title (f"Iteration 1 - Mask {mask_idx} with Bounding Box")
        plt .axis ('off')
        plt .show ()

    binning2_start =time .time ()
    print ("📊 Биннинг алгоритм - Итерация 2...")

    num_masks =len (mask_embeddings )
    all_distances =[]
    mask_distances =[]
    for i ,mask_emb in enumerate (mask_embeddings ):
        distances =[]
        for adj_emb in adjusted_embeddings :
            distance =np .linalg .norm (mask_emb -adj_emb )
            distances .append (distance )
            all_distances .append (distance )
        mask_distances .append ((i ,distances ))

    binning2_time =time .time ()-binning2_start
    print (f"   ⏱️ Биннинг итерация 2 завершена за: {binning2_time:.3f} сек")

    sorted_all_distances =sorted (all_distances )
    total_distances =len (sorted_all_distances )

    num_bins =total_distances //5
    bins =[]
    for i in range (0 ,total_distances ,5 ):
        bin_distances =sorted_all_distances [i :i +5 ]
        bins .append (bin_distances )

    num_bins =len (bins )
    bin_numbers =np .arange (1 ,num_bins +1 )

    bin_averages =[np .mean (bin_distances )for bin_distances in bins ]
    plt .figure (figsize =(10 ,6 ))
    plt .bar (bin_numbers ,bin_averages ,color ='skyblue')
    plt .title ('Average Distance per Bin (Iteration 2)')
    plt .xlabel ('Bin Number')
    plt .ylabel ('Average Euclidean Distance')
    plt .grid (True )
    plt .show ()

    distance_to_bin ={}
    for bin_idx ,bin_distances in enumerate (bins ):
        for distance in bin_distances :
            distance_to_bin [distance ]=bin_idx

    selected_masks =[]
    for mask_idx ,distances in mask_distances :
        bins_for_mask =[distance_to_bin [distance ]for distance in distances ]

        counts_in_bins_1_2 =sum (1 for bin_idx in bins_for_mask if bin_idx <=1 )
        if counts_in_bins_1_2 >=4 :
            print (f"Mask {mask_idx} selected (concept detected in bins 1 and 2) in Iteration 2.")
            selected_masks .append (mask_idx )
        else :
            print (f"Mask {mask_idx} not selected in Iteration 2.")

    input_image_cv =cv2 .cvtColor (input_image_np ,cv2 .COLOR_RGB2BGR )
    for mask_idx in selected_masks :
        mask =masks [mask_idx ]
        segmentation =mask ['segmentation']

        colored_mask =np .zeros_like (input_image_cv )
        colored_mask [segmentation ]=[0 ,255 ,0 ]

        overlaid_image =cv2 .addWeighted (input_image_cv ,0.7 ,colored_mask ,0.3 ,0 )

        coords =np .column_stack (np .where (segmentation ))
        y_min ,x_min =coords .min (axis =0 )
        y_max ,x_max =coords .max (axis =0 )

        cv2 .rectangle (overlaid_image ,(x_min ,y_min ),(x_max ,y_max ),(0 ,0 ,255 ),2 )

        overlaid_image_rgb =cv2 .cvtColor (overlaid_image ,cv2 .COLOR_BGR2RGB )

        plt .figure (figsize =(8 ,6 ))
        plt .imshow (overlaid_image_rgb )
        plt .title (f"Iteration 2 - Mask {mask_idx} with Bounding Box")
        plt .axis ('off')
        plt .show ()

    total_binning_time =time .time ()-total_binning_start
    print (f"\n🎉 ИТОГОВАЯ СТАТИСТИКА БИННИНГА:")
    print (f"   📁 Загрузка файлов: {load_time:.3f}с ({load_time/total_binning_time*100:.1f}%)")
    print (f"   🔧 Инициализация моделей: {total_init_time:.3f}с ({total_init_time/total_binning_time*100:.1f}%)")
    print (f"   🧠 Позитивные эмбеддинги: {pos_embed_time:.3f}с ({pos_embed_time/total_binning_time*100:.1f}%)")
    print (f"   🧠 Негативные эмбеддинги: {neg_embed_time:.3f}с ({neg_embed_time/total_binning_time*100:.1f}%)")
    print (f"   ⚖️ Корректировка эмбеддингов: {adjust_time:.3f}с ({adjust_time/total_binning_time*100:.1f}%)")
    print (f"   📸 Загрузка изображения: {image_load_time:.3f}с ({image_load_time/total_binning_time*100:.1f}%)")
    print (f"   🔪 SAM генерация масок: {sam_gen_time:.3f}с ({sam_gen_time/total_binning_time*100:.1f}%)")
    print (f"   🧠 Эмбеддинги масок: {mask_embed_time:.3f}с ({mask_embed_time/total_binning_time*100:.1f}%)")
    print (f"   📏 Вычисление дистанций: {distance_time:.3f}с ({distance_time/total_binning_time*100:.1f}%)")
    print (f"   📊 Биннинг итерация 1: {binning1_time:.3f}с ({binning1_time/total_binning_time*100:.1f}%)")
    print (f"   📊 Биннинг итерация 2: {binning2_time:.3f}с ({binning2_time/total_binning_time*100:.1f}%)")
    print (f"📊 ОБЩЕЕ ВРЕМЯ БИННИНГА: {total_binning_time:.3f} сек")

if __name__ =='__main__':
    main ()