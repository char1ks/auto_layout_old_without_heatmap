import argparse
import sys
from pathlib import Path
sys .path .insert (0 ,str (Path (__file__ ).parent .parent .parent ))

try :
    from ..utils .config import Config ,DEFAULT_CONFIG
except ImportError :
    try :
        from searchdet_pipeline .utils .config import Config ,DEFAULT_CONFIG
    except ImportError :
        print ("⚠️ Модули конфигурации недоступны, используем упрощенный режим")
        Config =None
        DEFAULT_CONFIG =None

def _add_detect_arguments (parser :argparse .ArgumentParser ):
    parser .add_argument ('image',help ='Путь к изображению для обработки')

    parser .add_argument ('--positive','-p',help ='Директория с положительными примерами')
    parser .add_argument ('--negative','-n',help ='Директория с отрицательными примерами')

    parser .add_argument ('--output','-o',help ='Директория для сохранения результатов')
    parser .add_argument ('--no-save',action ='store_true',help ='Не сохранять результаты на диск')

    parser .add_argument ('--config','-c',help ='Путь к файлу конфигурации JSON')
    parser .add_argument ('--config-preset',choices =['fast','balanced','quality','defect_detection'],
    help ='Предустановленная конфигурация (fast/balanced/quality/defect_detection)')
    parser .add_argument ('--backend',choices =['sam-hq','sam2','fastsam'],
    help ='Бэкенд для генерации масок')
    parser .add_argument ('--max-masks',type =int ,help ='Максимальное количество детекций')
    parser .add_argument ('--min-area',type =float ,help ='Минимальная площадь маски в процентах (0-100)')
    parser .add_argument ('--max-area',type =float ,help ='Максимальная площадь маски в процентах (0-100)')
    parser .add_argument ('--nested-iou',type =float ,help ='IoU порог для фильтра вложенных масок (0-1)')

    parser .add_argument ('--score-margin',type =float ,help ='Зазор между positive и negative скором')
    parser .add_argument ('--score-ratio',type =float ,help ='Соотношение positive/negative скора')
    parser .add_argument ('--score-confidence',type =float ,help ='Минимальная уверенность для скоринга')
    parser .add_argument ('--min-pos-score',type =float ,help ='Минимальный positive скор для принятия решения')
    parser .add_argument ('--decision-threshold',type =float ,help ='Порог разности positive-negative для принятия решения')
    parser .add_argument ('--adaptive-ratio',type =float ,help ='Коэффициент для адаптивного порога (0-1)')
    parser .add_argument ('--adaptive-diff-floor',type =float ,help ='Минимальная разность для адаптивного режима')
    parser .add_argument ('--topk',type =int ,help ='Количество топ-K примеров для агрегации')
    parser .add_argument ('--pos-agg',choices =['max','mean_topk','mean'],default ='max',
    help ='Режим агрегации positive-скоров: max (по умолчанию), mean_topk, mean')
    parser .add_argument (
    "--backbone",
    choices =[
    "resnet101",
    "dinov2_s","dinov2_b","dinov2_l","dinov2_g",
    "dinov3_vitb16","dinov3_vitl16","dinov3_vith14",
    "dinov3_convnext_tiny","dinov3_convnext_small","dinov3_convnext_base","dinov3_convnext_large",
    "vitb16","vitl16","vith14",
    "convnext_tiny","convnext_small","convnext_base","convnext_large",
    ],
    default ="dinov2_b",
    )
    parser .add_argument ("--dinov3-backbone",
    default =None ,
    choices =["vitb16","vitl16","vith14","vit7b16",
    "convnext_tiny","convnext_small",
    "convnext_base","convnext_large"],
    help ="Какой бэкенд DINOv3 использовать (ViT или ConvNeXt).")

    parser .add_argument ('--layer',help ='Слой для извлечения эмбеддингов (например, layer3)',default ='layer3')
    parser .add_argument ('--feat-short-side',type =int ,help ='Короткая сторона входа фич (например, 384/512/576)')
    parser .add_argument ('--dinov3-ckpt',help ='Путь к весам DINOv3 ConvNeXt-B (.pth)')
    parser .add_argument ('--pos-as-query-masks',action ='store_true',default =True ,
    help ='Строить q_pos из масок positive (по умолчанию: True для DINOv3)')
    parser .add_argument ('--no-pos-as-query-masks',dest ='pos_as_query_masks',action ='store_false')
    parser .add_argument ('--sam-checkpoint',help ='Путь к checkpoint SAM-HQ')
    parser .add_argument ('--sam-encoder',choices =['vit_b','vit_l','vit_h'],help ='Энкодер SAM-HQ/SAM2 (vit_b/vit_l/vit_h)')
    parser .add_argument ('--sam2-checkpoint',help ='Путь к checkpoint SAM2')
    parser .add_argument ('--sam2-config',help ='Путь к конфигурации SAM2')
    parser .add_argument ('--fastsam-checkpoint',help ='Путь к checkpoint FastSAM')
    parser .add_argument ("--vit-pooling",type =str ,choices =["cls","mean"],default ="cls",help ="Pooling for ViT (DINOv3)")
    parser .add_argument ("--loader",type =str ,default ="timm",choices =["timm","hub"],
    help ="Model loader: timm or torch.hub")
    parser .add_argument ("--repo-dir",type =str ,default =None ,
    help ="Path to local DINOv3 repository for hub loader")

    parser .add_argument ('--consensus-k',type =int ,help ='Минимум positive-попаданий для консенсуса')
    parser .add_argument ('--consensus-thr',type =float ,help ='Порог сходства для консенсуса [0-1]')
    parser .add_argument ('--nms-iou',type =float ,help ='IoU порог для NMS по боксам')

    parser .add_argument ('--sam-long-side',type =int ,help ='Даунскейл длинной стороны перед SAM/FastSAM')
    parser .add_argument ('--fastsam-imgsz',type =int ,help ='Размер входа для FastSAM')
    parser .add_argument (
    "--confidence",
    type =float ,
    default =0.3 ,
    help ="Порог уверенности для FastSAM",
    )

    parser .add_argument ('--enable-downscaling',action ='store_true',default =True ,
    help ='Включить даунскейлинг изображений для ускорения (по умолчанию: True)')
    parser .add_argument ('--disable-downscaling',dest ='enable_downscaling',action ='store_false',
    help ='Отключить даунскейлинг изображений')
    parser .add_argument ('--max-image-size',type =int ,default =512 ,
    help ='Максимальный размер изображения по длинной стороне (по умолчанию: 512)')
    parser .add_argument ('--downscale-quality',choices =['bilinear','bicubic','nearest'],default ='bilinear',
    help ='Качество ресайза изображения (по умолчанию: bilinear)')

    parser .add_argument (
    "--optimization",
    choices =["none","cutler","codet"],
    default ="none",
    help ="Оптимизация для DINOv3",
    )
    parser .add_argument (
    "--heatmap",
    action ="store_true",
    default =True ,
    help ="Сохранять heatmap",
    )
    parser .add_argument (
    "--no-heatmap",
    dest ="heatmap",
    action ="store_false",
    )
    parser .add_argument ('--fastsam-optimized',action ='store_true',
    help ='Использовать оптимизированный FastSAM режим с адаптивной производительностью (цель: <100ms)')
    parser .add_argument ('--performance-target',type =float ,default =0.1 ,
    help ='Целевое время обработки в секундах для адаптивного режима (по умолчанию: 0.1 = 100ms)')
    parser .add_argument ('--max-embedding-size',type =int ,default =1024 ,
    help ='Максимальный размер изображения для извлечения эмбеддингов (по умолчанию: 1024)')
    parser .add_argument ('--dino-half-precision',action ='store_true',
    help ='Использовать половинную точность (float16) для DINO модели для ускорения')

    parser .add_argument ('--ban-border-masks',dest ='ban_border_masks',action ='store_true',help ='Удалять маски, касающиеся рамки')
    parser .add_argument ('--no-ban-border-masks',dest ='ban_border_masks',action ='store_false',help ='Разрешить маски, касающиеся рамки')
    parser .set_defaults (ban_border_masks =True )
    parser .add_argument ('--border-width',type =int ,help ='Толщина рамки для фильтра границ (px)')

    parser .add_argument ('--device',default ='cuda',help ='Устройство для выполнения (например, cuda или cpu)')
    parser .add_argument ('--half',action ='store_true',help ='Использовать половинную точность (float16)')
    parser .add_argument ('--verbose','-v',action ='store_true',help ='Подробный вывод')
    parser .add_argument ('--quiet','-q',action ='store_true',help ='Минимальный вывод')
    parser .add_argument ('--defect',action ='store_true',help ='Включить режим поиска дефектов (beta)')
    parser .add_argument ('--use-heatmap',action ='store_true',help ='Использовать heatmap режим вместо SAM/FastSAM для генерации масок')
    parser .add_argument ('--use-heatmap-masks',action ='store_true',default =True ,help ='Генерировать маски из heatmap (по умолчанию: True)')
    parser .add_argument ('--no-heatmap-masks',dest ='use_heatmap_masks',action ='store_false',help ='Использовать SAM даже в heatmap режиме')

    parser .add_argument ('--fastsam-refinement-iou',type =float ,default =0.3 ,
    help ='Порог IoU для поиска соответствий при FastSAM постобработке (по умолчанию: 0.3)')
    parser .add_argument ('--use-heatmap-sam-hybrid',action ='store_true',help ='Использовать гибридный подход heatmap+SAM')
    parser .add_argument ('--heatmap-sam-threshold',type =float ,default =0.7 ,help ='Порог для извлечения горячих зон из heatmap (по умолчанию: 0.7)')
    parser .add_argument ('--sam-refinement-enabled',action ='store_true',default =True ,help ='Включить уточнение через SAM (по умолчанию: True)')
    parser .add_argument ('--max-hotspots-for-sam',type =int ,default =10 ,help ='Максимальное количество горячих зон для обработки SAM (по умолчанию: 10)')
    parser .add_argument ('--use-fastsam-with-heatmap',action ='store_true',help ='Включить FastSAM с горячими точками в heatmap пайплайне')

def execute_detect (args )->int :
    print ("\n"+"="*70 )
    print ("="*70 )
    print ("1️⃣ main.py → searchdet_pipeline.cli.main.main()")
    print ("2️⃣ searchdet_pipeline/cli/main.py → _execute_detect()")

    import sys
    import os
    sys .path .insert (0 ,str (Path (__file__ ).parent .parent .parent ))

    try:
        from ..core.detector import SearchDetDetector
        from ..core.config import get_preset_config
    except ImportError :
        try :
            from searchdet_pipeline.core.detector import SearchDetDetector
            from searchdet_pipeline.core.config import get_preset_config
        except ImportError :
            print ("❌ Ошибка импорта модульного пайплайна: не удается импортировать SearchDetDetector или get_preset_config")
            return 1

    # Импорты выполнены успешно, продолжаем инициализацию детектора
    print ("🔧 Инициализация SearchDet детектора...")
    use_heatmap =getattr (args ,'use_heatmap',False )
    use_heatmap_masks =getattr (args ,'use_heatmap_masks',True )
    use_heatmap_sam_hybrid =getattr (args ,'use_heatmap_sam_hybrid',False )
    use_fastsam_optimized =getattr (args ,'fastsam_optimized',False )
    use_fastsam_with_heatmap =getattr (args ,'use_fastsam_with_heatmap',False )

    if hasattr (args ,'config_preset')and args .config_preset :
        print (f"📋 Применение пресета конфигурации: {args.config_preset}")
        preset_config =get_preset_config (args .config_preset )
        detector_params ={
        'mask_backend':preset_config .mask_backend .value ,
        'device':args .device ,
        'half':preset_config .half_precision ,
        'min_confidence':preset_config .min_confidence ,
        'max_masks':preset_config .max_masks ,
        'backbone':preset_config .backbone .value ,
        'min_area_frac':preset_config .min_area_fraction ,
        'max_area_frac':preset_config .max_area_fraction ,
        'smart_rectangle_filter':preset_config .smart_rectangle_filter ,
        'defect_mode':preset_config .defect_mode ,
        }
        if preset_config .sam_long_side :
            detector_params ['sam_long_side']=preset_config .sam_long_side
        if preset_config .fastsam_image_size :
            detector_params ['fastsam_imgsz']=preset_config .fastsam_image_size

        detector_params ['enable_image_downscaling']=getattr (args ,'enable_downscaling',True )
        detector_params ['max_image_size']=getattr (args ,'max_image_size',512 )
        detector_params ['downscale_quality']=getattr (args ,'downscale_quality','bilinear')

        detector_params ['use_fastsam_with_heatmap']=use_fastsam_with_heatmap
    else :
         detector_params ={
         'mask_backend':'fastsam',
         "positive_aggregation":"max",
         'dinov3_backbone':"vit7b16",
         "layer":"layer3",
         'pos_as_query_masks':True ,
         'vit_pooling':'cls',
         "loader":"timm",
         "repo_dir":None ,
         "max_embedding_size":1024 ,
         'device':"cuda",
         'encoder_device':"cuda",
         "use-heatmap-masks":False ,
         'half':True ,
         'dinov3_ckpt':None ,
         'dino_half_precision':False ,
         'backbone':"dinov3_vitb16",
         'min_mask_area':100 ,
         'smart_rectangle_filter':True ,
         'rectangle_bbox_iou_threshold':0.95 ,
         'rectangle_straight_line_ratio':0.8 ,
         'rectangle_area_ratio_threshold':0.95 ,
         'rectangle_angle_tolerance':10.0 ,
         'rectangle_side_ratio_threshold':0.9 ,
         'perfect_rectangle_iou_threshold':0.99 ,
         'rectangle_similarity_iou_threshold':0.94 ,
         'square_similarity_iou_threshold':0.94 ,
         'rectangle_use_silhouette':True ,
         'hole_area_ratio_threshold':0.03 ,
         'min_positive_score':0.53 ,
         'decision_threshold':0.5 ,
         'enable_image_downscaling':getattr (args ,'enable_downscaling',True ),
         'max_image_size':getattr (args ,'max_image_size',512 ),
         'downscale_quality':getattr (args ,'downscale_quality','bilinear'),
         'use_fastsam_with_heatmap':use_fastsam_with_heatmap ,
         }

    print ("5️⃣ Создание: detector = SearchDetDetector(**params)")
    detector =SearchDetDetector (**detector_params )

    import os
    if hasattr (args ,'positive')and args .positive :
        args .positive =os .path .abspath (args .positive )
    if hasattr (args ,'negative')and args .negative :
        args .negative =os .path .abspath (args .negative )
    if hasattr (args ,'image')and args .image :
        args .image =os .path .abspath (args .image )

    if use_heatmap_sam_hybrid :
        detector_params .update ({
        'use_heatmap_sam_hybrid':True ,
        'heatmap_sam_threshold':getattr (args ,'heatmap_sam_threshold',0.7 ),
        'sam_refinement_enabled':getattr (args ,'sam_refinement_enabled',True ),
        'max_hotspots_for_sam':getattr (args ,'max_hotspots_for_sam',10 )
        })

    if use_fastsam_optimized :

        max_emb_size =getattr (args ,'max_embedding_size',None )
        if max_emb_size =='None'or max_emb_size =='none':
            max_emb_size =None
        elif isinstance (max_emb_size ,str )and max_emb_size .isdigit ():
            max_emb_size =int (max_emb_size )

        detector_params .update ({
        'performance_target':getattr (args ,'performance_target',0.1 ),
        'max_embedding_size':max_emb_size ,
        'dino_half_precision':getattr (args ,'dino_half_precision',False )
        })

        print ("6️⃣ Вызов: detector.find_present_elements() [FASTSAM+HEATMAP РЕЖИМ]")
        print ("   ↳ Это запустит FastSAM+heatmap пайплайн:")
        print ("   ↳ _load_example_images() → HeatmapGenerator → FastSAM (горячие точки) → фильтрация по совпадению 80% → результат")
    
    # Вызов детектора для всех режимов
    result =detector .find_present_elements (
    args .image ,
    args .positive ,
    args .negative ,
    args .output if hasattr (args ,'output')and args .output else "output"
    )
    _print_result (result ,args )
    if 'success'in result :
        return 0 if result ['success']else 1
    else :
        return 0 if 'found_elements'in result else 1

def _print_result (result :dict ,args ):

    if 'success'in result :

        if result ['success']:
            print (f"\n✅ Детекция завершена успешно!")
            print (f"⏱️ Время обработки: {result['processing_time']:.2f} сек")
            print (f"🔍 Найдено объектов: {len(result['detections'])}")

            if result ['detections']:
                confidences =[d ['confidence']for d in result ['detections']]
                print (f"📊 Средняя уверенность: {sum(confidences)/len(confidences):.3f}")

            if result .get ('saved_files'):
                print (f"💾 Сохранено файлов: {len(result['saved_files'])}")
                if hasattr (args ,'verbose')and args .verbose :
                    for file_type ,path in result ['saved_files'].items ():
                        print (f"   • {file_type}: {path}")
        else :
            print (f"\n❌ Ошибка детекции: {result.get('error', 'Неизвестная ошибка')}")
    else :

        if 'found_elements'in result :
            found_elements =result ['found_elements']
            print (f"\n✅ Детекция завершена успешно!")
            print (f"🔍 Найдено объектов: {len(found_elements)}")

            if found_elements :
                confidences =[elem .get ('confidence',0.0 )for elem in found_elements ]
                if any (c >0 for c in confidences ):
                    print (f"📊 Средняя уверенность: {sum(confidences)/len(confidences):.3f}")

            output_dir =result .get ('output_directory','output')
            print (f"💾 Результаты сохранены в: {output_dir}")

            if 'saved_files'in result and result ['saved_files']:
                saved_files =result ['saved_files']
                print (f"📁 Сохранено файлов: {len(saved_files)}")
                if hasattr (args ,'verbose')and args .verbose :
                    print ("   📋 Список файлов:")
                    for file_type ,file_path in saved_files .items ():
                        print (f"     • {file_type}: {Path(file_path).name}")

            if 'timing_info'in result and (hasattr (args ,'verbose')and args .verbose ):
                timing_info =result ['timing_info']
                print (f"\n⏱️ КРАТКАЯ СТАТИСТИКА ВРЕМЕНИ:")
                if 'mask_generation'in timing_info :
                    print (f"   🎯 Генерация масок: {timing_info['mask_generation']:.3f}с")
                if 'embedding_extraction'in timing_info :
                    print (f"   🧠 Извлечение эмбеддингов: {timing_info['embedding_extraction']:.3f}с")
                if 'scoring_and_decisions'in timing_info :
                    print (f"   📊 Скоринг: {timing_info['scoring_and_decisions']:.3f}с")
                if 'result_saving'in timing_info :
                    print (f"   💾 Сохранение: {timing_info['result_saving']:.3f}с")
        else :
            print (f"\n❌ Неожиданный формат результата: {result}")