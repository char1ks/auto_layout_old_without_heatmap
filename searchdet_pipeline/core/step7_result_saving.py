import cv2
import json
import numpy as np
from pathlib import Path
from typing import List ,Dict ,Any ,Optional
from datetime import datetime
from PIL import Image

from ..utils.helpers import save_json
from .binning_processor import BinningProcessor

class ResultSaver :

    def __init__ (self ,overlay_alpha :float =0.5 ,binning_processor :Optional [BinningProcessor ]=None ,
    fast_mode :bool =False ,save_visualizations :bool =True ,
    save_individual_masks :bool =True ,save_heatmap_debug :bool =True ,
    use_flat_output :bool =False ,skip_folder_creation :bool =False ):

        self .overlay_alpha =overlay_alpha
        self .colors =self ._generate_colors ()
        self .binning_processor =binning_processor
        self .fast_mode =fast_mode
        self .save_visualizations =save_visualizations
        self .save_individual_masks =save_individual_masks
        self .save_heatmap_debug =save_heatmap_debug
        self .use_flat_output =use_flat_output
        self .skip_folder_creation =skip_folder_creation

    def _create_output_folders (self ,result_dir :Path )->Dict [str ,Path ]:
        return {
        'heatmap':result_dir ,
        'result':result_dir
        }

    def _safe_draw_contours (self ,image :np .ndarray ,contours :list ,
    contour_idx :int =-1 ,color :tuple =(0 ,255 ,0 ),
    thickness :int =2 )->np .ndarray :
        if contours is None or len (contours )==0 :
            return image

        valid_contours =[]
        for c in contours :
            if c is not None and len (c )>0 :
                if isinstance (c ,np .ndarray )and c .size >0 :
                    if len (c )>=3 :
                        valid_contours .append (c )

        if not valid_contours :
            return image

        try :
            cv2 .drawContours (image ,valid_contours ,contour_idx ,color ,thickness )
        except Exception as e :
            print (f"   ⚠️ Ошибка при отрисовке контуров: {e}")

        return image

    def save_all_results (self ,image :np .ndarray ,final_masks :List [Dict [str ,Any ]],
    output_dir :str ,image_name :str ,
    pipeline_config :Optional [Dict ]=None ,
    heatmap :Optional [np .ndarray ]=None ,
    positive_images :Optional [List [Image .Image ]]=None ,
    negative_images :Optional [List [Image .Image ]]=None ,
    results_file_name :Optional [str ]=None ,
    class_info :Optional [Dict [str ,List ]]=None ,
    original_size :Optional [tuple ]=None )->Dict [str ,str ]:
        print ("\n🔄 ЭТАП 7: СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
        print ("="*60 )

        if results_file_name and class_info :
            base_dir =Path (output_dir )/results_file_name
            base_dir .mkdir (parents =True ,exist_ok =True )
            print (f"   📁 Создание структуры по классам в {base_dir}")

            masks_by_class ={}
            for mask in final_masks :
                class_name =mask .get ('class','unknown')
                if class_name not in masks_by_class :
                    masks_by_class [class_name ]=[]
                masks_by_class [class_name ].append (mask )

            saved_files ={}

            for class_name in class_info .keys ():
                class_dir =base_dir /class_name
                class_dir .mkdir (parents =True ,exist_ok =True )

                class_masks =masks_by_class .get (class_name ,[])
                print (f"   📂 Класс '{class_name}': {len(class_masks)} масок")

                if heatmap is not None :
                    class_saved =self ._save_heatmap_for_class (heatmap ,class_dir ,class_name ,original_size )
                    saved_files .update (class_saved )

                overlay_saved =self ._save_overlay_for_class (image ,class_masks ,class_dir ,class_name )
                saved_files .update (overlay_saved )

                annotations_saved =self ._save_annotations_for_class (
                image ,class_masks ,class_dir ,image_name ,pipeline_config ,class_name
                )
                saved_files .update (annotations_saved )
                
            # Сохраняем overlay_mask для всех классов
            overlay_mask_saved = self._save_overlay_mask(image, final_masks, None, base_dir)
            saved_files.update(overlay_mask_saved)
        else :
            # Создаем подпапку results_{image_name}
            base_name = Path(image_name).stem if image_name else "result"
            result_dir = Path(output_dir) / f"results_{base_name}"
            result_dir.mkdir(parents=True, exist_ok=True)
            print(f"   ⚡ Быстрое сохранение в {result_dir}")

            saved_files = {}

            # Сохранение полной heatmap только с bin1
            bin1_mask_global = None
            if heatmap is not None:
                print("   🔥 Сохранение полной heatmap только с bin1...")
                try:
                    # Конвертируем heatmap
                    if 'torch' in str(type(heatmap)):
                        heat = heatmap.detach().cpu().numpy()
                    else:
                        heat = np.array(heatmap)
                    heat = np.squeeze(heat)
                    
                    if heat.size > 0:
                        # Создание только bin1 (топ бин)
                        valid_mask = heat > 0
                        if np.any(valid_mask):
                            valid_values = heat[valid_mask]
                            bin_edges = np.quantile(valid_values, np.linspace(0, 1, 5))
                            bin1_mask = heat >= bin_edges[-2]  # Топ бин
                            
                            # Сохраняем маску для использования с FastSAM
                            if original_size is not None:
                                bin1_mask_global = cv2.resize(bin1_mask.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST).astype(bool)
                            else:
                                bin1_mask_global = cv2.resize(bin1_mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                            
                            # Создаем overlay поверх исходного изображения только с bin1
                            overlay_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).copy()
                            
                            # Ресайзим маску до размера изображения
                            if original_size is not None:
                                bin1_resized = cv2.resize(bin1_mask.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST).astype(bool)
                            else:
                                bin1_resized = cv2.resize(bin1_mask.astype(np.uint8), (overlay_img.shape[1], overlay_img.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                            
                            # Применяем красный цвет только для bin1
                            red_overlay = overlay_img.copy()
                            
                            # Bin1 - красный цвет
                            red_overlay[bin1_resized] = [0, 0, 255]  # BGR формат
                            overlay_img[bin1_resized] = cv2.addWeighted(overlay_img[bin1_resized], 0.6, red_overlay[bin1_resized], 0.4, 0)
                            
                            heatmap_path = result_dir / "full_heatmap.png"
                            cv2.imwrite(str(heatmap_path), overlay_img)
                            saved_files['full_heatmap'] = str(heatmap_path)
                            print(f"     🔥 Full heatmap overlay: bin1 (красный) {np.sum(bin1_resized)} пикселей")
                            
                            heat_global_normalized = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX)
                            
                            # Бин 1
                            if np.any(bin1_mask):
                                bin1_heatmap = np.zeros_like(heat_global_normalized)
                                bin1_heatmap[bin1_mask] = heat_global_normalized[bin1_mask]
                                bin1_colored = cv2.applyColorMap(bin1_heatmap.astype(np.uint8), cv2.COLORMAP_HOT)
                                if original_size is not None:
                                    bin1_colored = cv2.resize(bin1_colored, original_size, interpolation=cv2.INTER_LINEAR)
                                bin1_path = result_dir / "bin1_top.png"
                                cv2.imwrite(str(bin1_path), bin1_colored)
                                saved_files['bin1_top'] = str(bin1_path)
                            
                            # Отключено сохранение bin2 - отображаем только bin1
                except Exception as e:
                    print(f"   ❌ Ошибка при сохранении heatmap: {e}")

            # Создаем папку для масок FastSAM
            fastsam_dir = result_dir / "fastsam_masks"
            fastsam_dir.mkdir(parents=True, exist_ok=True)
            
            if not final_masks:
                print("   ⚠️ Нет детекций для сохранения")
                saved_files.update(self._save_annotations_only(
                    image, [], image_name, pipeline_config, result_dir
                ))
                # Сохраняем только красные зоны из heatmap как финальную маску
                if heatmap is not None:
                    print("   🎨 Создание финальной маски из heatmap...")
                    overlay_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).copy()
                    try:
                        if 'torch' in str(type(heatmap)):
                            heat = heatmap.detach().cpu().numpy()
                        else:
                            heat = np.array(heatmap)
                        heat = np.squeeze(heat)
                        if heat.size > 0:
                            if original_size is not None:
                                heat_resized = cv2.resize(heat, original_size, interpolation=cv2.INTER_LINEAR)
                            else:
                                heat_resized = cv2.resize(heat, (overlay_img.shape[1], overlay_img.shape[0]), interpolation=cv2.INTER_LINEAR)
                            heat_normalized = cv2.normalize(heat_resized, None, 0, 1, cv2.NORM_MINMAX)
                            active_mask = heat_normalized > 0.3
                            red_overlay = overlay_img.copy()
                            red_overlay[active_mask] = [0, 0, 255]
                            overlay_img[active_mask] = cv2.addWeighted(overlay_img[active_mask], 0.7, red_overlay[active_mask], 0.3, 0)
                            print(f"     🔴 Финальная маска из heatmap: {np.sum(active_mask)} пикселей")
                    except Exception as e:
                        print(f"   ⚠️ Ошибка при создании финальной маски: {e}")
                    # Создаем и сохраняем overlay_mask с heatmap bin1
                    overlay_mask_saved = self._save_overlay_mask(image, [], bin1_mask_global, result_dir)
                    saved_files.update(overlay_mask_saved)
            else:
                print(f"   💾 Быстрое сохранение {len(final_masks)} детекций...")

                saved_files.update(self._save_annotations_only(
                    image, final_masks, image_name, pipeline_config, result_dir
                ))

                # Фильтруем маски FastSAM только по bin1
                if bin1_mask_global is not None:
                    print("   🔍 Фильтрация масок FastSAM только по bin1...")
                    filtered_masks = []
                    
                    for mask in final_masks:
                        segmentation = mask['segmentation']
                        seg_bool = segmentation.astype(bool) if segmentation.dtype != bool else segmentation
                        
                        # Проверяем совместимость размеров для операции пересечения
                        if seg_bool.shape != bin1_mask_global.shape:
                            print(f"     ⚠️ Несовпадение размеров: маска {seg_bool.shape} vs bin1 {bin1_mask_global.shape}")
                            # Изменяем размер bin1_mask_global под размер маски
                            bin1_resized = cv2.resize(bin1_mask_global.astype(np.uint8), 
                                                    (seg_bool.shape[1], seg_bool.shape[0]), 
                                                    interpolation=cv2.INTER_NEAREST).astype(bool)
                            intersection = seg_bool & bin1_resized
                        else:
                            # Пересечение маски FastSAM только с областью bin1
                            intersection = seg_bool & bin1_mask_global
                        
                        intersection_ratio = np.sum(intersection) / np.sum(seg_bool) if np.sum(seg_bool) > 0 else 0
                        
                        if intersection_ratio > 0.1:  # Оставляем маски с пересечением > 10%
                            # Сохраняем оригинальную маску FastSAM без обрезки
                            filtered_mask = mask.copy()
                            # НЕ обрезаем маску - оставляем оригинальную segmentation
                            filtered_masks.append(filtered_mask)
                            print(f"     ✅ Маска класса '{mask.get('class', 'unknown')}': пересечение с bin1 {intersection_ratio:.2%}, сохранена оригинальная маска {np.sum(seg_bool)} пикселей")
                        else:
                            print(f"     ❌ Маска класса '{mask.get('class', 'unknown')}': пересечение с bin1 {intersection_ratio:.2%}, отфильтрована")
                    
                    print(f"   📊 Отфильтровано {len(filtered_masks)} из {len(final_masks)} масок по bin1")
                    final_masks = filtered_masks
                
                # Сохраняем маски FastSAM в отдельную папку
                print("   📁 Сохранение отфильтрованных масок FastSAM...")
                fastsam_overlay = self._create_overlay_visualization(image, final_masks)
                fastsam_overlay_path = fastsam_dir / "fastsam_overlay.png"
                cv2.imwrite(str(fastsam_overlay_path), fastsam_overlay)
                saved_files['fastsam_overlay'] = str(fastsam_overlay_path)
                
                # Сохраняем каждую FastSAM маску отдельно
                individual_fastsam_saved = self._save_individual_fastsam_masks(final_masks, result_dir, image_name)
                saved_files.update(individual_fastsam_saved)
                
                # Сохраняем каждую FastSAM маску с overlay (тепловая карта)
                individual_overlay_saved = self._save_individual_fastsam_overlays(image, final_masks, result_dir, image_name)
                saved_files.update(individual_overlay_saved)
                
                # Создаем финальную маску - только отфильтрованные FastSAM
                print("   🎨 Создание финальной маски от отфильтрованных FastSAM...")
                overlay_img = self._create_overlay_visualization(image, final_masks)
                overlay_path = result_dir / "overlay_masks.png"
                cv2.imwrite(str(overlay_path), overlay_img)
                saved_files['overlay'] = str(overlay_path)
                print(f"     ✅ Финальная маска содержит {len(final_masks)} отфильтрованных масок FastSAM")
                
                # Создаем и сохраняем overlay_mask
                overlay_mask_saved = self._save_overlay_mask(image, final_masks, bin1_mask_global, result_dir)
                saved_files.update(overlay_mask_saved)
                
                # Создание contours_only с толстыми контурами
                print("   🔲 Создание contours_only визуализации...")
                contours_img = self._create_contours_only_visualization(image, final_masks)
                contours_path = result_dir / "contours_only.png"
                cv2.imwrite(str(contours_path), contours_img)
                saved_files['contours_only'] = str(contours_path)
                
                # Сохранение итоговых отфильтрованных масок
                print("   📋 Сохранение итоговых отфильтрованных масок...")
                masks_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
                class_colors = {}
                unique_classes = list(set(mask.get('class', 'unknown') for mask in final_masks))
                for i, cls in enumerate(unique_classes):
                    class_colors[cls] = self.colors[i % len(self.colors)]
                for mask in final_masks:
                    cls = mask.get('class', 'unknown')
                    color = class_colors[cls]
                    segmentation = mask['segmentation']
                    seg_bool = segmentation.astype(bool) if segmentation.dtype != bool else segmentation
                    if seg_bool.shape != (masks_img.shape[0], masks_img.shape[1]):
                        print(f"     ⚠️ Несовпадение размеров: маска {seg_bool.shape} vs изображение {(masks_img.shape[0], masks_img.shape[1])}")
                        seg_bool_resized = cv2.resize(seg_bool.astype(np.uint8), 
                                                    (masks_img.shape[1], masks_img.shape[0]), 
                                                    interpolation=cv2.INTER_NEAREST).astype(bool)
                        masks_img[seg_bool_resized] = color
                    else:
                        masks_img[seg_bool] = color
                masks_path = result_dir / "final_masks.png"
                cv2.imwrite(str(masks_path), masks_img)
                saved_files['final_masks'] = str(masks_path)
                print(f"     💾 Сохранены отфильтрованные маски для {len(unique_classes)} классов")

        print (f"   ✅ Сохранено {len(saved_files)} файлов")
        return saved_files

    def _save_overlay_mask(self, image: np.ndarray, final_masks: List[Dict[str, Any]], 
                          bin1_mask: Optional[np.ndarray], result_dir: Path) -> Dict[str, str]:
        """
        Создает и сохраняет overlay_mask с наложением на оригинальное изображение
        """
        print("   🎭 Создание overlay_mask...")
        saved_files = {}
        
        # Создаем копию оригинального изображения
        overlay_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).copy()
        
        if final_masks:
            # Если есть маски FastSAM, используем их
            print(f"     📋 Используем {len(final_masks)} масок FastSAM")
            
            # Создаем общую маску из всех FastSAM масок
            combined_mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
            for mask in final_masks:
                segmentation = mask['segmentation']
                seg_bool = segmentation.astype(bool) if segmentation.dtype != bool else segmentation
                
                # Проверяем совместимость размеров
                if seg_bool.shape != combined_mask.shape:
                    print(f"     ⚠️ Несовпадение размеров в overlay: маска {seg_bool.shape} vs изображение {combined_mask.shape}")
                    seg_bool = cv2.resize(seg_bool.astype(np.uint8), 
                                        (combined_mask.shape[1], combined_mask.shape[0]), 
                                        interpolation=cv2.INTER_NEAREST).astype(bool)
                
                combined_mask |= seg_bool
            
            # Применяем сглаживание к маске
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            smoothed_mask = cv2.morphologyEx(combined_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            smoothed_mask = cv2.GaussianBlur(smoothed_mask.astype(np.float32), (5, 5), 1.0)
            smoothed_mask = (smoothed_mask > 0.5).astype(bool)
            
            # Создаем красное наложение
            red_overlay = overlay_image.copy()
            
            # Проверяем совместимость размеров для наложения
            if smoothed_mask.shape != (overlay_image.shape[0], overlay_image.shape[1]):
                print(f"     ⚠️ Несовпадение размеров маски overlay: {smoothed_mask.shape} vs {(overlay_image.shape[0], overlay_image.shape[1])}")
                smoothed_mask = cv2.resize(smoothed_mask.astype(np.uint8), 
                                         (overlay_image.shape[1], overlay_image.shape[0]), 
                                         interpolation=cv2.INTER_NEAREST).astype(bool)
            
            red_overlay[smoothed_mask] = [0, 0, 255]  # BGR формат - красный
            
            # Смешиваем с оригинальным изображением
            overlay_image[smoothed_mask] = cv2.addWeighted(
                overlay_image[smoothed_mask], 0.6, 
                red_overlay[smoothed_mask], 0.4, 0
            )
            
            print(f"     ✅ Overlay mask содержит {len(final_masks)} масок FastSAM")
            
        elif bin1_mask is not None:
            # Если нет масок FastSAM, используем heatmap bin1
            print("     🔥 Используем heatmap bin1")
            
            # Применяем сглаживание к bin1 маске
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            smoothed_mask = cv2.morphologyEx(bin1_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            smoothed_mask = cv2.GaussianBlur(smoothed_mask.astype(np.float32), (7, 7), 2.0)
            smoothed_mask = (smoothed_mask > 0.3).astype(bool)
            
            # Создаем красное наложение
            red_overlay = overlay_image.copy()
            
            # Проверяем совместимость размеров для bin1 наложения
            if smoothed_mask.shape != (overlay_image.shape[0], overlay_image.shape[1]):
                print(f"     ⚠️ Несовпадение размеров bin1 overlay: {smoothed_mask.shape} vs {(overlay_image.shape[0], overlay_image.shape[1])}")
                smoothed_mask = cv2.resize(smoothed_mask.astype(np.uint8), 
                                         (overlay_image.shape[1], overlay_image.shape[0]), 
                                         interpolation=cv2.INTER_NEAREST).astype(bool)
            
            red_overlay[smoothed_mask] = [0, 0, 255]  # BGR формат - красный
            
            # Смешиваем с оригинальным изображением
            overlay_image[smoothed_mask] = cv2.addWeighted(
                overlay_image[smoothed_mask], 0.7, 
                red_overlay[smoothed_mask], 0.3, 0
            )
            
            print(f"     ✅ Overlay mask содержит heatmap bin1: {np.sum(smoothed_mask)} пикселей")
            
        else:
            # Если нет ни масок FastSAM, ни bin1, сохраняем оригинальное изображение
            print("     ⚠️ Нет данных для overlay_mask, сохраняется оригинальное изображение")
        
        # Сохраняем overlay_mask с наложением
        overlay_mask_path = result_dir / "overlay_mask.png"
        cv2.imwrite(str(overlay_mask_path), overlay_image)
        saved_files['overlay_mask'] = str(overlay_mask_path)
        
        print(f"     💾 Сохранен overlay_mask с наложением на оригинальное изображение")
        return saved_files

    def _save_heatmap_for_class (self ,heatmap :np .ndarray ,class_dir :Path ,class_name :str ,original_size :Optional [tuple ]=None )->Dict [str ,str ]:
        print (f"   💾 Сохранение heatmap для класса '{class_name}'...")

        saved_files ={}

        try :
            if 'torch'in str (type (heatmap )):
                try :
                    heat =heatmap .detach ().cpu ().numpy ()
                except Exception :
                    heat =np .array (heatmap )
            else :
                heat =np .array (heatmap )
            heat =np .squeeze (heat )

            if heat .size ==0 :
                print (f"   ⚠️ Пустой heatmap для класса '{class_name}'")
                return saved_files
            heatmap_normalized =cv2 .normalize (heat ,None ,0 ,255 ,cv2 .NORM_MINMAX )
            heatmap_colored =cv2 .applyColorMap (heatmap_normalized .astype (np .uint8 ),cv2 .COLORMAP_HOT )
            if original_size is not None :
                heatmap_colored =cv2 .resize (heatmap_colored ,original_size ,interpolation =cv2 .INTER_LINEAR )
                print (f"     📐 Ресайз heatmap: {heat.shape} -> {original_size}")
            heatmap_path =class_dir /f"heatmap_{class_name}.png"
            cv2 .imwrite (str (heatmap_path ),heatmap_colored )
            saved_files [f'heatmap_{class_name}']=str (heatmap_path )
            try :
                valid_mask =heat >0
                if np .any (valid_mask ):
                    valid_values =heat [valid_mask ]
                    num_bins =4
                    bin_edges =np .quantile (valid_values ,np .linspace (0 ,1 ,num_bins +1 ))
                    bin_edges [0 ]=float (valid_values .min ())-1e-6
                    bin_edges [-1 ]=float (valid_values .max ())+1e-6
                    bin1_mask =heat >=bin_edges [-2 ]
                    # Отображаем только bin1
                    print (f"     📊 Статистика bin1:")
                    print (f"        Бин1 (топ): {np.sum(bin1_mask)} пикселей")
                    print (f"        Границы бинов: {bin_edges}")
                    print (f"        Диапазон значений: {valid_values.min():.4f} - {valid_values.max():.4f}")
                    heat_global_normalized =cv2 .normalize (heat ,None ,0 ,255 ,cv2 .NORM_MINMAX )

                    if np .any (bin1_mask ):
                        bin1_heatmap =np .zeros_like (heat_global_normalized )
                        bin1_heatmap [bin1_mask ]=heat_global_normalized [bin1_mask ]
                        bin1_colored =cv2 .applyColorMap (bin1_heatmap .astype (np .uint8 ),cv2 .COLORMAP_HOT )
                        if original_size is not None :
                            bin1_colored =cv2 .resize (bin1_colored ,original_size ,interpolation =cv2 .INTER_LINEAR )

                        bin1_path =class_dir /f"bin1_{class_name}.png"
                        cv2 .imwrite (str (bin1_path ),bin1_colored )
                        saved_files [f'bin1_{class_name}']=str (bin1_path )
                        print (f"     💾 Сохранен Бин1: {np.sum(bin1_mask)} пикселей")

                    # Отключено сохранение bin2 - отображаем только bin1
                    # if np .any (bin2_mask ):
                    #     bin2_heatmap =np .zeros_like (heat_global_normalized )
                    #     bin2_heatmap [bin2_mask ]=heat_global_normalized [bin2_mask ]
                    #     bin2_colored =cv2 .applyColorMap (bin2_heatmap .astype (np .uint8 ),cv2 .COLORMAP_HOT )

                    #     if original_size is not None :
                    #         bin2_colored =cv2 .resize (bin2_colored ,original_size ,interpolation =cv2 .INTER_LINEAR )

                    #     bin2_path =class_dir /f"bin2_{class_name}.png"
                    #     cv2 .imwrite (str (bin2_path ),bin2_colored )
                    #     saved_files [f'bin2_{class_name}']=str (bin2_path )
                    #     print (f"     💾 Сохранен Бин2: {np.sum(bin2_mask)} пикселей")

            except Exception as e :
                print (f"   ⚠️ Ошибка при создании отдельных бинов для класса '{class_name}': {e}")

            if self .binning_processor is not None :
                try :

                    binary_mask =self .binning_processor ._hysteresis_mask (heat )

                    contours_img =np .zeros ((heat .shape [0 ],heat .shape [1 ],3 ),dtype =np .uint8 )

                    result_contours =cv2 .findContours (
                    binary_mask .astype (np .uint8 ),
                    cv2 .RETR_EXTERNAL ,
                    cv2 .CHAIN_APPROX_SIMPLE
                    )
                    if len (result_contours )==3 :
                        _ ,contours ,_ =result_contours
                    else :
                        contours ,_ =result_contours

                    if contours :
                        cv2 .drawContours (contours_img ,contours ,-1 ,(0 ,255 ,0 ),2 )

                    contours_path =class_dir /f"contours_only_{class_name}.png"
                    cv2 .imwrite (str (contours_path ),contours_img )
                    saved_files [f'contours_only_{class_name}']=str (contours_path )

                except Exception as e :
                    print (f"   ⚠️ Ошибка при создании contours_only для класса '{class_name}': {e}")

        except Exception as e :
            print (f"   ❌ Ошибка при сохранении heatmap для класса '{class_name}': {e}")

        return saved_files

    def _save_overlay_for_class (self ,image :np .ndarray ,class_masks :List [Dict [str ,Any ]],
    class_dir :Path ,class_name :str )->Dict [str ,str ]:
        saved_files ={}

        try :
            bin1_path =class_dir /f"bin1_{class_name}.png"

            # Отображаем только bin1
            if bin1_path .exists ():
                overlay_img =self ._create_overlay_with_bins (image ,bin1_path )
                print (f"     ✅ Создан overlay с bin1 из heatmap")
            else :

                overlay_img =self ._create_overlay_visualization (image ,class_masks )
                print (f"     ⚠️ Heatmap bin1 не найден, используем маски объектов")

            overlay_path =class_dir /f"overlay_{class_name}.png"
            cv2 .imwrite (str (overlay_path ),overlay_img )
            saved_files [f'overlay_{class_name}']=str (overlay_path )

        except Exception as e :
            print (f"   ❌ Ошибка при создании overlay для класса '{class_name}': {e}")

        return saved_files

    def _save_annotations_for_class (self ,image :np .ndarray ,class_masks :List [Dict [str ,Any ]],
    class_dir :Path ,image_name :str ,
    pipeline_config :Optional [Dict ],class_name :str )->Dict [str ,str ]:
        print (f"   📋 Создание аннотаций для класса '{class_name}'...")

        saved_files ={}

        annotations =self ._build_annotations (image ,class_masks ,image_name ,pipeline_config )
        annotations ['class_name']=class_name

        annotations_path =class_dir /f"annotations_{class_name}.json"
        save_json (annotations ,str (annotations_path ))
        saved_files [f'annotations_{class_name}']=str (annotations_path )

        print (f"     ✅ Аннотации для класса '{class_name}': {len(class_masks)} объектов")
        return saved_files

    def _save_empty_results (self ,image :np .ndarray ,result_dir :Path ,
    image_name :str ,pipeline_config :Optional [Dict ])->Dict [str ,str ]:
        saved_files ={}

        original_path =result_dir /"original.png"
        image_bgr =cv2 .cvtColor (image ,cv2 .COLOR_RGB2BGR )
        cv2 .imwrite (str (original_path ),image_bgr )
        saved_files ['original']=str (original_path )

        empty_annotations =self ._build_annotations (
        image ,[],image_name ,pipeline_config
        )
        annotations_path =result_dir /"annotations.json"
        save_json (empty_annotations ,str (annotations_path ))
        saved_files ['annotations']=str (annotations_path )

        return saved_files

    def _save_visualizations (self ,image :np .ndarray ,masks :List [Dict [str ,Any ]],
    result_dir :Path )->Dict [str ,str ]:
        print ("   🎨 Создание визуализаций...")

        saved_files ={}

        original_path =result_dir /"original.png"
        image_bgr =cv2 .cvtColor (image ,cv2 .COLOR_RGB2BGR )
        cv2 .imwrite (str (original_path ),image_bgr )
        saved_files ['original']=str (original_path )

        overlay_path =result_dir /"overlay_masks.png"
        overlay_img =self ._create_overlay_visualization (image ,masks )
        cv2 .imwrite (str (overlay_path ),overlay_img )
        saved_files ['overlay']=str (overlay_path )

        contours_path =result_dir /"contours_only.png"
        contours_img =self ._create_contours_visualization (image ,masks )
        cv2 .imwrite (str (contours_path ),contours_img )
        saved_files ['contours']=str (contours_path )

        semantic_path =result_dir /"semantic_mask.png"
        semantic_img =self ._create_semantic_visualization (image .shape ,masks )
        cv2 .imwrite (str (semantic_path ),semantic_img )
        saved_files ['semantic']=str (semantic_path )

        print (f"     ✅ Создано {len(saved_files)} визуализаций")
        return saved_files

    def _create_overlay_visualization (self ,image :np .ndarray ,
    masks :List [Dict [str ,Any ]])->np .ndarray :

        result =cv2 .cvtColor (image ,cv2 .COLOR_RGB2BGR ).copy ()
        overlay =result .copy ()

        class_colors ={}
        unique_classes =list (set (mask .get ('class','unknown')for mask in masks ))
        for i ,cls in enumerate (unique_classes ):
            class_colors [cls ]=self .colors [i %len (self .colors )]

        for mask in masks :
            cls =mask .get ('class','unknown')
            color =class_colors [cls ]
            segmentation =mask ['segmentation']

            seg_bool =segmentation .astype (bool )if segmentation .dtype !=bool else segmentation
            if seg_bool.shape != (overlay.shape[0], overlay.shape[1]):
                print(f"     ⚠️ Несовпадение размеров в overlay: маска {seg_bool.shape} vs изображение {(overlay.shape[0], overlay.shape[1])}")
                seg_bool_resized = cv2.resize(seg_bool.astype(np.uint8), 
                                            (overlay.shape[1], overlay.shape[0]), 
                                            interpolation=cv2.INTER_NEAREST).astype(bool)
                overlay[seg_bool_resized] = color
            else:
                overlay [seg_bool ]=color

        cv2 .addWeighted (overlay ,self .overlay_alpha ,result ,1 -self .overlay_alpha ,0 ,result )

        return result

    def _create_overlay_visualization_fast (self ,image :np .ndarray ,
    masks :List [Dict [str ,Any ]])->np .ndarray :
        result =cv2 .cvtColor (image ,cv2 .COLOR_RGB2BGR ).copy ()

        basic_colors =[(0 ,255 ,0 ),(255 ,0 ,0 ),(0 ,0 ,255 ),(255 ,255 ,0 ),(255 ,0 ,255 )]

        for i ,mask in enumerate (masks [:10 ]):
            color =basic_colors [i %len (basic_colors )]
            segmentation =mask ['segmentation']

            seg_bool =segmentation .astype (bool )if segmentation .dtype !=bool else segmentation
            
            # Проверяем совместимость размеров для boolean индексации
            if seg_bool.shape != (result.shape[0], result.shape[1]):
                print(f"     ⚠️ Несовпадение размеров в fast overlay: маска {seg_bool.shape} vs изображение {(result.shape[0], result.shape[1])}")
                # Изменяем размер маски под размер изображения
                seg_bool_resized = cv2.resize(seg_bool.astype(np.uint8), 
                                            (result.shape[1], result.shape[0]), 
                                            interpolation=cv2.INTER_NEAREST).astype(bool)
                result[seg_bool_resized] = color
            else:
                result [seg_bool ]=color

        return result

    def _create_overlay_with_bins (self ,image :np .ndarray ,bin1_path :Path ,bin2_path :Path =None )->np .ndarray :

        try :

            result =cv2 .cvtColor (image ,cv2 .COLOR_RGB2BGR ).copy ()

            bin1_img =cv2 .imread (str (bin1_path ))

            if bin1_img is None :
                print (f"     ⚠️ Не удалось загрузить файл bin1")
                return result

            if bin1_img .shape !=result .shape :
                bin1_img =cv2 .resize (bin1_img ,(result .shape [1 ],result .shape [0 ]))

            bin1_mask =np .any (bin1_img >10 ,axis =2 )

            alpha_bin1 =0.7

            # Отображаем только bin1
            if np .any (bin1_mask ):

                bin1_red =np .zeros_like (bin1_img )
                bin1_red [:,:,2 ]=255
                bin1_red [~bin1_mask ]=0

                result [bin1_mask ]=cv2 .addWeighted (
                result [bin1_mask ],1 -alpha_bin1 ,
                bin1_red [bin1_mask ],alpha_bin1 ,0
                )

            # Добавляем контуры только для bin1
            self ._add_bin_contours (result ,bin1_mask ,np .zeros_like (bin1_mask ))

            print (f"     📊 Overlay статистика: Бин1={np.sum(bin1_mask)} пикс (только bin1)")
            print (f"     🎨 Цвета: Бин1=красный (alpha={alpha_bin1})")
            return result

        except Exception as e :
            print (f"     ❌ Ошибка при создании overlay с бинами: {e}")

            return cv2 .cvtColor (image ,cv2 .COLOR_RGB2BGR ).copy ()

    def _add_bin_contours (self ,image :np .ndarray ,bin1_mask :np .ndarray ,bin2_mask :np .ndarray =None )->None :

        try :

            # Отображаем контуры только для bin1
            if np .any (bin1_mask ):

                bin1_uint8 =bin1_mask .astype (np .uint8 )*255
                contours1 ,_ =cv2 .findContours (bin1_uint8 ,cv2 .RETR_EXTERNAL ,cv2 .CHAIN_APPROX_SIMPLE )

                cv2 .drawContours (image ,contours1 ,-1 ,(0 ,255 ,255 ),4 )
                print (f"     🔴 Добавлено {len(contours1)} контуров для Бин1")

        except Exception as e :
            print (f"     ⚠️ Ошибка при добавлении контуров: {e}")

    def _create_contours_visualization (self ,image :np .ndarray ,
    masks :List [Dict [str ,Any ]])->np .ndarray :
        result =cv2 .cvtColor (image ,cv2 .COLOR_RGB2BGR ).copy ()
        overlay =result .copy ()

        class_colors ={}
        unique_classes =list (set (mask .get ('class','unknown')for mask in masks ))
        for i ,cls in enumerate (unique_classes ):
            class_colors [cls ]=self .colors [i %len (self .colors )]

        for i ,mask in enumerate (masks ):

            cls =mask .get ('class','unknown')
            color =class_colors [cls ]
            segmentation =mask ['segmentation']

            seg_bool =segmentation .astype (bool )if segmentation .dtype !=bool else segmentation
            
            # Проверяем совместимость размеров для contours overlay
            if seg_bool.shape != (overlay.shape[0], overlay.shape[1]):
                print(f"     ⚠️ Несовпадение размеров в contours overlay: маска {seg_bool.shape} vs изображение {(overlay.shape[0], overlay.shape[1])}")
                seg_bool = cv2.resize(seg_bool.astype(np.uint8), 
                                    (overlay.shape[1], overlay.shape[0]), 
                                    interpolation=cv2.INTER_NEAREST).astype(bool)
            
            overlay [seg_bool ]=color

            result_contours =cv2 .findContours (
            segmentation .astype (np .uint8 ),
            cv2 .RETR_EXTERNAL ,
            cv2 .CHAIN_APPROX_SIMPLE
            )
            if len (result_contours )==3 :

                _ ,contours ,_ =result_contours
            else :

                contours ,_ =result_contours

            self ._safe_draw_contours (result ,contours ,-1 ,color ,3 )

            valid_contours =[c for c in contours if len (c )>0 ]if contours else []
            if valid_contours :

                M =cv2 .moments (valid_contours [0 ])
                if M ["m00"]!=0 :
                    cx =int (M ["m10"]/M ["m00"])
                    cy =int (M ["m01"]/M ["m00"])
                    cv2 .putText (result ,cls ,(cx -20 ,cy ),cv2 .FONT_HERSHEY_SIMPLEX ,
                    0.7 ,(255 ,255 ,255 ),2 )

        cv2 .addWeighted (overlay ,0.3 ,result ,0.7 ,0 ,result )

        return result

    def _create_contours_only_visualization(self, image: np.ndarray, 
                                          masks: List[Dict[str, Any]]) -> np.ndarray:
        """Создает изображение только с контурами масок (без заливки) с толстыми линиями"""
        # Создаем черное изображение
        result = np.zeros_like(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        class_colors = {}
        unique_classes = list(set(mask.get('class', 'unknown') for mask in masks))
        for i, cls in enumerate(unique_classes):
            class_colors[cls] = self.colors[i % len(self.colors)]
        
        for mask in masks:
            cls = mask.get('class', 'unknown')
            color = class_colors[cls]
            segmentation = mask['segmentation']
            
            # Находим контуры
            result_contours = cv2.findContours(
                segmentation.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            if len(result_contours) == 3:
                _, contours, _ = result_contours
            else:
                contours, _ = result_contours
            
            # Рисуем толстые контуры (толщина 6 вместо 3)
            self._safe_draw_contours(result, contours, -1, color, 6)
        
        return result

    def _create_semantic_visualization (self ,image_shape :tuple ,
    masks :List [Dict [str ,Any ]])->np .ndarray :
        h ,w =image_shape [:2 ]
        semantic_mask =np .zeros ((h ,w ,3 ),dtype =np .uint8 )

        class_colors ={}
        unique_classes =list (set (mask .get ('class','unknown')for mask in masks ))
        for i ,cls in enumerate (unique_classes ):
            class_colors [cls ]=self .colors [i %len (self .colors )]

        for i ,mask in enumerate (masks ):
            cls =mask .get ('class','unknown')
            color =class_colors [cls ]
            segmentation =mask ['segmentation']

            seg_bool =segmentation .astype (bool )if segmentation .dtype !=bool else segmentation
            
            if seg_bool.shape != (semantic_mask.shape[0], semantic_mask.shape[1]):
                print(f"     ⚠️ Несовпадение размеров в semantic: маска {seg_bool.shape} vs изображение {(semantic_mask.shape[0], semantic_mask.shape[1])}")
                seg_bool_resized = cv2.resize(seg_bool.astype(np.uint8), 
                                            (semantic_mask.shape[1], semantic_mask.shape[0]), 
                                            interpolation=cv2.INTER_NEAREST).astype(bool)
                semantic_mask[seg_bool_resized] = color
            else:
                semantic_mask [seg_bool ]=color

        return semantic_mask

    def _save_individual_masks (self ,masks :List [Dict [str ,Any ]],
    result_dir :Path )->Dict [str ,str ]:
        print ("   💾 Сохранение отдельных масок (оптимизированно)...")

        saved_files ={}

        for i ,mask in enumerate (masks ):
            segmentation =mask ['segmentation']
            confidence =mask .get ('confidence',1.0 )

            if segmentation is None or not hasattr (segmentation ,'shape'):
                print (f"     ⚠️ Пропускаем маску {i}: пустая segmentation")
                continue

            try :
                mask_img =(segmentation *255 ).astype (np .uint8 )
            except Exception as e :
                print (f"     ⚠️ Ошибка конвертации маски {i}: {e}")
                continue

            mask_filename =f"mask_{i:03d}_conf_{confidence:.3f}.png"
            mask_path =result_dir /mask_filename

            try :
                cv2 .imwrite (str (mask_path ),mask_img )
                saved_files [f'mask_{i}']=str (mask_path )
            except Exception as e :
                print (f"     ⚠️ Ошибка сохранения маски {i}: {e}")
                continue

        print(f"     ✅ Сохранено {len(saved_files)} отдельных масок")
        return saved_files

    def _save_individual_fastsam_masks(self, masks: List[Dict[str, Any]], 
                                     result_dir: Path, image_name: str) -> Dict[str, str]:
        """Сохраняет каждую FastSAM маску в отдельный файл с именем result_{image_name}SAM_{i}.png"""
        print("   💾 Сохранение отдельных FastSAM масок...")
        
        saved_files = {}
        base_name = Path(image_name).stem
        
        for i, mask in enumerate(masks):
            segmentation = mask['segmentation']
            confidence = mask.get('confidence', 1.0)
            class_name = mask.get('class', 'unknown')
            
            if segmentation is None or not hasattr(segmentation, 'shape'):
                print(f"     ⚠️ Пропускаем FastSAM маску {i}: пустая segmentation")
                continue
                
            try:
                # Создаем изображение маски
                mask_img = (segmentation * 255).astype(np.uint8)
                
                # Формируем имя файла: result_{image_name}SAM_{i}.png
                mask_filename = f"result_{base_name}SAM_{i:03d}_class_{class_name}_conf_{confidence:.3f}.png"
                mask_path = result_dir / mask_filename
                
                # Сохраняем маску
                cv2.imwrite(str(mask_path), mask_img)
                saved_files[f'fastsam_mask_{i}'] = str(mask_path)
                
                print(f"     ✅ Сохранена FastSAM маска {i}: {mask_filename}")
                
            except Exception as e:
                print(f"     ⚠️ Ошибка сохранения FastSAM маски {i}: {e}")
                continue
                
        print(f"     ✅ Сохранено {len(saved_files)} отдельных FastSAM масок")
        return saved_files

    def _save_individual_fastsam_overlays(self, image: np.ndarray, masks: List[Dict[str, Any]], 
                                        result_dir: Path, image_name: str) -> Dict[str, str]:
        """Сохраняет каждую FastSAM маску как overlay с оригинальным изображением"""
        print("   🎨 Сохранение отдельных FastSAM overlay масок...")
        
        saved_files = {}
        base_name = Path(image_name).stem
        
        for i, mask in enumerate(masks):
            segmentation = mask['segmentation']
            confidence = mask.get('confidence', 1.0)
            class_name = mask.get('class', 'unknown')
            
            if segmentation is None or not hasattr(segmentation, 'shape'):
                print(f"     ⚠️ Пропускаем FastSAM overlay маску {i}: пустая segmentation")
                continue
                
            try:
                # Создаем overlay изображение для одной маски
                overlay_img = image.copy()
                
                # Применяем цветную маску
                color = self.colors[i % len(self.colors)]
                mask_colored = np.zeros_like(image)
                
                seg_bool = segmentation.astype(bool)
                # Проверяем совместимость размеров для overlay маски
                if seg_bool.shape != (mask_colored.shape[0], mask_colored.shape[1]):
                    print(f"     ⚠️ Несовпадение размеров в fastsam overlay {i}: маска {seg_bool.shape} vs изображение {(mask_colored.shape[0], mask_colored.shape[1])}")
                    seg_bool = cv2.resize(seg_bool.astype(np.uint8), 
                                        (mask_colored.shape[1], mask_colored.shape[0]), 
                                        interpolation=cv2.INTER_NEAREST).astype(bool)
                
                mask_colored[seg_bool] = color
                
                # Смешиваем с оригинальным изображением
                overlay_img = cv2.addWeighted(overlay_img, 1 - self.overlay_alpha, 
                                            mask_colored, self.overlay_alpha, 0)
                
                # Формируем имя файла: result_{image_name}SAM_overlay_{i}.png
                overlay_filename = f"result_{base_name}SAM_overlay_{i:03d}_class_{class_name}_conf_{confidence:.3f}.png"
                overlay_path = result_dir / overlay_filename
                
                # Сохраняем overlay
                cv2.imwrite(str(overlay_path), overlay_img)
                saved_files[f'fastsam_overlay_{i}'] = str(overlay_path)
                
                print(f"     ✅ Сохранена FastSAM overlay маска {i}: {overlay_filename}")
                
            except Exception as e:
                print(f"     ⚠️ Ошибка сохранения FastSAM overlay маски {i}: {e}")
                continue
                
        print(f"     ✅ Сохранено {len(saved_files)} отдельных FastSAM overlay масок")
        return saved_files

    def _save_total_mask (self ,masks :List [Dict [str ,Any ]],image_shape :tuple ,
    result_dir :Path )->Dict [str ,str ]:
        print ("   🔗 Создание общей маски...")

        h ,w =image_shape [:2 ]
        total_mask =np .zeros ((h ,w ),dtype =bool )

        for mask in masks :
            total_mask =np .logical_or (total_mask ,mask ['segmentation'])

        total_mask_img =(total_mask *255 ).astype (np .uint8 )
        total_mask_path =result_dir /"total_mask.png"
        cv2 .imwrite (str (total_mask_path ),total_mask_img )

        print (f"     ✅ Общая маска покрывает {np.sum(total_mask)} пикселей")
        return {'total_mask':str (total_mask_path )}

    def _save_annotations (self ,image :np .ndarray ,masks :List [Dict [str ,Any ]],
    result_dir :Path ,image_name :str ,
    pipeline_config :Optional [Dict ])->Dict [str ,str ]:

        print ("   📋 Создание JSON аннотаций...")

        annotations =self ._build_annotations (image ,masks ,image_name ,pipeline_config )

        annotations_path =result_dir /"annotations.json"
        save_json (annotations ,str (annotations_path ))

        print (f"     ✅ Аннотации сохранены: {len(masks)} объектов")
        return {'annotations':str (annotations_path )}

    def _build_annotations (self ,image :np .ndarray ,masks :List [Dict [str ,Any ]],
    image_name :str ,pipeline_config :Optional [Dict ])->Dict [str ,Any ]:

        h ,w =image .shape [:2 ]

        annotations ={
        "metadata":{
        "timestamp":datetime .now ().isoformat (),
        "image_name":image_name ,
        "image_width":w ,
        "image_height":h ,
        "total_pixels":h *w ,
        "total_detections":len (masks ),
        "pipeline_version":"2.0.0",
        },
        "pipeline_config":pipeline_config or {},
        "detections":[]
        }

        if masks :
            confidences =[mask ['confidence']for mask in masks ]
            areas =[mask ['area']for mask in masks ]

            annotations ["statistics"]={
            "mean_confidence":float (np .mean (confidences )),
            "min_confidence":float (np .min (confidences )),
            "max_confidence":float (np .max (confidences )),
            "std_confidence":float (np .std (confidences )),
            "total_detected_area":int (sum (areas )),
            "coverage_fraction":float (sum (areas )/(h *w )),
            "mean_detection_area":float (np .mean (areas )),
            "min_detection_area":int (np .min (areas )),
            "max_detection_area":int (np .max (areas )),
            }
        else :
            annotations ["statistics"]={
            "mean_confidence":0 ,
            "min_confidence":0 ,
            "max_confidence":0 ,
            "std_confidence":0 ,
            "total_detected_area":0 ,
            "coverage_fraction":0 ,
            "mean_detection_area":0 ,
            "min_detection_area":0 ,
            "max_detection_area":0 ,
            }

        for i ,mask in enumerate (masks ):
            segmentation =mask ['segmentation']
            bbox =mask ['bbox']

            result_contours =cv2 .findContours (
            segmentation .astype (np .uint8 ),
            cv2 .RETR_EXTERNAL ,
            cv2 .CHAIN_APPROX_SIMPLE
            )
            if len (result_contours )==3 :
                _ ,contours ,_ =result_contours
            else :
                contours ,_ =result_contours

            if contours is not None and len (contours )>0 :
                largest_contour =max (contours ,key =cv2 .contourArea )
                polygon =largest_contour .reshape (-1 ,2 ).tolist ()
            else :
                polygon =[]

            detection ={
            "id":i ,
            "bbox":bbox ,
            "area":int (mask ['area']),
            "area_fraction":float (mask ['area']/(h *w )),
            "confidence":float (mask ['confidence']),
            "class":mask .get ("class"),
            "segmentation":polygon ,
            "center":[
            int (bbox [0 ]+bbox [2 ]/2 ),
            int (bbox [1 ]+bbox [3 ]/2 )
            ],
            "original_mask_index":mask .get ('original_index',i ),
            }

            for field in ['stability_score','predicted_iou']:
                if field in mask :
                    detection [field ]=float (mask [field ])

            annotations ["detections"].append (detection )

        return annotations

    def _generate_colors (self ,num_colors :int =20 )->List [tuple ]:
        colors =[]

        hue_step =180 /num_colors
        for i in range (num_colors ):
            hue =int (i *hue_step )
            saturation =255
            value =220

            hsv_color =np .uint8 ([[[hue ,saturation ,value ]]])
            bgr_color =cv2 .cvtColor (hsv_color ,cv2 .COLOR_HSV2BGR )[0 ][0 ]
            colors .append (tuple (map (int ,bgr_color )))

        return colors

    def create_summary_report (self ,saved_files :Dict [str ,str ],
    processing_time :float ,
    final_masks :List [Dict [str ,Any ]])->str :

        summary_lines =[
        "🎯 СВОДКА РЕЗУЛЬТАТОВ ДЕТЕКЦИИ",
        "="*50 ,
        f"⏱️  Время обработки: {processing_time:.2f} сек",
        f"🔍 Найдено объектов: {len(final_masks)}",]

        if final_masks :
            confidences =[mask ['confidence']for mask in final_masks ]
            areas =[mask ['area']for mask in final_masks ]

            summary_lines .extend ([
            f"📈 Confidence: {np.mean(confidences):.3f} ± {np.std(confidences):.3f}",
            f"📐 Средняя площадь: {np.mean(areas):.0f} пикселей",
            f"📊 Диапазон confidence: {np.min(confidences):.3f} - {np.max(confidences):.3f}",
            ])

        summary_lines .extend ([
        "",
        "📁 Сохранённые файлы:",
        ])

        for file_type ,file_path in saved_files .items ():
            summary_lines .append (f"   • {file_type}: {Path(file_path).name}")

        return "\n".join (summary_lines )

    def _apply_pixel_binning_to_heatmap (self ,heatmap :np .ndarray ,image :np .ndarray ,
    positive_images :List [Image .Image ]=None ,
    negative_images :List [Image .Image ]=None )->np .ndarray :
        try :
            print ("   🔥 Применение бинаризации к хитмапу на основе косинусных расстояний...")
            valid_mask =~np .isnan (heatmap )&(heatmap >0 )
            if not np .any (valid_mask ):
                print ("   ⚠️ Нет валидных значений в хитмапе")
                return heatmap

            valid_values =heatmap [valid_mask ]

            print (f"   📊 Статистика хитмапа: min={valid_values.min():.4f}, max={valid_values.max():.4f}, mean={valid_values.mean():.4f}")

            num_bins =getattr (self .binning_processor ,'num_bins',7 )if self .binning_processor else 7
            bin_edges =np .quantile (valid_values ,np .linspace (0 ,1 ,num_bins +1 ))
            bin_edges [0 ]=float (valid_values .min ())-1e-6
            bin_edges [-1 ]=float (valid_values .max ())+1e-6

            print (f"   📦 Создано {num_bins} бинов")
            print (f"   🥇 Первый бин (топ-пиксели): [{bin_edges[-2]:.4f}, {bin_edges[-1]:.4f})")
            enhanced_heatmap =np .zeros_like (heatmap )
            first_bin_mask =(heatmap >=bin_edges [-2 ])&(heatmap <bin_edges [-1 ])
            enhanced_heatmap [first_bin_mask ]=heatmap [first_bin_mask ]
            if num_bins >2 :
                second_bin_mask =(heatmap >=bin_edges [-3 ])&(heatmap <bin_edges [-2 ])
                enhanced_heatmap [second_bin_mask ]=heatmap [second_bin_mask ]*0.7
            if num_bins >3 :
                third_bin_mask =(heatmap >=bin_edges [-4 ])&(heatmap <bin_edges [-3 ])
                enhanced_heatmap [third_bin_mask ]=heatmap [third_bin_mask ]*0.4

            pixels_in_first_bin =int (np .sum (first_bin_mask ))
            pixels_in_second_bin =int (np .sum (second_bin_mask ))if num_bins >2 else 0
            pixels_in_third_bin =int (np .sum (third_bin_mask ))if num_bins >3 else 0
            total_valid_pixels =int (np .sum (valid_mask ))
            total_enhanced_pixels =pixels_in_first_bin +pixels_in_second_bin +pixels_in_third_bin

            print (f"   ✅ Бинаризация завершена: {total_enhanced_pixels}/{total_valid_pixels} пикселей ({100*total_enhanced_pixels/max(total_valid_pixels,1):.1f}%)")
            print (f"     🥇 Бин 1: {pixels_in_first_bin} пикселей (100% интенсивность)")
            if pixels_in_second_bin >0 :
                print (f"     🥈 Бин 2: {pixels_in_second_bin} пикселей (70% интенсивность)")
            if pixels_in_third_bin >0 :
                print (f"     🥉 Бин 3: {pixels_in_third_bin} пикселей (40% интенсивность)")

            return enhanced_heatmap

        except Exception as e :
            print (f"   ❌ Ошибка при бинаризации хитмапа: {e}")
            return heatmap

    def _save_heatmap_minimal (self ,heatmap :np .ndarray ,result_dir :Path )->Dict [str ,str ]:
        saved ={}
        try :
            print ("   💾 Минимальное сохранение heatmap (только raw)...")
            if 'torch'in str (type (heatmap )):
                try :
                    heat =heatmap .detach ().cpu ().numpy ()
                except Exception :
                    heat =np .array (heatmap )
            else :
                heat =np .array (heatmap )
            heat =np .squeeze (heat )

            raw_path =result_dir /"heatmap_raw.npy"
            np .save (str (raw_path ),heat )
            saved ['heatmap_raw']=str (raw_path )

        except Exception as e :
            print (f"   ❌ Ошибка при сохранении heatmap: {e}")
        return saved

    def _save_annotations_only (self ,image :np .ndarray ,masks :List [Dict [str ,Any ]],
    image_name :str ,pipeline_config :Optional [Dict ],
    result_dir :Path )->Dict [str ,str ]:
        print ("   📋 Сохранение только JSON аннотаций...")

        annotations =self ._build_annotations (image ,masks ,image_name ,pipeline_config )

        annotations_path =result_dir /"annotations.json"
        save_json (annotations ,str (annotations_path ))

        print (f"     ✅ Аннотации сохранены: {len(masks)} объектов")
        return {'annotations':str (annotations_path )}