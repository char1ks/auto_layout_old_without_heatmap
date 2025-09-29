
import numpy as np
from typing import List ,Dict ,Any ,Tuple

# TODO: (@gas) review and test

class MaskFilter :
    def __init__ (self ,params :Dict [str ,Any ]=None )->None :
        if params is None :
            params ={}

        self .min_area_frac =params .get ('min_area_frac',0.002 )
        self .max_area_frac =params .get ('max_area_frac',0.95 )
        self .perfect_rectangle_iou =params .get ('perfect_rectangle_iou',0.95 )
        self .containment_iou =params .get ('containment_iou',0.70 )
        self .border_ban =params .get ('border_ban',False )
        self .border_width =params .get ('border_width',2 )

        self .enable_mask_correction =params .get ('enable_mask_correction',True )
        self .erosion_iterations =params .get ('erosion_iterations',1 )
        self .dilation_iterations =params .get ('dilation_iterations',2 )
        self .correction_kernel_size =params .get ('correction_kernel_size',3 )

        self .smart_rectangle_filter =params .get ('smart_rectangle_filter',True )
        self .rectangle_bbox_iou_threshold =params .get ('rectangle_bbox_iou_threshold',0.85 )
        self .rectangle_straight_line_ratio =params .get ('rectangle_straight_line_ratio',0.7 )
        self .rectangle_area_ratio_threshold =params .get ('rectangle_area_ratio_threshold',0.9 )
        self .rectangle_angle_tolerance =params .get ('rectangle_angle_tolerance',15.0 )
        self .rectangle_side_ratio_threshold =params .get ('rectangle_side_ratio_threshold',0.8 )
        self .rectangle_similarity_iou_threshold =params .get ('rectangle_similarity_iou_threshold',0.92 )
        self .square_similarity_iou_threshold =params .get ('square_similarity_iou_threshold',0.92 )
        self .rectangle_use_silhouette =params .get ('rectangle_use_silhouette',True )
        self .hole_area_ratio_threshold =params .get ('hole_area_ratio_threshold',0.02 )

        self ._bbox_cache ={}
        self ._area_cache ={}

    def apply_all_filters (self ,masks :List [Dict [str ,Any ]],image_np :np .ndarray )->List [Dict [str ,Any ]]:

        if not masks :
            return masks

        masks ,_ ,big_count =self ._filter_by_size (masks ,image_np .shape )
        if big_count >0 :
            print (f"📏 Удалено {big_count} слишком больших масок")

        if self .smart_rectangle_filter :
            initial_count =len (masks )
            masks =self ._filter_perfect_rectangles_optimized (masks )
            dropped =initial_count -len (masks )
            if dropped >0 :
                print (f"🔳 Удалено {dropped} прямоугольных масок")

        initial_count =len (masks )
        masks =self ._filter_border_masks (masks ,image_np )
        dropped =initial_count -len (masks )
        if dropped >0 :
            print (f"🖼️ Удалено {dropped} граничных масок")

        initial_count =len (masks )
        masks =self ._filter_nested_masks_optimized (masks )
        dropped =initial_count -len (masks )
        if dropped >0 :
            print (f"🔗 Удалено {dropped} вложенных масок")

        if self .enable_mask_correction :
            masks =self ._apply_mask_correction_fast (masks )

        return masks

    def _filter_by_size (self ,masks :List [Dict [str ,Any ]],image_shape :tuple )->Tuple [List [Dict [str ,Any ]],int ,int ]:
        h ,w =image_shape [:2 ]
        total_pixels =h *w

        max_area_abs =self .max_area_frac *total_pixels

        filtered_masks =[]
        small_count =0
        big_count =0

        for mask_dict in masks :
            area =mask_dict .get ('area',0 )
            if area >max_area_abs :
                big_count +=1
            else :
                filtered_masks .append (mask_dict )

        return filtered_masks ,small_count ,big_count

    def _filter_perfect_rectangles_optimized (self ,masks :List [Dict [str ,Any ]])->List [Dict [str ,Any ]]:
        filtered_masks =[]
        threshold =self .perfect_rectangle_iou
        eps =1e-8

        for mask_dict in masks :

            area =float (mask_dict .get ("area",0.0 ))
            x ,y ,w ,h =mask_dict .get ("bbox",(0 ,0 ,0 ,0 ))
            w_i =max (0 ,int (round (w )))
            h_i =max (0 ,int (round (h )))
            bbox_area =float (max (1 ,w_i *h_i ))
            bbox_iou =area /(bbox_area +eps )

            if bbox_iou <threshold :
                filtered_masks .append (mask_dict )
            elif bbox_iou >0.95 :

                continue
            else :

                filtered_masks .append (mask_dict )

        return filtered_masks

    def _is_rectangular_shape_fast (self ,mask :np .ndarray ,cv2 )->bool :

        bbox_iou =self ._calculate_bbox_iou_fast (mask )

        return bbox_iou >0.92

    def _check_rectangle_properties (self ,approx :np .ndarray ,mask :np .ndarray ,cv2 )->bool :

        if len (approx )!=4 :
            return False

        angles =[]
        points =approx .reshape (-1 ,2 )

        for i in range (4 ):
            p1 =points [i ]
            p2 =points [(i +1 )%4 ]
            p3 =points [(i +2 )%4 ]

            v1 =p1 -p2
            v2 =p3 -p2

            cos_angle =np .dot (v1 ,v2 )/(np .linalg .norm (v1 )*np .linalg .norm (v2 )+1e-8 )
            angle =np .arccos (np .clip (cos_angle ,-1 ,1 ))
            angles .append (np .degrees (angle ))

        right_angles =sum (1 for angle in angles if abs (angle -90 )<self .rectangle_angle_tolerance )

        side_lengths =[]
        for i in range (4 ):
            p1 =points [i ]
            p2 =points [(i +1 )%4 ]
            length =np .linalg .norm (p2 -p1 )
            side_lengths .append (length )

        side_ratio1 =min (side_lengths [0 ],side_lengths [2 ])/(max (side_lengths [0 ],side_lengths [2 ])+1e-8 )
        side_ratio2 =min (side_lengths [1 ],side_lengths [3 ])/(max (side_lengths [1 ],side_lengths [3 ])+1e-8 )

        return right_angles >=3 and side_ratio1 >self .rectangle_side_ratio_threshold and side_ratio2 >self .rectangle_side_ratio_threshold

    def _get_mask_hash (self ,mask :np .ndarray )->str :

        return str (hash (mask .tobytes ()))

    def _calculate_bbox_iou_fast (self ,mask :np .ndarray )->float :

        mask_hash =self ._get_mask_hash (mask )

        if mask_hash in self ._bbox_cache :
            return self ._bbox_cache [mask_hash ]

        ys ,xs =np .where (mask )
        if len (ys )==0 :
            result =0.0
        else :
            x1 ,y1 ,x2 ,y2 =xs .min (),ys .min (),xs .max (),ys .max ()

            mask_area =mask .sum ()

            bbox_area =(x2 -x1 +1 )*(y2 -y1 +1 )

            result =float (mask_area )/float (bbox_area )

        self ._bbox_cache [mask_hash ]=result
        return result

    def _calculate_area_ratio (self ,mask :np .ndarray ,contour :np .ndarray ,cv2 )->float :

        mask_area =mask .sum ()
        contour_area =cv2 .contourArea (contour )

        if contour_area ==0 :
            return 0.0

        return mask_area /contour_area

    def _overlay_similarity_scores_fast (self ,mask :np .ndarray ,cv2 )->Dict [str ,float ]:

        H ,W =mask .shape [:2 ]

        ys ,xs =np .where (mask )
        if ys .size ==0 :
            return {'axis_rect_iou':0.0 ,'rot_rect_iou':0.0 ,'square_iou':0.0 ,'hole_ratio':0.0 }

        x1 ,y1 ,x2 ,y2 =xs .min (),ys .min (),xs .max (),ys .max ()
        w =x2 -x1 +1
        h =y2 -y1 +1
        cx =(x1 +x2 )/2.0
        cy =(y1 +y2 )/2.0

        axis_rect =np .zeros ((H ,W ),dtype =bool )
        axis_rect [y1 :y2 +1 ,x1 :x2 +1 ]=True

        rot_rect_iou =self ._calculate_bbox_iou_fast (mask )

        side_min =int (min (w ,h ))
        side_max =int (max (w ,h ))

        def _square_mask (side :int )->np .ndarray :
            half =side //2
            sx1 =int (max (0 ,int (round (cx ))-half ))
            sy1 =int (max (0 ,int (round (cy ))-half ))
            sx2 =int (min (W ,sx1 +side ))
            sy2 =int (min (H ,sy1 +side ))
            sq =np .zeros ((H ,W ),dtype =bool )
            if sx2 >sx1 and sy2 >sy1 :
                sq [sy1 :sy2 ,sx1 :sx2 ]=True
            return sq

        square_min =_square_mask (side_min )
        square_max =_square_mask (side_max )

        def _iou_fast (a :np .ndarray ,b :np .ndarray )->float :
            inter =np .logical_and (a ,b ).sum ()
            uni =np .logical_or (a ,b ).sum ()
            return float (inter )/(float (uni )+1e-8 )

        axis_rect_iou =_iou_fast (mask ,axis_rect )
        square_iou =max (_iou_fast (mask ,square_min ),_iou_fast (mask ,square_max ))

        bbox_area =w *h
        mask_area =mask .sum ()
        hole_ratio =max (0.0 ,float (bbox_area -mask_area )/float (bbox_area +1e-8 ))

        return {
        'axis_rect_iou':axis_rect_iou ,
        'rot_rect_iou':rot_rect_iou ,
        'square_iou':square_iou ,
        'hole_ratio':hole_ratio ,
        }

    def _filter_rectangles_simple (self ,masks :List [Dict [str ,Any ]])->List [Dict [str ,Any ]]:

        filtered_masks =[]
        threshold =self .perfect_rectangle_iou

        for mask_dict in masks :
            mask =mask_dict ["segmentation"]
            bbox =mask_dict ["bbox"]

            x ,y ,w ,h =bbox
            H ,W =mask .shape [:2 ]
            x =int (round (float (x )))
            y =int (round (float (y )))
            w =int (round (float (w )))
            h =int (round (float (h )))
            if w <0 :w =0
            if h <0 :h =0
            x0 =max (0 ,min (x ,W ))
            y0 =max (0 ,min (y ,H ))
            x1 =max (x0 ,min (x0 +w ,W ))
            y1 =max (y0 ,min (y0 +h ,H ))
            rect_mask =np .zeros_like (mask ,dtype =bool )
            if (x1 -x0 )>0 and (y1 -y0 )>0 :
                rect_mask [y0 :y1 ,x0 :x1 ]=True

            intersection =np .logical_and (mask ,rect_mask ).sum ()
            union =np .logical_or (mask ,rect_mask ).sum ()
            iou =intersection /union if union >0 else 0

            if iou <threshold :
                filtered_masks .append (mask_dict )

        return filtered_masks

    def _filter_border_masks (self ,masks ,image_np ):
        h ,w =image_np .shape [:2 ]
        border_width =self .border_width
        ban_border =self .border_ban

        if not ban_border or border_width <=0 :
            print (f"🖼️ Обработка границ отключена (ban_border={ban_border}, border_width={border_width}): {len(masks)} → {len(masks)}")
            return masks

        filtered_masks =[]
        dropped =0
        clipped =0

        for mask_dict in masks :
            mask =mask_dict ["segmentation"]

            touches_border =(
            mask [:border_width ,:].any ()or
            mask [-border_width :,:].any ()or
            mask [:,:border_width ].any ()or
            mask [:,-border_width :].any ()
            )

            if touches_border :
                dropped +=1
            else :
                filtered_masks .append (mask_dict )

        print (f"🖼️ Обработка границ (запрет, {border_width}px): {len(masks)} → {len(filtered_masks)} (dropped={dropped}, clipped={clipped})")

        return filtered_masks

    def _filter_nested_masks_optimized (self ,masks ):

        if len (masks )<=1 :
            return masks

        filtered_masks =sorted (masks ,key =lambda m :m ['area'],reverse =True )
        to_remove =set ()

        for i in range (len (filtered_masks )):
            if i in to_remove :
                continue

            mask_i =filtered_masks [i ]
            bbox_i =mask_i ['bbox']
            area_i =mask_i ['area']
            x1_i ,y1_i ,w_i ,h_i =bbox_i
            x2_i ,y2_i =x1_i +w_i ,y1_i +h_i

            for j in range (i +1 ,len (filtered_masks )):
                if j in to_remove :
                    continue

                mask_j =filtered_masks [j ]
                area_j =mask_j ['area']

                if area_j >area_i *0.8 :
                    continue

                x1_j ,y1_j ,w_j ,h_j =mask_j ['bbox']
                x2_j ,y2_j =x1_j +w_j ,y1_j +h_j

                xi1 =max (x1_i ,x1_j )
                yi1 =max (y1_i ,y1_j )
                xi2 =min (x2_i ,x2_j )
                yi2 =min (y2_i ,y2_j )

                if xi2 <=xi1 or yi2 <=yi1 :
                    continue

                bbox_inter_area =(xi2 -xi1 )*(yi2 -yi1 )

                if area_j <=0 or (bbox_inter_area /float (area_j ))<self .containment_iou :
                    continue

                seg_i =mask_i ['segmentation']
                seg_j =mask_j ['segmentation']
                roi_i =seg_i [yi1 :yi2 ,xi1 :xi2 ]
                roi_j =seg_j [yi1 :yi2 ,xi1 :xi2 ]

                intersection =np .logical_and (roi_i ,roi_j ).sum ()
                containment =intersection /float (area_j )

                if containment >=self .containment_iou :
                    to_remove .add (j )

        return [mask for idx ,mask in enumerate (filtered_masks )if idx not in to_remove ]

    def _merge_overlapping_masks (self ,masks :List [Dict [str ,Any ]])->List [Dict [str ,Any ]]:
        if len (masks )<=1 :
            print (f"🔗 Слияние масок: {len(masks)} → {len(masks)}")
            return masks

        print (f"🔗 Объединение перекрывающихся масок: {len(masks)} → {len(masks)}")
        return masks

    def _apply_mask_correction_fast (self ,masks :List [Dict [str ,Any ]])->List [Dict [str ,Any ]]:

        if not self .enable_mask_correction :
            return masks

        try :
            import cv2
        except ImportError :
            return masks

        corrected_masks =[]

        kernel_size =max (3 ,self .correction_kernel_size )
        kernel =cv2 .getStructuringElement (cv2 .MORPH_ELLIPSE ,(kernel_size ,kernel_size ))
        pad =kernel_size

        for mask_dict in masks :
            mask =mask_dict ['segmentation']
            area_val =int (mask_dict .get ('area',0 ))

            if area_val <100 :
                corrected_masks .append (mask_dict )
                continue

            x ,y ,w ,h =mask_dict .get ('bbox',(0 ,0 ,0 ,0 ))
            x =int (round (x ));y =int (round (y ));w =max (0 ,int (round (w )));h =max (0 ,int (round (h )))
            if w ==0 or h ==0 :
                corrected_masks .append (mask_dict )
                continue
            H ,W =mask .shape [:2 ]
            px1 =max (0 ,x -pad );py1 =max (0 ,y -pad )
            px2 =min (W ,x +w +pad );py2 =min (H ,y +h +pad )

            roi =mask [py1 :py2 ,px1 :px2 ].astype (np .uint8 )

            if self .erosion_iterations >0 and self .dilation_iterations >0 :

                roi_proc =cv2 .morphologyEx (roi ,cv2 .MORPH_OPEN ,kernel ,iterations =1 )
                roi_proc =cv2 .morphologyEx (roi_proc ,cv2 .MORPH_CLOSE ,kernel ,iterations =1 )
            else :
                roi_proc =roi

            cy1 =y -py1 ;cx1 =x -px1
            cy2 =cy1 +h ;cx2 =cx1 +w
            roi_core =roi_proc [cy1 :cy2 ,cx1 :cx2 ]

            smoothed_mask =mask .copy ()
            smoothed_mask [y :y +h ,x :x +w ]=roi_core .astype (bool )

            co =np .column_stack (np .where (roi_core ))
            corrected_mask_dict =mask_dict .copy ()
            corrected_mask_dict ['segmentation']=smoothed_mask
            if len (co )>0 :
                yy_min ,xx_min =co .min (axis =0 )
                yy_max ,xx_max =co .max (axis =0 )

                new_x =x +int (xx_min )
                new_y =y +int (yy_min )
                new_w =int (xx_max -xx_min )
                new_h =int (yy_max -yy_min )
                corrected_mask_dict ['bbox']=(new_x ,new_y ,new_w ,new_h )
                corrected_mask_dict ['area']=int ((roi_core >0 ).sum ())
            else :
                corrected_mask_dict ['bbox']=(0 ,0 ,0 ,0 )
                corrected_mask_dict ['area']=0

            corrected_masks .append (corrected_mask_dict )

        return corrected_masks