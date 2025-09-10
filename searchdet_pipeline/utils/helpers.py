import numpy as np
from typing import Any

def save_json (data :Any ,filepath :str )->None :

    import json

    class NumpyEncoder (json .JSONEncoder ):
        def default (self ,obj ):
            if isinstance (obj ,np .integer ):
                return int (obj )
            elif isinstance (obj ,np .floating ):
                return float (obj )
            elif isinstance (obj ,np .ndarray ):
                return obj .tolist ()
            elif isinstance (obj ,np .bool_ ):
                return bool (obj )
            return super ().default (obj )

    with open (filepath ,'w',encoding ='utf-8')as f :
        json .dump (data ,f ,ensure_ascii =False ,indent =2 ,cls =NumpyEncoder )