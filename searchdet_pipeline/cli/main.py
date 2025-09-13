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
def create_parser ()->argparse .ArgumentParser :

    parser =argparse .ArgumentParser (
    prog ='searchdet-pipeline',
    description ='SearchDet Pipeline - профессиональный инструмент для детекции объектов на изображениях',
    formatter_class =argparse .RawDescriptionHelpFormatter ,
    )

    subparsers =parser .add_subparsers (dest ='command',help ='Доступные команды')

    detect_parser =subparsers .add_parser (
    'detect',
    help ='Детекция объектов на одном изображении',
    description ='Выполняет детекцию объектов на указанном изображении'
    )
    try :
        from .detect import _add_detect_arguments
    except ImportError :
        from searchdet_pipeline .cli .detect import _add_detect_arguments
    _add_detect_arguments (detect_parser )
    return parser
def main ():

    print ("🚀 ЗАПУСК МОДУЛЬНОГО SEARCHDET ПАЙПЛАЙНА")
    print ("="*60 )
    print ("📁 Точка входа: searchdet_pipeline/cli/main.py → main()")

    parser =create_parser ()
    args =parser .parse_args ()
    if not args .command :
        parser .print_help ()
        return 0

    try :
        if args .command =='detect':
            try :
                from .detect import execute_detect
            except ImportError :
                from searchdet_pipeline .cli .detect import execute_detect
            return execute_detect (args )
        else :
            print (f"❌ Неизвестная команда: {args.command}")
            return 1

    except KeyboardInterrupt :
        print ("\n⚠️ Обработка прервана пользователем")
        return 130
    except Exception as e :
        print (f"❌ Неожиданная ошибка: {e}")
        if hasattr (args ,'verbose')and args .verbose :
            import traceback
            traceback .print_exc ()
        return 1

if __name__ =="__main__":
    import sys
    sys .exit (main ())
