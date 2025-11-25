import sys
from src.logging import logger

def error_message_detail(error,error_detail:sys):  # type: ignore
    _,_, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename  # type: ignore
    error_message = (
    "Error occurred in \npython script name [{0}] "
    "\nline number [{1}] \nerror message [{2}]"
    ).format(file_name, exc_tb.tb_lineno, str(error)) # type: ignore

    return error_message
    
class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):  # type: ignore
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)
        
    def __str__(self): 
        return self.error_message
    
    
if __name__ == '__main__':
    try:
        logger.logging.info('Enter the try block')
        a=1/0
        print('not printed',a)
    except Exception as e:
        raise NetworkSecurityException(e,sys)
