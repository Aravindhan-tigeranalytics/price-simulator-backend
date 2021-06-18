from rest_framework.views import exception_handler
import logging.config
from utils import constants as CONST
logging.config.dictConfig(CONST.LOGGING_CONFIG)
logger = logging.getLogger(__name__)
def custom_exception_handler(exc, context):
    # Call REST framework's default exception handler first,
    # to get the standard error response.
    # print(exc , "EXC")
    # print(context , "context..")
    response = exception_handler(exc, context)
    print(response , "response")
    logger.error(exc)
    

    # Now add the HTTP status code to the response.
    if response is not None:
       
        logger.error(response.data)
        # print(response.status_code , "status code")
        # print(response.status_text , "status text")
        # print(response.data , "error response")
        response.data['status_code'] = response.status_code

    return response