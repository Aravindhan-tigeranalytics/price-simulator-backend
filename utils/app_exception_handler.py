from rest_framework.views import exception_handler

def custom_exception_handler(exc, context):
    # Call REST framework's default exception handler first,
    # to get the standard error response.
    print(exc , "EXC")
    print(context , "context..")
    response = exception_handler(exc, context)
    print(response , "response")
    

    # Now add the HTTP status code to the response.
    if response is not None:
        print(response.status_code , "status code")
        print(response.status_text , "status text")
        print(response.data , "error response")
        response.data['status_code'] = response.status_code

    return response