from rest_framework.exceptions import APIException

class CountExceedException(APIException):
    status_code = 400
    default_detail = 'No of scenarios Exceeded 20, cannot save'
    default_code = 'count_exceeded'

class AlredyExistsException(APIException):
    status_code = 400
    default_detail = 'Name Alredy Exists'
    default_code = 'name_exists'

class EmptyException(APIException):
    status_code = 400
    default_detail = 'Name cannot be empty'
    default_code = 'name_empty'


