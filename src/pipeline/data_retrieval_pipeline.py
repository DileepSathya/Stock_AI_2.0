from src import logger
from src.configurations.config import config_manager



class data_retrieval:
    def data_retrieval():
        auth_code=config_manager.authentication()
        