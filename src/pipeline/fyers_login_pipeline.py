from src import logger
from src.utils.common import read_yaml, write_yaml
from src.constants import CONFIG_FILE_PATH
from src.configurations.config import config_manager
redirect_url, client_id, secret_key, grant_type, response_type, state = config_manager.login_info()