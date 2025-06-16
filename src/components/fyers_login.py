from src import logger
from fyers_apiv3 import fyersModel
import webbrowser
from src.utils.common import read_yaml, write_yaml
from src.constants import CONFIG_FILE_PATH
from src.configurations.config import config_manager  # your original class

class fyers_login:
    def login():
        redirect_uri, client_id, secret_key, grant_type, response_type, state = config_manager.login_info()

        # Start session
        appSession = fyersModel.SessionModel(
            client_id=client_id,
            redirect_uri=redirect_uri,
            response_type=response_type,
            state=state,
            secret_key=secret_key,
            grant_type=grant_type
        )

        # Generate URL and open browser
        generateTokenUrl = appSession.generate_authcode()

        webbrowser.open(generateTokenUrl, new=1)

        # Take auth code and generate token
        auth_code = input("Enter the authentication code: ")
        appSession.set_token(auth_code)
        response = appSession.generate_token()


        # Extract and save access token
        try:
            access_token = response["access_token"]
            

            # Read current YAML
            config_data = read_yaml(CONFIG_FILE_PATH)

            # Save access token under Input_params
            config_data["Input_params"]["auth_code"] = access_token

            # Write updated YAML back
            write_yaml(CONFIG_FILE_PATH, config_data)
            logger.info("Access token saved successfully in config.yaml")
            logger.info("access_token",access_token)
        except Exception as e:
            print("Error:", e)
            print("Response:", response)
