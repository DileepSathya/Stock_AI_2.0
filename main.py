from src import logger
from src.components.fyers_login import fyers_login
from src.components.data_retrieval import DataRetrieval


def login():
    STAGE="login Stage"
    try:
        logger.info(f"-----{STAGE}-----")
        login=fyers_login.login()
        logger.info(f"{STAGE} sucessful")
    except Exception as e:
        logger.exception(e)
        raise e
    

def user_info():
    STAGE="user information- Stage"
    try:
        logger.info(f"-----{STAGE}-----")
        data=DataRetrieval.userdata()
        logger.info(f"{STAGE} sucessful")
    except Exception as e:
        logger.exception(e)
        raise e
    


def historical_data():
    STAGE = "Historical Data Retrieval - Stage"
    try:
        logger.info(f"----- {STAGE} -----")
        
        retriever = DataRetrieval()
        retriever.hist_data()

        logger.info(f"{STAGE} successful")

    except Exception as e:
        logger.exception(f"{STAGE} failed: {e}")
        raise




if __name__=="__main__":


    historical_data()