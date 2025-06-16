from src import logger
from src.utils.common import format_symbol  # should handle both str and list if needed
from datetime import datetime

class hist_data_retrival_pipeline:
    @staticmethod
    def hist_data():
        raw_input = input("Enter symbols (comma-separated): ")
        input_symbols = [s.strip().upper() for s in raw_input.split(",") if s.strip()]
        
        # Apply format_symbol to each symbol
        symbols_list = [format_symbol(symbol) for symbol in input_symbols]
  
        start_date = input("Enter the start date (YYYY-MM-DD): ")

        while True:
            end_date = input("Enter the end date (YYYY-MM-DD): ")
            try:
                end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
                today = datetime.today()

                if end_date_obj >= today:
                    print("❌ End date should be **before** today's date. Please enter again.")
                else:
                    break
            except ValueError:
                print("❌ Invalid date format. Use YYYY-MM-DD.")

        return symbols_list, start_date, end_date


if __name__ == "__main__":
    symbols, start_date, end_date = hist_data_retrival_pipeline.hist_data()
    print("Formatted symbols:", symbols)
    print("Start date:", start_date)
    print("End date:", end_date)
