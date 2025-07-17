import sys
import logging
from logger import logger # Import the configured logger from your logger.py Canvas

class CustomException(Exception):
    """
    Custom exception class to provide detailed error information including
    the file name, line number, and the original error message.
    """
    def __init__(self, error_message, error_details: sys):
        # Call the base Exception class constructor
        super().__init__(error_message)
        self.error_message = error_message

        # Get the traceback information
        # exc_info() returns a tuple (type, value, traceback)
        _, _, exc_tb = error_details.exc_info()

        # Extract line number and file name from the traceback object
        self.lineno = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename

        # Log the detailed error using the project's configured logger
        logger.error(
            f"Error in file: [{self.file_name}] at line: [{self.lineno}] - Error message: [{self.error_message}]"
        )

    def __str__(self):
        """
        Returns a user-friendly string representation of the exception.
        """
        return "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
            self.file_name, self.lineno, str(self.error_message)
        )

    def __repr__(self):
        """
        Returns a developer-friendly representation of the exception for debugging.
        """
        return f"CustomException(file_name='{self.file_name}', lineno={self.lineno}, error_message='{self.error_message}')"

# Example usage (for testing this module independently)
if __name__ == "__main__":
    # This block demonstrates how CustomException would be used.
    # In your main Streamlit app, you would wrap potentially problematic code
    # in try-except blocks and raise CustomException.
    try:
        a = 1 / 0  # This will cause a ZeroDivisionError
    except Exception as e:
        # Raise your custom exception, passing the original exception and sys module
        # The CustomException's __init__ will automatically log the error details.
        raise CustomException(e, sys)

