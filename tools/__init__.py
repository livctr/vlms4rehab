import logging

tools_logger = logging.getLogger("tools")
tools_logger.setLevel(logging.INFO)

# Avoid duplicate handlers if imported multiple times
if not tools_logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    tools_logger.addHandler(ch)
