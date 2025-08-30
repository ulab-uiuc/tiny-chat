import logging
from logging import FileHandler, StreamHandler
from pathlib import Path

from beartype.typing import Any, Dict, List, Literal, Mapping, Union
from termcolor import colored

LogType = Union[List[Dict[str, str]], None]

ColorType = Literal[
    'red',
    'green',
    'yellow',
    'blue',
    'magenta',
    'cyan',
    'light_grey',
    'dark_grey',
    'light_red',
    'light_green',
    'light_yellow',
    'light_blue',
    'light_magenta',
    'light_cyan',
    'white',
]

LOG_COLORS: Mapping[str, ColorType] = {
    'BACKGROUND LOG': 'blue',
    'ACTION': 'green',
    'OBSERVATION': 'yellow',
    'DETAIL': 'cyan',
    'ERROR': 'red',
    'PLAN': 'light_magenta',
}


class ColoredFormatter(logging.Formatter):
    def format(self: logging.Formatter, record: logging.LogRecord) -> Any:
        msg_type = record.__dict__.get('msg_type', None)
        if msg_type in LOG_COLORS:
            msg_type_color = colored(msg_type, LOG_COLORS[msg_type])
            msg = colored(record.msg, LOG_COLORS[msg_type])
            time_str = colored(
                self.formatTime(record, self.datefmt), LOG_COLORS[msg_type]
            )
            name_str = colored(record.name, LOG_COLORS[msg_type])
            level_str = colored(record.levelname, LOG_COLORS[msg_type])
            if msg_type == 'ERROR':
                return f'{time_str} - {name_str}:{level_str}: {record.filename}:{record.lineno}\n{msg_type_color}\n{msg}'
            return f'{time_str} - {msg_type_color}\n{msg}'
        elif msg_type == 'STEP':
            msg = '\n\n==============\n' + record.msg + '\n'
            return f'{msg}'
        return logging.Formatter.format(self, record)


console_formatter = ColoredFormatter(
    '\033[92m%(asctime)s - %(name)s:%(levelname)s\033[0m: %(filename)s:%(lineno)s - %(message)s',
    datefmt='%H:%M:%S',
)

file_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
)


def get_console_handler() -> Any:
    """
    Returns a console handler for logging.
    """
    console_handler = StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    return console_handler


def get_file_handler(log_file_path: str = 'logs/tiny_chat.log') -> Any:
    """
    Returns a file handler for logging.
    """
    # 确保日志目录存在
    log_dir = Path(log_file_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    file_handler = FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    return file_handler


def setup_logging(log_file_path: str = None, log_level: str = 'INFO') -> None:
    """
    设置统一的日志配置
    """
    # 获取根logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # 清除现有的handlers，避免重复
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 添加console handler
    root_logger.addHandler(get_console_handler())

    # 如果指定了文件路径，添加文件handler
    if log_file_path:
        root_logger.addHandler(get_file_handler(log_file_path))


# 创建tiny_chat专用的logger
logger = logging.getLogger('tiny_chat')
logger.setLevel(logging.DEBUG)
# 不设置propagate=False，让日志可以向上传播到根logger
# 这样所有子logger都能通过根logger输出日志

# 不为tiny_chat logger添加重复的handler，避免重复输出
# 让它通过根logger的handlers输出
