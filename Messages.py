from dataclasses import dataclass, field
from langchain_core.runnables.base import Runnable
from langchain_core.runnables.config import RunnableConfig



@dataclass
class Doc:
    file:str
    index:int
    content:str


@dataclass
class Question:
    content:str
    correct_answer:str
    doc:Doc


@dataclass
class Answer:
    student_name:str
    content:str
    question:Question


@dataclass
class Book:
    file_path:str
    chunck_size:str = field(
        metadata={ "help": "one book will be split in several chuncks, which size equals to chunck_size." }
    )