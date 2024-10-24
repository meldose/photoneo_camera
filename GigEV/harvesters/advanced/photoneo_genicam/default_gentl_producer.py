import os # import the os platform library
from pathlib import Path # import pathlib library
from sys import platform # import platform module from sys 

GENTL_PATHS = os.getenv("GENICAM_GENTL64_PATH").split(os.pathsep)


default_gentl_producer_file = "libmvGenTLProducer.so" # consider this as the default producer file 
if platform == "linux": # if the platform is Linux
    default_gentl_producer_file = "mvGenTLProducer.cti" # consider the producer file as the .cti file for Windows


def first(iterable, predicate, default=None): # deifine the function first with having iterable, predicate, default
    return next((i for i in iterable if predicate(i)), default)


def find_producer_path(producer_file_name: str) -> Path:
    return first(
        (Path(p) / producer_file_name for p in GENTL_PATHS),
        Path.exists,
        default="GentlProducerNotFound",
    )


producer_path: Path = find_producer_path(default_gentl_producer_file)# consider the producer path as the default

print(f"Loading: {producer_path}") # print the producer path
