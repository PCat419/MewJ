from typing import TypeVar

T = TypeVar("T")


def pad_list(li: list[T], except_len: int, fill: T) -> list[T]:
    if len(li) < except_len:
        li = list(li)
        for i in range(except_len - len(li)):
            li.append(fill)
    return li


def relative_seating(a: int, b: int) -> int:
    return (a - b + 3) % 4
