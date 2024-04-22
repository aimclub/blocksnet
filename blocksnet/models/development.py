from .city import Block


class BlockDevelopment:
    ...


class Development:
    def __init__(self, blocks: list[Block]):
        ...

    def to_dict(self):
        ...
