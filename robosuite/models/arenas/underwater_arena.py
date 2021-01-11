from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import xml_path_completion


class UnderwaterArena(Arena):
    """Underwater workspace."""

    def __init__(self):
        super().__init__(xml_path_completion("arenas/underwater_arena.xml"))
