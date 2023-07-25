from typing import Optional


class GathInnDrawer:
    def __init__(self,draw_meta: dict):
        self.__draw_meta=draw_meta

    def draw(self,output_image_file: Optional[str] = None):
        pass

class PureDrawer:
    pass

class CannyDrawer:
    pass

class OpenposeDrawer:
    pass