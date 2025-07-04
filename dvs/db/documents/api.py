import dvs


class Documents:
    def __init__(self, dvs: dvs.DVS):
        self.dvs = dvs

    def touch(self) -> bool:
        return True
