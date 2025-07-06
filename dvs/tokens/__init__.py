import tiktoken

import dvs


class Tokens:
    def __init__(self, dvs: "dvs.DVS"):
        self.dvs = dvs
        self.enc = tiktoken.encoding_for_model("gpt-4o")
