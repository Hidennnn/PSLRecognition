
class PathToImageNotExistError(Exception):

    def __init__(self):
        self.message = "Image on this path doesn't exist."
        super().__init__(self.message)


class PathToVideoNotExistError(Exception):

    def __init__(self):
        self.message = "Video on this path doesn't exist."
        super().__init__(self.message)


class ImageNotExistError(Exception):

    def __init__(self):
        self.message = "Image doesn't exist. Check if input image isn't None."
        super().__init__(self.message)
