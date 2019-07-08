# Params
class Params:
    pixels = 512
    image_size = (pixels, pixels)
    batch_size = 8
    tf_cache_train = "cache_train.tf-data"
    tf_cache_validate = "cache_validate.tf-data"
    tfrec_train = "train.tfrec"
    tfrec_validate = "validate.tfrec"

    def get_pixels(self):
        return self.pixels

    def get_image_size(self):
        return self.image_size

    def get_batch_size(self):
        return self.batch_size

    def get_tf_cache_train_filename(self):
        return self.tf_cache_train

    def get_tf_cache_validate_filename(self):
        return self.tf_cache_validate

    def get_tfrec_train_filename(self):
        return self.tfrec_train

    def get_tfrec_validate_filename(self):
        return self.tfrec_validate
