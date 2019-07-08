class Constants:
    #  Always reference from Sources root
    database_dir = "./datasets"
    all_imgs_no_class = database_dir + "/all_imgs_no_class"
    predict_dir = database_dir + "/predict_files/predict_mole"

    tfdata_dir = "./tfdata_dir"
    tfrec_dir = tfdata_dir + "/tfrec_dir"
    tfcache_dir = tfdata_dir + "/tfcache_dir"

    hmnist_metadata = database_dir + "/HAM10000_metadata.csv"

    # ------------------------------------------------------------------------------

    def get_database_dir_path(self):
        return self.database_dir

    def get_all_imgs_no_class_dir_path(self):
        return self.all_imgs_no_class

    def get_predict_dir_path(self):
        return self.predict_dir

    def get_tfdata_dir_path(self):
        return self.tfdata_dir

    def get_tfrec_dir_path(self):
        return self.tfrec_dir

    def get_tfcache_dir_path(self):
        return self.tfcache_dir

    def get_hmnist_metadata_file_path(self):
        return self.hmnist_metadata
