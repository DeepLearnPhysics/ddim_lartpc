import os

class WandbLogger:
    """
    Log using `Weights and Biases`.
    """
    def __init__(self, cfg):
        try:
            import wandb
            wandb.require("core")
        except ImportError:
            raise ImportError(
                "To use the Weights and Biases Logger please install wandb."
                "Run `pip install wandb` to install it."
            )
        
        self._wandb = wandb

        if not os.path.exists(cfg.wandb_path):
            os.makedirs(cfg.wandb_path, exist_ok=True)

        # Initialize a W&B run
        if self._wandb.run is None:
            self._wandb.init(
                project=cfg.wandb.project,
                config=vars(cfg),
                dir=cfg.wandb_path
            )

        self.config = self._wandb.config

        self.sample_table = self._wandb.Table(columns=['sr_image', 
                                                        'hr_image'])

    def log_metrics(self, metrics, commit=True): 
        """
        Log train/validation metrics onto W&B.

        metrics: dictionary of metrics to be logged
        """
        self._wandb.log(metrics, commit=commit)

    def log_image(self, key_name, image_array):
        """
        Log image array onto W&B.

        key_name: name of the key 
        image_array: numpy array of image.
        """
        self._wandb.log({key_name: self._wandb.Image(image_array)}, commit=True)

    def log_images(self, key_name, list_images):
        """
        Log list of image array onto W&B

        key_name: name of the key 
        list_images: list of numpy image arrays
        """
        self._wandb.log({key_name: [self._wandb.Image(img) for img in list_images]})


    def log_sample_data(self, sr_img, hr_img):
        """
        Add data row-wise to the initialized table.
        """
        self.sample_table.add_data(
            self._wandb.Image(sr_img),
            self._wandb.Image(hr_img)
        )

    def log_sample_table(self, commit=False):
        """
        Log the table
        """
        self._wandb.log({'sample_data': self.sample_table}, commit=commit)
