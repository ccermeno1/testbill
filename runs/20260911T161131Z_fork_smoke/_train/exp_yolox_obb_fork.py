# Generado por testbank. No editar a mano: se reescribe en cada ejecucion.
import os

import torch

from yolox.exp import ExpOBB_KLD as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 1
        self.depth = 0.33
        self.width = 0.5
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        # Sin redimensionado aleatorio: este proyecto entrena de forma
        # determinista y `random_size` mete una fuente de ruido por lote.
        self.random_size = None
        self.seed = 20260910
        self.exp_name = "yolox-obb-fork-small"
        self.output_dir = r"runs\20260911T161131Z_fork_smoke\_train"
        self.data_num_workers = 0

        self.degrees = 0.0
        self.translate = 0.1
        self.scale = (0.5, 1.5)
        self.shear = 2.0
        self.perspective = 0.0
        self.enable_mixup = False

        self.warmup_epochs = 0
        self.max_epoch = 1
        self.no_aug_epochs = 0
        self.basic_lr_per_img = 0.0025 / 16.0
        self.scheduler = "yoloxwarmcos"
        self.min_lr_ratio = 0.05
        self.ema = True
        self.save_interval = 1
        self.print_interval = 5
        # Su evaluador devuelve 0.0 fijo, asi que evaluar durante el
        # entrenamiento solo gasta tiempo. Se puntua despues con `evaluate`.
        self.eval_interval = 1 + 1

        self.test_conf = 0.01
        self.nmsthre = 0.5

    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        from yolox.data import (
            DataLoader,
            DOTAOBBDetection,
            InfiniteSampler,
            MosaicDetectionOBB,
            TrainTransformOBB,
            YoloBatchSampler,
        )

        dataset = DOTAOBBDetection(
            data_dir=r"runs\20260911T161131Z_fork_smoke\_train\voc",
            image_sets=[("2007", "trainval")],
            img_size=self.input_size,
            preproc=TrainTransformOBB(
                rgb_means=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_labels=50
            ),
        )
        dataset = MosaicDetectionOBB(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransformOBB(
                rgb_means=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_labels=100
            ),
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=self.enable_mixup,
        )
        self.dataset = dataset
        sampler = InfiniteSampler(len(self.dataset), seed=self.seed or 0)
        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            input_dimension=self.input_size,
            mosaic=not no_aug,
        )
        return DataLoader(
            self.dataset,
            num_workers=self.data_num_workers,
            pin_memory=True,
            batch_sampler=batch_sampler,
        )

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        from yolox.data import DOTAOBBDetection, ValTransformOBB

        dataset = DOTAOBBDetection(
            data_dir=r"runs\20260911T161131Z_fork_smoke\_train\voc",
            image_sets=[("2007", "val")],
            img_size=self.test_size,
            preproc=ValTransformOBB(rgb_means=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        )
        return torch.utils.data.DataLoader(
            dataset,
            num_workers=self.data_num_workers,
            pin_memory=True,
            sampler=torch.utils.data.SequentialSampler(dataset),
            batch_size=batch_size,
        )

    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        from yolox.evaluators import DOTAEvaluator

        return DOTAEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )
