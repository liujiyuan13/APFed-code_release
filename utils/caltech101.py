import os
import os.path
from typing import Any, Callable, List, Optional, Tuple, Union

from PIL import Image
import scipy.io

from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from torchvision.datasets.vision import VisionDataset

from sklearn.model_selection import train_test_split


class Caltech101(VisionDataset):
    """`Caltech 101 <https://data.caltech.edu/records/20086>`_ Dataset.

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Args:
        root (string): Root directory of dataset where directory
            ``caltech101`` exists or will be saved to if download is set to True.
        target_type (string or list, optional): Type of target to use, ``category`` or
            ``annotation``. Can also be a list to output a tuple with all specified
            target types.  ``category`` represents the target class, and
            ``annotation`` is a list of points from a hand-generated outline.
            Defaults to ``category``.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        target_type: Union[List[str], str] = "category",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(os.path.join(root, "caltech101"), transform=transform, target_transform=target_transform)
        os.makedirs(self.root, exist_ok=True)
        if isinstance(target_type, str):
            target_type = [target_type]
        self.target_type = [verify_str_arg(t, "target_type", ("category", "annotation")) for t in target_type]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        self.categories = sorted(os.listdir(os.path.join(self.root, "101_ObjectCategories")))
        self.categories.remove("BACKGROUND_Google")  # this is not a real class

        # For some reason, the category names in "101_ObjectCategories" and
        # "Annotations" do not always match. This is a manual map between the
        # two. Defaults to using same name, since most names are fine.
        name_map = {
            "Faces": "Faces_2",
            "Faces_easy": "Faces_3",
            "Motorbikes": "Motorbikes_16",
            "airplanes": "Airplanes_Side_2",
        }
        self.annotation_categories = list(map(lambda x: name_map[x] if x in name_map else x, self.categories))

        self.indexes: List[int] = []
        self.y = []
        for (i, c) in enumerate(self.categories):
            n = len(os.listdir(os.path.join(self.root, "101_ObjectCategories", c)))
            self.indexes.extend(range(1, n + 1))
            self.y.extend(n * [i])

        # load train-test split
        tmp = scipy.io.loadmat(os.path.join(self.root, 'split.mat'))
        index_train, index_test = tmp['index_train'], tmp['index_test']

        # read, transform and save

        images, targets = [], []

        for index in range(len(self.indexes)):

            img = Image.open(
                os.path.join(
                    self.root,
                    "101_ObjectCategories",
                    self.categories[self.y[index]],
                    f"image_{self.indexes[index]:04d}.jpg",
                )
            ).convert("RGB")

            target: Any = []
            for t in self.target_type:
                if t == "category":
                    target.append(self.y[index])
                elif t == "annotation":
                    data = scipy.io.loadmat(
                        os.path.join(
                            self.root,
                            "Annotations",
                            self.annotation_categories[self.y[index]],
                            f"annotation_{self.indexes[index]:04d}.mat",
                        )
                    )
                    target.append(data["obj_contour"])
            target = tuple(target) if len(target) > 1 else target[0]

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            images.append(img)
            targets.append(target)

        index_tmp = index_train.tolist()[0] if split == 'train' else index_test.tolist()[0]
        self._images, self._targets = [images[i] for i in index_tmp], [targets[i] for i in index_tmp]


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where the type of target specified by target_type.
        """

        img, target = self._images[index], self._targets[index]

        return img, target

    def _check_integrity(self) -> bool:
        # can be more robust and check hash of files
        return os.path.exists(os.path.join(self.root, "101_ObjectCategories"))

    def __len__(self) -> int:
        return len(self._targets)

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_and_extract_archive(
            "https://drive.google.com/file/d/137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp",
            self.root,
            filename="101_ObjectCategories.tar.gz",
            md5="b224c7392d521a49829488ab0f1120d9",
        )
        download_and_extract_archive(
            "https://drive.google.com/file/d/175kQy3UsZ0wUEHZjqkUDdNVssr7bgh_m",
            self.root,
            filename="Annotations.tar",
            md5="6f83eeb1f24d99cab4eb377263132c91",
        )

    def extra_repr(self) -> str:
        return "Target type: {target_type}".format(**self.__dict__)

# import torchvision
#
# Caltech101('../../APFed-code/datasets/', split='train', transform=torchvision.transforms.Compose([
#     torchvision.transforms.Resize(64),
#     torchvision.transforms.CenterCrop(64),
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Normalize(
#         [0.485, 0.456, 0.406],
#         [0.229, 0.224, 0.225])]),
#            download=True)
