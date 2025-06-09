from torch.utils.data import RandomSampler, DataLoader

from data.module import EyeTrackingDataModule 
from data.utils import load_yaml_config

from onnxruntime.quantization import (
    CalibrationDataReader)

# Representative dataset function for calibration
class EyeTrackingDataReader(CalibrationDataReader):
    """
    A class used to provide a representative dataset for calibration.

    Attributes
    ----------
    train_dataloader : DataLoader
        The training data loader
    enum_data : iter
        Enumerator for iterating through the dataset
    """

    def __init__(self, model_path: str, train_dataloader: DataLoader) -> None:
        """
        Initializes the RepresentativeDataset class.

        Parameters
        ----------
        train_dataloader : DataLoader
            The data loader for training data
        """
        self.train_dataloader = train_dataloader
        self.enum_data = None  # Enumerator for calibration data 
        
        try:
            first_batch = next(iter(self.train_dataloader))
            print("First batch of data:", first_batch[0].shape)  # Print the shape of the first batch
        except StopIteration:
            print("train_dataloader is empty!")
            
        # Use inference session to get input shape
        session = onnxruntime.InferenceSession(model_path, None)
        (_, channel, height, width) = session.get_inputs()[0].shape
        self.input_name = session.get_inputs()[0].name

    def get_next(self) -> list:
        if self.enum_data is None:
            self.enum_data = self._create_enumerator()

        data = next(self.enum_data, None)
        if data is None:
            print("No data returned!") 
        return data

    def rewind(self) -> None:
        """
        Resets the enumeration of the dataset.
        """
        self.enum_data = None  # Reset the enumerator for the dataset

    def _create_enumerator(self):
        """
        Creates an iterator that generates representative dataset items.

        Yields
        -------
        list
            A list containing the input data for calibration
        """
        for input_data, _, _ in self.train_dataloader:
            input_data = input_data.detach().cpu().numpy().astype(np.float32)
            for i in range(input_data.shape[0]): 
                yield {self.input_name: input_data[i]} 
            