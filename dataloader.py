import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from typing import Tuple, Optional
from transformers import PreTrainedTokenizerFast
from model import ModelArgs
from trainer import TrainingConfig
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

class TinyStoriesDataset(IterableDataset, Dataset):
    """Dataset for TinyStories, supporting streaming and non-streaming modes."""
    
    def __init__(
        self,
        seq_len: int = 128,
        is_streaming: bool = True,
        device: str = 'cuda',
        prefetch_size: int = 4096,
        token_file_path: Optional[str] = None,
        token_ids: Optional[torch.Tensor] = None
    ):
        """Initialize the TinyStories dataset.
        
        Args:
            seq_len (int): Sequence length for input and target. Defaults to 128.
            is_streaming (bool): Whether to use streaming mode. Defaults to True.
            device (str): Device to move data to ('cuda' or 'cpu'). Defaults to 'cuda'.
            prefetch_size (int): Number of samples to prefetch in streaming mode. Defaults to 4096.
            token_file_path (Optional[str]): Path to tokenized data file. Defaults to None.
            token_ids (Optional[torch.Tensor]): Pre-loaded token IDs. Defaults to None.
        
        Raises:
            ValueError: If neither token_file_path nor token_ids is provided, or if token length is too short.
        """
        super().__init__()
        self.seq_len = seq_len
        self.is_streaming = is_streaming
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.prefetch_size = prefetch_size

        if token_ids is not None:
            self.token_ids = token_ids  # Keep on CPU initially
        elif token_file_path is not None:
            try:
                self.token_ids = torch.load(token_file_path, map_location='cpu')
            except Exception as e:
                raise ValueError(f"Failed to load token file {token_file_path}: {e}")
        else:
            raise ValueError("Either 'token_file_path' or 'token_ids' must be provided.")

        self.token_len = len(self.token_ids)
        if self.token_len < self.seq_len + 1:
            raise ValueError(f"Token length ({self.token_len}) is too short for seq_len ({self.seq_len}).")

        logging.info("TinyStoriesDataset initialized: token_len=%d, seq_len=%d, device=%s, streaming=%s",
                     self.token_len, self.seq_len, self.device, self.is_streaming)

    def __len__(self) -> int:
        """Return the number of samples in non-streaming mode.
        
        Raises:
            NotImplementedError: If in streaming mode.
        """
        if self.is_streaming:
            raise NotImplementedError("Length is not defined for streaming dataset.")
        return self.token_len - self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample (input and target) by index.
        
        Args:
            idx (int): Index of the sample.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Input and target tensors.
        """
        x = self.token_ids[idx:idx + self.seq_len].to(self.device, non_blocking=True)
        y = self.token_ids[idx + 1:idx + self.seq_len + 1].to(self.device, non_blocking=True)
        return x, y

    def __iter__(self):
        """Iterate over samples in streaming or non-streaming mode.
        
        Yields:
            Tuple[torch.Tensor, torch.Tensor]: Input and target tensors.
        """
        if not self.is_streaming:
            for idx in range(len(self)):
                yield self.__getitem__(idx)
        else:
            while True:
                idxs = torch.randint(
                    0, self.token_len - self.seq_len - 1,
                    (self.prefetch_size,), device='cpu'
                )
                for idx in idxs:
                    yield self.__getitem__(idx)

class DataLoaderFactory:
    """Factory class to create DataLoaders for training and validation."""
    
    def __init__(
        self,
        model_args: ModelArgs,
        cfg: TrainingConfig,
        train_token_file: str = 'tokenized-train-samples_vocab-10k.pt',
        valid_token_file: str = 'tokenized-valid-samples_vocab-10k.pt',
        tokenizer_file: str = 'bpe-tokenizer_tinystories.json',
        pad_token: str = '</s>'
    ):
        """Initialize the DataLoaderFactory.
        
        Args:
            model_args (ModelArgs): Model configuration.
            cfg (TrainingConfig): Training configuration.
            train_token_file (str): Path to training token file. Defaults to 'tokenized-train-samples_vocab-10k.pt'.
            valid_token_file (str): Path to validation token file. Defaults to 'tokenized-valid-samples_vocab-10k.pt'.
            tokenizer_file (str): Path to tokenizer file. Defaults to 'bpe-tokenizer_tinystories.json'.
            pad_token (str): Padding token for tokenizer. Defaults to '</s>'.
        """
        self.model_args = model_args
        self.cfg = cfg
        self.train_token_file = train_token_file
        self.valid_token_file = valid_token_file
        
        try:
            self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file, pad_token=pad_token)
        except Exception as e:
            logging.error("Failed to load tokenizer from %s: %s", tokenizer_file, e)
            raise
        
        logging.info("DataLoaderFactory initialized with tokenizer vocab_size=%d", self.tokenizer.vocab_size)

    def create_train_loader(self) -> DataLoader:
        """Create the training DataLoader.
        
        Returns:
            DataLoader: Training DataLoader with streaming dataset.
        """
        dataset = TinyStoriesDataset(
            token_file_path=self.train_token_file,
            seq_len=self.cfg.seq_len,
            is_streaming=True,
            device='cuda',
            prefetch_size=self.cfg.batch_size * 2
        )
        
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            pin_memory=False,
            drop_last=True,
            num_workers=0
        )
        
        logging.info("Training DataLoader created: batch_size=%d, seq_len=%d",
                     self.cfg.batch_size, self.cfg.seq_len)
        return loader

    def create_valid_loader(self) -> DataLoader:
        """Create the validation DataLoader.
        
        Returns:
            DataLoader: Validation DataLoader with non-streaming dataset.
        """
        try:
            valid_ids = torch.load(self.valid_token_file, map_location='cpu')
        except Exception as e:
            logging.error("Failed to load validation token file %s: %s", self.valid_token_file, e)
            raise
        
        dataset = TinyStoriesDataset(
            token_ids=valid_ids,
            seq_len=self.cfg.seq_len,
            is_streaming=False,
            device='cuda'
        )
        
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            pin_memory=False,
            num_workers=0
        )
        
        logging.info("Validation DataLoader created: batch_size=%d, seq_len=%d",
                     self.cfg.batch_size, self.cfg.seq_len)
        return loader

    def create_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create both training and validation DataLoaders.
        
        Returns:
            Tuple[DataLoader, DataLoader]: Training and validation DataLoaders.
        """
        train_loader = self.create_train_loader()
        valid_loader = self.create_valid_loader()
        return train_loader, valid_loader