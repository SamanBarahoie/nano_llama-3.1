import pytest
import torch
from model import ModelArgs, apply_scaling, precompute_freqs_cis, DecoderBlock, llama
from utils import RMSNorm, FeedForward, MultiHeadAttention
from dataloader import TinyStoriesDataset, DataLoaderFactory
from trainer import TrainingConfig
from torch.utils.data import DataLoader

@pytest.fixture
def model_args():
    """Fixture to provide ModelArgs with TinyStories settings."""
    return ModelArgs(
        dim=4096,
        n_heads=32,
        n_kv_heads=8,
        intermediate_dim=11008,
        multiple_of=256,
        ffn_dim_multiplier=1.0,
        norm_eps=1e-5,
        max_batch_size=2,
        max_seq_len=128,
        vocab_size=10000,
        n_layers=32,
        rope_theta=10000.0,
        use_scaled_rope=False,
        flash=True
    )

@pytest.fixture
def training_config():
    """Fixture to provide TrainingConfig."""
    return TrainingConfig(
        batch_size=2,
        seq_len=128,
        epochs=3,
        steps_per_epoch=100,
        report_interval=20,
        grad_clip_norm=1.0
    )

@pytest.fixture
def device():
    """Fixture to provide device (cuda if available, else cpu)."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def test_rmsnorm(model_args, device):
    """Test RMSNorm module."""
    norm = RMSNorm(dim=model_args.dim, eps=model_args.norm_eps).half().to(device)
    x = torch.randn(2, 128, 4096, device=device, dtype=torch.float16)
    output = norm(x)
    assert output.shape == (2, 128, 4096), f"Expected shape (2, 128, 4096), got {output.shape}"
    assert output.dtype == torch.float16, f"Expected dtype float16, got {output.dtype}"

def test_feedforward(model_args, device):
    """Test FeedForward module."""
    ffn = FeedForward(
        dim=model_args.dim,
        intermediate_dim=model_args.intermediate_dim,
        multiple_of=model_args.multiple_of,
        ffn_dim_multiplier=model_args.ffn_dim_multiplier
    ).half().to(device)
    x = torch.randn(2, 128, 4096, device=device, dtype=torch.float16)
    output = ffn(x)
    assert output.shape == (2, 128, 4096), f"Expected shape (2, 128, 4096), got {output.shape}"
    assert output.dtype == torch.float16, f"Expected dtype float16, got {output.dtype}"

def test_multihead_attention(model_args, device):
    """Test MultiHeadAttention module."""
    attn = MultiHeadAttention(model_args).half().to(device)
    x = torch.randn(2, 128, 4096, device=device, dtype=torch.float16)
    freqs_cis = precompute_freqs_cis(model_args.dim // model_args.n_heads, model_args.max_seq_len).to(device)
    output = attn(x, start_pos=0, freqs_cis=freqs_cis, mask=None)
    assert output.shape == (2, 128, 4096), f"Expected shape (2, 128, 4096), got {output.shape}"
    assert output.dtype == torch.float16, f"Expected dtype float16, got {output.dtype}"

def test_decoder_block(model_args, device):
    """Test DecoderBlock module."""
    block = DecoderBlock(model_args).half().to(device)
    x = torch.randn(2, 128, 4096, device=device, dtype=torch.float16)
    freqs_cis = precompute_freqs_cis(model_args.dim // model_args.n_heads, model_args.max_seq_len).to(device)
    output = block(x, start_pos=0, freqs_cis=freqs_cis, mask=None)
    assert output.shape == (2, 128, 4096), f"Expected shape (2, 128, 4096), got {output.shape}"
    assert output.dtype == torch.float16, f"Expected dtype float16, got {output.dtype}"

# def test_transformer_inference(model_args, device):
#     """Test Transformer in inference mode."""
#     model = llama(model_args).half().to(device)
#     input_tokens = torch.randint(0, model_args.vocab_size, (2, 128), device=device)
#     logits, loss = model(input_tokens, start_pos=0)
#     assert logits.shape == (2, 128, model_args.vocab_size), f"Expected shape (2, 128, {model_args.vocab_size}), got {logits.shape}"
#     assert loss is None, "Expected loss to be None in inference mode"
#     assert logits.dtype == torch.float32, f"Expected dtype float32, got {logits.dtype}"

# def test_transformer_training(model_args, device):
#     """Test Transformer in training mode."""
#     model = llama(model_args).half().to(device)
#     input_tokens = torch.randint(0, model_args.vocab_size, (2, 128), device=device)
#     targets = torch.randint(0, model_args.vocab_size, (2, 128), device=device)
#     logits, loss = model(input_tokens, start_pos=-1, targets=targets)
#     assert logits.shape == (2, 128, model_args.vocab_size), f"Expected shape (2, 128, {model_args.vocab_size}), got {logits.shape}"
#     assert isinstance(loss, torch.Tensor), "Expected loss to be a tensor"
#     assert loss.dtype == torch.float32, f"Expected loss dtype float32, got {loss.dtype}"

def test_tinystories_dataset(model_args, device):
    """Test TinyStoriesDataset in non-streaming mode."""
    token_ids = torch.randint(0, model_args.vocab_size, (1000,), dtype=torch.long)
    dataset = TinyStoriesDataset(
        token_ids=token_ids,
        seq_len=model_args.max_seq_len,
        is_streaming=False,
        device=device
    )
    assert len(dataset) == 1000 - model_args.max_seq_len, f"Expected length {1000 - model_args.max_seq_len}, got {len(dataset)}"
    x, y = dataset[0]
    assert x.shape == (model_args.max_seq_len,), f"Expected x shape ({model_args.max_seq_len},), got {x.shape}"
    assert y.shape == (model_args.max_seq_len,), f"Expected y shape ({model_args.max_seq_len},), got {y.shape}"
    assert x.dtype == torch.long, f"Expected x dtype long, got {x.dtype}"
    assert y.dtype == torch.long, f"Expected y dtype long, got {y.dtype}"

def test_tinystories_dataset_streaming(model_args, device):
    """Test TinyStoriesDataset in streaming mode."""
    token_ids = torch.randint(0, model_args.vocab_size, (1000,), dtype=torch.long)
    dataset = TinyStoriesDataset(
        token_ids=token_ids,
        seq_len=model_args.max_seq_len,
        is_streaming=True,
        device=device,
        prefetch_size=10
    )
    iterator = iter(dataset)
    x, y = next(iterator)
    assert x.shape == (model_args.max_seq_len,), f"Expected x shape ({model_args.max_seq_len},), got {x.shape}"
    assert y.shape == (model_args.max_seq_len,), f"Expected y shape ({model_args.max_seq_len},), got {y.shape}"
    assert x.dtype == torch.long, f"Expected x dtype long, got {x.dtype}"
    assert y.dtype == torch.long, f"Expected y dtype long, got {y.dtype}"

def test_dataloader_factory(model_args, training_config):
    """Test DataLoaderFactory with dummy token files."""
    # Use small dummy token files for testing
    train_tokens = torch.randint(0, model_args.vocab_size, (1000,), dtype=torch.long)
    valid_tokens = torch.randint(0, model_args.vocab_size, (200,), dtype=torch.long)
    torch.save(train_tokens, "test_train.pt")
    torch.save(valid_tokens, "test_valid.pt")

    # Create a dummy tokenizer file
    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast.from_pretrained("gpt2")
    tokenizer.save_pretrained("test_tokenizer")
    tokenizer_file = "test_tokenizer/tokenizer.json"

    dl_factory = DataLoaderFactory(
        model_args=model_args,
        cfg=training_config,
        train_token_file="test_train.pt",
        valid_token_file="test_valid.pt",
        tokenizer_file=tokenizer_file,
        pad_token="</s>"
    )
    train_loader, valid_loader = dl_factory.create_data_loaders()
    assert isinstance(train_loader, DataLoader), "Expected train_loader to be a DataLoader"
    assert isinstance(valid_loader, DataLoader), "Expected valid_loader to be a DataLoader"
    assert train_loader.batch_size == training_config.batch_size
    assert valid_loader.batch_size == training_config.batch_size