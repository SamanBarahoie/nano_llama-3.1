import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from trainer import TrainingConfig, Trainer
from model import ModelArgs, llama
from dataloader import DataLoaderFactory
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("main.log"),
        logging.StreamHandler()
    ]
)

def main():
    """Main function to set up and run the training process."""
    # Model configuration (LLaMA 3.1 8B-like, adjusted for TinyStories)
    model_args = ModelArgs(
        dim=1024,
        n_heads=8,
        n_kv_heads=2,
        intermediate_dim=1536,
        multiple_of=256,
        ffn_dim_multiplier=1.0,
        norm_eps=1e-5,
        max_batch_size=128,
        max_seq_len=128,  # Match TinyStories seq_len
        vocab_size=10000,  # Match TinyStories vocab
        n_layers=4,
        rope_theta=10000.0,
        use_scaled_rope=False,
        flash=True
    )
    
    # Training configuration
    cfg = TrainingConfig(
        batch_size=128,
        seq_len=128,
        epochs=2,
        steps_per_epoch=15000,
        report_interval=20_000_000,
        grad_clip_norm=1.0,
        learning_rate=6e-4,
        warmup_steps=10,
        max_lr=6e-4,
        min_lr=3e-5
    )
    
    # Initialize model
    model = llama(model_args).to('cuda')
    logging.info("Model initialized with %d parameters", sum(p.numel() for p in model.parameters()))
    
    # Create data loaders with DataLoaderFactory
    try:
        dl_factory = DataLoaderFactory(
            model_args=model_args,
            cfg=cfg,
            train_token_file='tokenized-train-samples_vocab-10k.pt',
            valid_token_file='tokenized-valid-samples_vocab-10k.pt',
            tokenizer_file='bpe-tokenizer_tinystories.json',
            pad_token='</s>'
        )
        train_loader, valid_loader = dl_factory.create_data_loaders()
        tokenizer = dl_factory.tokenizer
    except Exception as e:
        logging.error("Failed to create data loaders: %s", e)
        raise
    
    logging.info("Data loaders created: %d validation batches, training in streaming mode",
                 len(valid_loader))
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs * cfg.steps_per_epoch,
        eta_min=cfg.min_lr
    )
    scaler = torch.cuda.amp.GradScaler()
    
    # Fallback tokenizer if TinyStories tokenizer fails
    if tokenizer is None:
        logging.warning("Using fallback tokenizer (GPT-2)")
        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        except Exception as e:
            logging.error("Failed to load fallback tokenizer: %s", e)
            raise
    
    # Define prompts for text generation
    prompts = [
        "Hi Jane, have you seen Alice? ",
        "Max had two dogs",
        "Once upon a time"
    ]
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        tokenizer=tokenizer,
        device='cuda',
        cfg=cfg
    )
    logging.info("Trainer initialized")
    
    # Run training
    try:
        trainer.train(prompts=prompts)
    except Exception as e:
        logging.error("Training failed: %s", e)
        raise
    
    # Save model
    torch.save(model.state_dict(), "model_final.pth")
    logging.info("Model saved to model_final.pth")

if __name__ == "__main__":
    main()