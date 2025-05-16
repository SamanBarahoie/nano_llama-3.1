import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List, Tuple
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer  # Assuming Hugging Face tokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)

@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""
    batch_size: int = 2
    seq_len: int = 32
    epochs: int = 10
    steps_per_epoch: int = 1000
    report_interval: int = 100  # Tokens before reporting loss
    grad_clip_norm: float = 1.0
    learning_rate: float = 3e-4
    warmup_steps: int = 100
    max_lr: float = 3e-4
    min_lr: float = 3e-5

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {self.seq_len}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.steps_per_epoch <= 0:
            raise ValueError(f"steps_per_epoch must be positive, got {self.steps_per_epoch}")
        if self.report_interval <= 0:
            raise ValueError(f"report_interval must be positive, got {self.report_interval}")
        if self.grad_clip_norm <= 0:
            raise ValueError(f"grad_clip_norm must be positive, got {self.grad_clip_norm}")

class Trainer:
    """A class to handle training, evaluation, and text generation for a Transformer model."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        valid_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        scaler: GradScaler,
        tokenizer: AutoTokenizer,
        device: str,
        cfg: TrainingConfig
    ):
        """Initialize the Trainer with model and training components.
        
        Args:
            model (nn.Module): The Transformer model to train.
            train_loader (DataLoader): DataLoader for training data.
            valid_loader (DataLoader): DataLoader for validation data.
            optimizer (Optimizer): Optimizer for training.
            scheduler (LRScheduler): Learning rate scheduler.
            scaler (GradScaler): Gradient scaler for mixed precision.
            tokenizer (AutoTokenizer): Tokenizer for text generation.
            device (str): Device to run the model on (e.g., 'cuda').
            cfg (TrainingConfig): Training configuration.
        """
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.tokenizer = tokenizer
        self.device = device
        self.cfg = cfg
        
        self.model.to(self.device)
        logging.info("Trainer initialized with model on %s", self.device)


    def train_epoch(self) -> List[float]:
        """Run a single training epoch.

        Returns:
            List[float]: List of average losses per reporting interval.
        """
        self.model.train()
        total_loss = 0.0
        total_tokens = 0
        tokens_processed = 0
        loss_accum = 0.0
        accum_count = 0
        loss_per_interval = []
        invalid_batches = 0

        iterator = iter(self.train_loader)
        progress_bar = tqdm(
            range(self.cfg.steps_per_epoch),
            desc="Training",
            dynamic_ncols=True
        )

        for step in progress_bar:
            try:
                x, y = next(iterator)
            except StopIteration:
                iterator = iter(self.train_loader)
                x, y = next(iterator)

            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            # Sanity check on token IDs
            vs = getattr(self.model, "args", None)
            vocab_size = vs.vocab_size if vs else 10000
            if (x < 0).any() or (y < 0).any() or (x >= vocab_size).any() or (y >= vocab_size).any():
                logging.error("Invalid token IDs at step %d", step)
                invalid_batches += 1
                continue

            # Zero grads
            self.optimizer.zero_grad(set_to_none=True)

            # --- Mixed precision on CUDA ---
            if self.device == "cuda":
                with autocast(dtype=torch.float16):
                    logits, _ = self.model(x, start_pos=-1)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        y.view(-1),
                        reduction="mean",
                        ignore_index=-1
                    )
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.error("Invalid loss at step %d: %s", step, loss.item())
                    invalid_batches += 1
                    continue

                # Backward pass (FP16 scaled)
                self.scaler.scale(loss).backward()

                # Only unscale & step if we actually have gradients
                if any(p.grad is not None for p in self.model.parameters()):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.cfg.grad_clip_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                else:
                    logging.warning("No FP16 grads at step %d, skipping optimizer step.", step)
                    invalid_batches += 1
                    continue

            # --- Standard FP32 path (CPU/MPS) ---
            else:
                logits, _ = self.model(x, start_pos=-1)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    reduction="mean",
                    ignore_index=-1
                )
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.error("Invalid loss at step %d: %s", step, loss.item())
                    invalid_batches += 1
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.cfg.grad_clip_norm
                )
                self.optimizer.step()
                self.scheduler.step()

            # Accumulate statistics
            loss_val = loss.item()
            total_loss += loss_val
            loss_accum += loss_val
            accum_count += 1
            processed = self.cfg.batch_size * self.cfg.seq_len
            tokens_processed += processed
            total_tokens += processed

            # Reporting
            if tokens_processed >= self.cfg.report_interval:
                avg_loss = loss_accum / accum_count
                loss_per_interval.append(avg_loss)
                ppl = torch.exp(torch.tensor(avg_loss).clamp(max=100)).item()
                progress_bar.set_postfix(loss=f"{avg_loss:.4f}", ppl=f"{ppl:.2f}")
                logging.info(
                    "Step %d: avg_loss=%.4f, ppl=%.2f, tokens=%d, invalid=%d",
                    step, avg_loss, ppl, tokens_processed, invalid_batches
                )
                loss_accum, accum_count, tokens_processed = 0.0, 0, 0

        # Final report if any remaining
        if accum_count > 0:
            avg_loss = loss_accum / accum_count
            loss_per_interval.append(avg_loss)
            logging.info(
                "End of epoch: last interval tokens=%d, avg_loss=%.4f, invalid=%d",
                tokens_processed, avg_loss, invalid_batches
            )

        avg_epoch_loss = total_loss / self.cfg.steps_per_epoch
        logging.info(
            "Epoch done: total_tokens=%d, avg_loss=%.4f, invalid_batches=%d",
            total_tokens, avg_epoch_loss, invalid_batches
        )
        return loss_per_interval



    def eval_epoch(self) -> Tuple[float, float]:
        """Run a single evaluation epoch.
        
        Returns:
            Tuple[float, float]: Average loss and perplexity.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.valid_loader)
        
        with torch.no_grad():
            for x, y in tqdm(self.valid_loader, desc="Evaluating", dynamic_ncols=True):
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                with autocast():
                    logits, _ = self.model(x, start_pos=-1)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        y.view(-1),
                        reduction="mean"
                    )
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        logging.info("Evaluation completed: avg_loss=%.4f, perplexity=%.2f", avg_loss, perplexity)
        return avg_loss, perplexity

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int,
        top_k: int = 50,
        temperature: float = 1.0
    ) -> str:
        """Generate text using the model.
        
        Args:
            prompt (str): Input prompt to start generation.
            max_new_tokens (int): Maximum number of new tokens to generate.
            top_k (int): Number of top logits to sample from. Defaults to 50.
            temperature (float): Temperature for softmax sampling. Defaults to 1.0.
        
        Returns:
            str: Generated text.
        """
        self.model.eval()
        with torch.no_grad():
            tokens = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            prompt_len = tokens.size(1)
            
            if prompt_len > self.cfg.seq_len:
                tokens = tokens[:, -self.cfg.seq_len:]
                prompt_len = self.cfg.seq_len
            
            seq = tokens
            for _ in range(max_new_tokens):
                context = seq[:, -self.cfg.seq_len:]
                logits, _ = self.model(context, start_pos=seq.size(1) - context.size(1))
                logits = logits[:, -1, :] / temperature
                top_k_logits, top_k_indices = torch.topk(logits, top_k)
                probs = F.softmax(top_k_logits, dim=-1)
                next_tok_idx = torch.multinomial(probs, num_samples=1)
                next_tok = top_k_indices.gather(-1, next_tok_idx)
                seq = torch.cat([seq, next_tok], dim=1)
            
            generated_text = self.tokenizer.decode(seq[0].tolist())
        return generated_text

    def train(self, prompts: Optional[List[str]] = None) -> None:
        """Run the full training loop with evaluation and text generation.
        
        Args:
            prompts (Optional[List[str]]): List of prompts for text generation after each epoch.
        """
        prompts = prompts or [
            "Hi Jane, have you seen Alice? I can’t find her anywhere,” said Jack.",
            "Max had two dogs. One was white and the other was black. Max walked up the street and saw a kid with a dog. He told the kid, ”I see you have a Brown dog. I also have",
            "Anne had a piece of candy in her left pocket and a piece of chocolate in her right pocket. Anne’s mom asked her, ”Anne, what is that you have in your left pocket?”"
        ]
        
        for epoch in range(1, self.cfg.epochs + 1):
            logging.info("=== Epoch %d/%d ===", epoch, self.cfg.epochs)
            start_time = time.time()
            
            # Train
            loss_per_interval = self.train_epoch()
            
            # Generate samples
            for prompt in prompts:
                generated_text = self.generate_text(
                    prompt=prompt,
                    max_new_tokens=64,
                    top_k=50,
                    temperature=0.8
                )
                logging.info("Generated sample: %s", generated_text)
            
            # Evaluate
            val_loss, val_ppl = self.eval_epoch()
            
            # Log epoch summary
            epoch_duration = time.time() - start_time
            duration_str = time.strftime("%H:%M:%S", time.gmtime(epoch_duration))
            logging.info(
                "Epoch %d completed in %s: val_loss=%.4f, val_ppl=%.2f",
                epoch, duration_str, val_loss, val_ppl
            )