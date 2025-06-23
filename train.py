import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import json
from model import GPT, ctx_len
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table
import mlx.utils as utils
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import time

# Initialize Rich console
console = Console()

### High-speed hyper params
num_epochs = 20
batch_size = 128  # Reduced for faster iteration and better progress tracking
lr = 1e-3
accumulation_steps = 2  # Reduced for more frequent updates
NUM_WORKERS = mp.cpu_count() -1  #just in case you wanna do something with that core... 

# Training history for tracking
train_losses = []
val_losses = []
epochs = []

console.print(Panel.fit("[bold blue]GPT-2 Training with MLX (High-Speed)[/bold blue]", border_style="blue"))
console.print(f"[cyan]Using {NUM_WORKERS} CPU workers for data processing[/cyan]")

### Fast Data Loading and Preprocessing
console.print("[yellow]Loading and preprocessing data (high-speed)...[/yellow]")

def load_text_chunk(chunk):
    """Load and parse a chunk of text lines"""
    texts = []
    for line in chunk:
        obj = json.loads(line)
        texts.append(obj["text"])
    return texts

# Load data in chunks for parallel processing
with open("data/small-117M.train.jsonl", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Split lines into chunks for parallel processing
chunk_size = max(1, len(lines) // NUM_WORKERS)
chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]

# Process chunks in parallel
with Progress(console=console) as progress:
    task = progress.add_task("[cyan]Loading training file (parallel)", total=len(chunks))
    
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(load_text_chunk, chunk) for chunk in chunks]
        
        texts = []
        for future in futures:
            texts.extend(future.result())
            progress.advance(task)

text = "\n".join(texts)
console.print(f"[green]✓[/green] Loaded {len(texts)} text samples")

# Fast vocabulary creation
console.print("[yellow]Creating vocabulary...[/yellow]")
vocab = sorted(list(set(text)))
vocab_size = len(vocab)
itos = {i:c for i,c in enumerate(vocab)}
stoi = {c:i for i,c in enumerate(vocab)}

console.print(f"[green]✓[/green] Vocabulary size: [bold]{vocab_size}[/bold]")

# Fast data encoding with multiprocessing
console.print("[yellow]Encoding training data (parallel)...[/yellow]")

def encode_text_chunk(chunk_data):
    """Encode a chunk of text to tokens"""
    return [stoi[char] for char in chunk_data]

# Split text into chunks for parallel encoding
text_chunk_size = max(10000, len(text) // NUM_WORKERS)
text_chunks = [text[i:i + text_chunk_size] for i in range(0, len(text), text_chunk_size)]

with Progress(console=console) as progress:
    task = progress.add_task("[cyan]Encoding text to tokens (parallel)", total=len(text_chunks))
    
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(encode_text_chunk, chunk) for chunk in text_chunks]
        
        data = []
        for future in futures:
            data.extend(future.result())
            progress.advance(task)

split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]

console.print(f"[green]✓[/green] Training samples: [bold]{len(train_data)}[/bold]")
console.print(f"[green]✓[/green] Validation samples: [bold]{len(val_data)}[/bold]")

### Pre-process all data into MLX arrays (FAST)
console.print("[yellow]Pre-processing all data into MLX arrays...[/yellow]")

def create_sequences_fast(data, ctx_len):
    """Create all sequences efficiently with pre-allocated arrays"""
    # Use non-overlapping sequences for better memory efficiency
    total_sequences = len(data) // (ctx_len + 1)  # +1 for the target
    
    # Convert data to MLX array if it isn't already
    if not isinstance(data, mx.array):
        data = mx.array(data)
    
    # Pre-allocate final arrays - much more memory efficient!
    X = mx.zeros((total_sequences, ctx_len), dtype=data.dtype)
    y = mx.zeros((total_sequences, ctx_len), dtype=data.dtype)
    
    with Progress(console=console) as progress:
        task = progress.add_task("[cyan]Creating sequences", total=total_sequences)
        
        # Process in chunks to reduce memory pressure and enable better progress tracking
        chunk_size = min(2000, total_sequences)
        for chunk_start in range(0, total_sequences, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_sequences)
            
            # Vectorized approach for the chunk
            for i in range(chunk_start, chunk_end):
                start_pos = i * (ctx_len + 1)
                X[i] = data[start_pos:start_pos + ctx_len]
                y[i] = data[start_pos + 1:start_pos + ctx_len + 1]
            
            # Evaluate periodically to free temporary memory
            mx.eval(X[chunk_start:chunk_end], y[chunk_start:chunk_end])
            progress.update(task, completed=chunk_end)
    
    return X, y

# Create all training sequences
X_train, y_train = create_sequences_fast(train_data, ctx_len)
X_val, y_val = create_sequences_fast(val_data, ctx_len)

console.print(f"[green]✓[/green] Created {X_train.shape[0]:,} training sequences")
console.print(f"[green]✓[/green] Created {X_val.shape[0]:,} validation sequences")

console.print(f"[green]✓[/green] Training shape: {X_train.shape}")
console.print(f"[green]✓[/green] Validation shape: {X_val.shape}")

### Fast Batch Generator
def get_batches_fast(X, y, batch_size, shuffle=True):
    """Ultra-fast batch generator using only MLX arrays"""
    if shuffle:
        # Create shuffled indices using MLX
        indices = mx.arange(X.shape[0])
        indices = mx.random.permutation(indices)
        X = X[indices]
        y = y[indices]
    
    # Yield batches directly
    for i in range(0, X.shape[0], batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]

### Fast Training
def loss_fn(model, x, y):
    logits = model(x)
    B, T, C = logits.shape
    logits = logits.reshape(B*T, C)
    y = y.reshape(B*T)
    loss = nn.losses.cross_entropy(logits, y, reduction='mean')
    return loss

def accumulate_gradients(model, loss_and_grad, optimizer, batch_data, accumulation_steps, progress_callback=None):
    """Accumulate gradients over multiple batches with progress updates"""
    accumulated_grads = None
    total_loss = 0
    
    for i, (input, label) in enumerate(batch_data):
        loss, grads = loss_and_grad(model, input, label)
        total_loss += loss.item()
        
        # Scale gradients by accumulation steps
        scaled_grads = utils.tree_map(lambda g: g / accumulation_steps, grads)
        
        if accumulated_grads is None:
            accumulated_grads = scaled_grads
        else:
            accumulated_grads = utils.tree_map(lambda acc, new: acc + new, accumulated_grads, scaled_grads)
        
        # If we've reached accumulation_steps or this is the last batch, update the model
        if (i + 1) % accumulation_steps == 0 or i == len(batch_data) - 1:
            optimizer.update(model, accumulated_grads)
            mx.eval(model.parameters(), optimizer.state)
            accumulated_grads = None
        
        # Update progress bar
        if progress_callback:
            current_loss = total_loss / (i + 1)
            progress_callback(i + 1, current_loss)
    
    return total_loss / len(batch_data)

# Setup model and optimizer
console.print("[yellow]Initializing model and optimizer...[/yellow]")
model = GPT(vocab_size)
mx.eval(model.parameters())
loss_and_grad = nn.value_and_grad(model, loss_fn)
optimizer = optim.AdamW(learning_rate=lr)

# Count total parameters
total_params = sum([p.size for n, p in utils.tree_flatten(model.parameters())])
console.print(f"[green]✓[/green] Model parameters: [bold]{total_params:,}[/bold]")

# Calculate total batches
train_batches = (X_train.shape[0] + batch_size - 1) // batch_size
val_batches = (X_val.shape[0] + batch_size - 1) // batch_size

console.print(f"[cyan]Training batches per epoch:[/cyan] {train_batches}")
console.print(f"[cyan]Validation batches per epoch:[/cyan] {val_batches}")
console.print(f"[cyan]Batch size:[/cyan] {batch_size}")
console.print(f"[cyan]Gradient accumulation steps:[/cyan] {accumulation_steps}")
console.print(f"[cyan]Effective batch size:[/cyan] {batch_size * accumulation_steps}")
console.print(f"[cyan]Workers:[/cyan] {NUM_WORKERS}")

console.print("\n[bold green]Starting high-speed training...[/bold green]\n")

# Training loop with rich progress tracking
with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
    console=console
) as progress:
    
    epoch_task = progress.add_task("[cyan]Training Epochs", total=num_epochs)
    
    for epoch in range(num_epochs):
        model.train(True)
        running_loss = 0
        batch_cnt = 0
        epoch_start_time = time.time()
        
        # Training phase
        batch_task = progress.add_task(f"[yellow]Epoch {epoch+1}/{num_epochs} - Training", total=train_batches)
        
        # Collect all batches for this epoch
        all_batches = list(get_batches_fast(X_train, y_train, batch_size))
        
        # Define progress callback to update the progress bar in real-time
        def update_progress(batch_num, current_loss):
            progress.update(batch_task, completed=batch_num, 
                          description=f"[yellow]Epoch {epoch+1}/{num_epochs} - Training (Loss: {current_loss:.4f})")
        
        # Process batches with gradient accumulation and real-time progress
        avg_train_loss = accumulate_gradients(model, loss_and_grad, optimizer, all_batches, 
                                            accumulation_steps, progress_callback=update_progress)
        
        epoch_time = time.time() - epoch_start_time
        progress.remove_task(batch_task)
        
        # Validation phase
        model.train(False)
        running_loss = 0
        batch_cnt = 0
        
        val_task = progress.add_task(f"[green]Epoch {epoch+1}/{num_epochs} - Validation", total=val_batches)
        
        for input, label in get_batches_fast(X_val, y_val, batch_size, shuffle=False):
            batch_cnt += 1
            loss = loss_fn(model, input, label)
            running_loss += loss.item()
            progress.update(val_task, advance=1, description=f"[green]Epoch {epoch+1}/{num_epochs} - Validation (Loss: {loss.item():.4f})")
        
        avg_val_loss = running_loss / batch_cnt
        progress.remove_task(val_task)
        
        # Update epoch progress
        progress.update(epoch_task, advance=1)
        
        # Send data to plotting thread
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        epochs.append(epoch + 1)
        
        # Create rich table for epoch summary
        table = Table(title=f"Epoch {epoch+1}/{num_epochs} Summary")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        table.add_column("Status", style="green")
        
        train_status = "🟢" if avg_train_loss < 3.0 else "🟡" if avg_train_loss < 5.0 else "🔴"
        val_status = "🟢" if avg_val_loss < 3.0 else "🟡" if avg_val_loss < 5.0 else "🔴"
        
        table.add_row("Training Loss", f"{avg_train_loss:.4f}", train_status)
        table.add_row("Validation Loss", f"{avg_val_loss:.4f}", val_status)
        table.add_row("Loss Difference", f"{avg_train_loss - avg_val_loss:.4f}", 
                     "🟢" if abs(avg_train_loss - avg_val_loss) < 0.5 else "🟡")
        table.add_row("Epoch Time", f"{epoch_time:.1f}s", "🟢")
        table.add_row("Samples/sec", f"{X_train.shape[0]/epoch_time:.0f}", "🟢")
        table.add_row("Batches/sec", f"{train_batches/epoch_time:.1f}", "🟢")
        
        console.print(table)
        console.print("─" * 80)

console.print("\n[bold green]High-speed training completed![/bold green]")

### Save model and tokenizer
console.print("[yellow]Saving model and tokenizer...[/yellow]")
utils.save(model.state_dict(), "data/gpt2_mlx_model_fast.safetensors")

# Save tokenizer info
tokenizer_info = {
    "vocab_size": vocab_size,
    "itos": itos,
    "stoi": stoi
}
with open("data/tokenizer_fast.json", "w") as f:
    json.dump(tokenizer_info, f)

console.print("[green]✓[/green] Model saved as: [bold]gpt2_mlx_model_fast.safetensors[/bold]")
console.print("[green]✓[/green] Tokenizer saved as: [bold]data/tokenizer_fast.json[/bold]")

# Final training summary
final_table = Table(title="Final High-Speed Training Summary")
final_table.add_column("Metric", style="cyan")
final_table.add_column("Value", style="magenta")

final_table.add_row("Total Epochs", str(num_epochs))
final_table.add_row("Final Training Loss", f"{train_losses[-1]:.4f}")
final_table.add_row("Final Validation Loss", f"{val_losses[-1]:.4f}")
final_table.add_row("Best Training Loss", f"{min(train_losses):.4f}")
final_table.add_row("Best Validation Loss", f"{min(val_losses):.4f}")
final_table.add_row("Model Parameters", f"{total_params:,}")
final_table.add_row("Batch Size", str(batch_size))
final_table.add_row("Gradient Accumulation Steps", str(accumulation_steps))
final_table.add_row("Effective Batch Size", str(batch_size * accumulation_steps))
final_table.add_row("CPU Workers", str(NUM_WORKERS))

console.print(final_table) 