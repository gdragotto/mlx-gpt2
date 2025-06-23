import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import json
from model import GPT, ctx_len
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table
import mlx.utils as utils

# Initialize Rich console
console = Console()

### hyper params
# training
num_epochs=20
batch_size=64
lr = 1e-3

# Training history for tracking
train_losses = []
val_losses = []
epochs = []

console.print(Panel.fit("[bold blue]GPT-2 Training with MLX[/bold blue]", border_style="blue"))

### Tokenization
console.print("[yellow]Loading and tokenizing data...[/yellow]")
texts = []
with open("data/small-117M.train.jsonl", "r", encoding="utf-8") as f:
    # Add progress bar for file reading
    lines = f.readlines()
    with Progress(console=console) as progress:
        task = progress.add_task("[cyan]Reading training file", total=len(lines))
        for line in lines:
            obj = json.loads(line)
            texts.append(obj["text"])
            progress.advance(task)

text = "\n".join(texts)
console.print(f"[green]✓[/green] Loaded {len(texts)} text samples")

# Add progress for vocabulary creation
console.print("[yellow]Creating vocabulary...[/yellow]")
vocab = sorted(list(set(text)))
vocab_size = len(vocab)
itos = {i:c for i,c in enumerate(vocab)} # int to string
stoi = {c:i for i,c in enumerate(vocab)} # string to int
encode = lambda x: [stoi[c] for c in x]
decode = lambda x: ''.join([itos[i] for i in x])

console.print(f"[green]✓[/green] Vocabulary size: [bold]{vocab_size}[/bold]")

# Add progress for data encoding
console.print("[yellow]Encoding training data...[/yellow]")
with Progress(console=console) as progress:
    task = progress.add_task("[cyan]Encoding text to tokens", total=len(text))
    data = []
    for i, char in enumerate(text):
        data.append(stoi[char])
        if i % 100000 == 0:  # Update progress every 100k characters
            progress.update(task, completed=i)
    progress.update(task, completed=len(text))

split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]

console.print(f"[green]✓[/green] Training samples: [bold]{len(train_data)}[/bold]")
console.print(f"[green]✓[/green] Validation samples: [bold]{len(val_data)}[/bold]")

### Streaming Data Generator
class StreamingDataGenerator:
    def __init__(self, data, ctx_len, batch_size, shuffle=True):
        self.data = data
        self.ctx_len = ctx_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.total_sequences = (len(data) - ctx_len) // ctx_len
        
    def __iter__(self):
        # Create sequence indices
        indices = list(range(self.total_sequences))
        if self.shuffle:
            np.random.shuffle(indices)
        
        # Yield batches
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i+self.batch_size]
            batch_x = []
            batch_y = []
            
            for idx in batch_indices:
                start_pos = idx * self.ctx_len
                end_pos = start_pos + self.ctx_len
                batch_x.append(self.data[start_pos:end_pos])
                batch_y.append(self.data[start_pos+1:end_pos+1])
            
            # Convert to MLX arrays
            x = mx.array(batch_x)
            y = mx.array(batch_y)
            yield x, y

### Training
def loss_fn(model, x, y):
    logits = model(x)
    B, T, C = logits.shape
    logits = logits.reshape(B*T, C)
    y = y.reshape(B*T)
    loss = nn.losses.cross_entropy(logits, y, reduction='mean')
    return loss

# Setup model and optimizer
console.print("[yellow]Initializing model and optimizer...[/yellow]")
model = GPT(vocab_size)
mx.eval(model.parameters())
loss_and_grad = nn.value_and_grad(model, loss_fn)
optimizer = optim.AdamW(learning_rate=lr)

# Count total parameters
total_params = sum([p.size for n, p in utils.tree_flatten(model.parameters())])
console.print(f"[green]✓[/green] Model parameters: [bold]{total_params:,}[/bold]")

# Create data generators
train_generator = StreamingDataGenerator(train_data, ctx_len, batch_size, shuffle=True)
val_generator = StreamingDataGenerator(val_data, ctx_len, batch_size, shuffle=False)

# Calculate total batches
train_batches = (len(train_data) - ctx_len) // ctx_len // batch_size
val_batches = (len(val_data) - ctx_len) // ctx_len // batch_size

console.print(f"[cyan]Training batches per epoch:[/cyan] {train_batches}")
console.print(f"[cyan]Validation batches per epoch:[/cyan] {val_batches}")

console.print("\n[bold green]Starting training...[/bold green]\n")

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
        
        # Training phase
        batch_task = progress.add_task(f"[yellow]Epoch {epoch+1}/{num_epochs} - Training", total=train_batches)
        
        for input, label in train_generator:
            batch_cnt += 1
            loss, grads = loss_and_grad(model, input, label)
            optimizer.update(model, grads)
            running_loss += loss.item()
            mx.eval(model.parameters(), optimizer.state)
            
            # Update batch progress
            progress.update(batch_task, advance=1, description=f"[yellow]Epoch {epoch+1}/{num_epochs} - Training (Loss: {loss.item():.4f})")
        
        avg_train_loss = running_loss / batch_cnt
        progress.remove_task(batch_task)
        
        # Validation phase
        model.train(False)
        running_loss = 0
        batch_cnt = 0
        
        val_task = progress.add_task(f"[green]Epoch {epoch+1}/{num_epochs} - Validation", total=val_batches)
        
        for input, label in val_generator:
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
        
        console.print(table)
        console.print("─" * 80)

console.print("\n[bold green]Training completed![/bold green]")

### Save model and tokenizer
console.print("[yellow]Saving model and tokenizer...[/yellow]")
import mlx.utils as utils
utils.save(model.state_dict(), "data/gpt2_mlx_model.safetensors")

# Save tokenizer info
tokenizer_info = {
    "vocab_size": vocab_size,
    "itos": itos,
    "stoi": stoi
}
with open("data/tokenizer.json", "w") as f:
    json.dump(tokenizer_info, f)

console.print("[green]✓[/green] Model saved as: [bold]gpt2_mlx_model.safetensors[/bold]")
console.print("[green]✓[/green] Tokenizer saved as: [bold]data/tokenizer.json[/bold]")

# Final training summary
final_table = Table(title="Final Training Summary")
final_table.add_column("Metric", style="cyan")
final_table.add_column("Value", style="magenta")

final_table.add_row("Total Epochs", str(num_epochs))
final_table.add_row("Final Training Loss", f"{train_losses[-1]:.4f}")
final_table.add_row("Final Validation Loss", f"{val_losses[-1]:.4f}")
final_table.add_row("Best Training Loss", f"{min(train_losses):.4f}")
final_table.add_row("Best Validation Loss", f"{min(val_losses):.4f}")
final_table.add_row("Model Parameters", f"{total_params:,}")

console.print(final_table) 