"""
TinyAlphaZero Visual Dashboard
Interactively analyze positions and run benchmarks against specific model checkpoints.
"""

import sys
import glob
import torch
import gradio as gr
import chess
import chess.svg
import torch.nn.functional as F
from pathlib import Path

# ==============================================================================
# Setup & Imports
# ==============================================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from model.transformer import ChessTransformer
from data.encoding import encode_board, encode_move, decode_move
from benchmarks import ChessBenchmarkRunner, BENCHMARKS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# Model Container (Manages State)
# ==============================================================================

class ModelManager:
    def __init__(self):
        self.model = None
        self.current_checkpoint = "None (Random Initialization)"
        self.current_checkpoint_path = ""
        self.device = DEVICE
        
        # Default Config (will be overwritten by checkpoint if available)
        self.config = {
            "d_model": 256,
            "n_layers": 6,
            "n_heads": 8,
            "vocab_size": 23,
            "num_moves": 4096
        }
        
        # Initialize default random model
        self._build_model()

    def _build_model(self):
        print(f"Building model with config: {self.config}")
        self.model = ChessTransformer(
            vocab_size=self.config.get("vocab_size", 23),
            num_moves=self.config.get("num_moves", 4096),
            d_model=self.config.get("d_model", 256),
            n_layers=self.config.get("n_layers", 6),
            n_heads=self.config.get("n_heads", 8)
        ).to(self.device)
        self.model.eval()

    def load_checkpoint(self, path: str):
        if not path or not Path(path).exists():
            return f"Error: File not found - {path}"
        
        try:
            print(f"Loading checkpoint: {path}")
            checkpoint = torch.load(path, map_location=self.device)
            
            # 1. Update Config if present
            if 'config' in checkpoint:
                saved_config = checkpoint['config']
                # Basic compatibility check
                if saved_config.get('d_model') != self.config['d_model']:
                    print("Architecture change detected, rebuilding model...")
                    self.config.update(saved_config)
                    self._build_model()
            
            # 2. Load Weights
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            self.current_checkpoint = Path(path).name
            self.current_checkpoint_path = str(path)
            return f"Successfully loaded: {self.current_checkpoint}"
            
        except Exception as e:
            return f"Error loading checkpoint: {str(e)}"

    def predict(self, fen):
        board = chess.Board(fen)
        tokens = torch.tensor([encode_board(board)], device=self.device)
        
        with torch.no_grad():
            policy_logits, value_logits = self.model(tokens)
            
            # Policy
            policy_probs = F.softmax(policy_logits[0], dim=-1)
            
            # Value (3-class: Loss, Draw, Win)
            value_probs = F.softmax(value_logits[0], dim=-1)
            value = value_probs[2] - value_probs[0] # P(Win) - P(Loss)
            
        return policy_probs, value.item()

    def fallback_to_cpu(self, err: RuntimeError):
        if self.device.type == "cpu":
            raise err
        print(f"Device error on {self.device}, retrying on CPU: {err}")
        self.device = torch.device("cpu")
        self._build_model()
        if self.current_checkpoint_path:
            try:
                checkpoint = torch.load(self.current_checkpoint_path, map_location=self.device)
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                self.model.load_state_dict(state_dict)
                self.model.eval()
            except Exception as reload_err:
                print(f"Failed to reload checkpoint on CPU: {reload_err}")

# Initialize Global Manager
manager = ModelManager()

# ==============================================================================
# Helper Functions
# ==============================================================================

def get_available_checkpoints():
    """Finds all .pt files in checkpoints directory recursively."""
    files = glob.glob(str(ROOT / "checkpoints/**/*.pt"), recursive=True)
    return sorted(files, key=lambda t: -1 * Path(t).stat().st_mtime) # Sort by newest

def analyze_position(fen):
    if not fen:
        return None, None, "Please enter a FEN string"
        
    try:
        board = chess.Board(fen)
    except ValueError:
        return None, None, "Invalid FEN string"

    # 1. Render Board
    board_svg = chess.svg.board(board=board, size=400)
    
    # 2. Get Predictions
    try:
        policy_probs, value = manager.predict(fen)
    except RuntimeError as err:
        manager.fallback_to_cpu(err)
        policy_probs, value = manager.predict(fen)
    
    # 3. Format Top Moves
    top_k = 10
    top_indices = torch.topk(policy_probs, top_k).indices.tolist()
    
    move_data = []
    for idx in top_indices:
        try:
            move = decode_move(idx, board) # Decode with board context for promotions
            uci = move.uci()
            prob = policy_probs[idx].item()
            is_legal = move in board.legal_moves
            
            # Fancy formatting
            prob_str = f"{prob:.4f}"
            note = "" if is_legal else " (ILLEGAL)"
            
            move_data.append([uci + note, prob])
        except Exception:
            move_data.append([f"Idx {idx} (Decode Err)", policy_probs[idx].item()])

    value_str = f"{value:.4f} (Win Prob - Loss Prob)"
    
    return board_svg, move_data, value_str

def run_benchmarks_ui():
    runner = ChessBenchmarkRunner(manager.model, device=manager.device)
    try:
        report = runner.run_all()
    except RuntimeError as err:
        manager.fallback_to_cpu(err)
        runner = ChessBenchmarkRunner(manager.model, device=manager.device)
        report = runner.run_all()
    
    # Convert report to Markdown
    md = f"## Benchmark Results ({manager.current_checkpoint})\n\n"
    
    # Summary Table
    md += "| Category | Passed | Failed |\n|---|---|---|\n"
    summary = report['summary']
    md += f"| **TOTAL** | **{summary['passed']}** | **{summary['failed']}** |\n"
    
    for cat_key, cat_data in report['categories'].items():
        passed = sum(1 for t in cat_data['tests'] if t['passed'])
        failed = sum(1 for t in cat_data['tests'] if not t['passed'])
        md += f"| {cat_data['name']} | {passed} | {failed} |\n"
        
    md += "\n---\n"
    
    # Details
    for cat_key, cat_data in report['categories'].items():
        md += f"### {cat_data['name']}\n"
        for test in cat_data['tests']:
            icon = "✅" if test['passed'] else "❌"
            md += f"**{icon} {test['name']}**\n"
            md += f"*Input:* `{test['fen']}`\n"
            for check in test['checks']:
                check_icon = "ok" if check['passed'] else "FAIL"
                md += f"- [{check_icon}] {check['detail']}\n"
            md += "\n"
            
    return md

def refresh_dropdown():
    ckpts = get_available_checkpoints()
    return gr.Dropdown.update(choices=ckpts, value=ckpts[0] if ckpts else None)

# ==============================================================================
# Gradio Interface
# ==============================================================================

with gr.Blocks(title="TinyAlphaZero Dashboard", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ♟️ TinyAlphaZero Dashboard")
    
    # --- Model Selection Header ---
    with gr.Row(variant="panel"):
        with gr.Column(scale=3):
            ckpt_dropdown = gr.Dropdown(
                label="Select Checkpoint", 
                choices=get_available_checkpoints(),
                interactive=True
            )
        with gr.Column(scale=1):
            load_btn = gr.Button("Load Checkpoint", variant="primary")
            refresh_btn = gr.Button("🔄 Refresh List")
        with gr.Column(scale=2):
            status_text = gr.Textbox(label="Status", value=f"Active: {manager.current_checkpoint}", interactive=False)

    # Wire up loading logic
    load_btn.click(
        fn=manager.load_checkpoint,
        inputs=[ckpt_dropdown],
        outputs=[status_text]
    )
    refresh_btn.click(
        fn=refresh_dropdown,
        inputs=[],
        outputs=[ckpt_dropdown]
    )

    # --- Main Content ---
    with gr.Tabs():
        # Tab 1: Analyzer
        with gr.Tab("Position Analyzer"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_fen = gr.Textbox(label="FEN String", value=chess.STARTING_FEN)
                    analyze_btn = gr.Button("Analyze Position", variant="primary")
                    
                    gr.Markdown("### Examples")
                    gr.Examples(
                        examples=[
                            chess.STARTING_FEN,
                            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQK2R w KQkq - 0 1", # Castling rights
                            "4k3/8/8/8/8/8/8/4R1K1 w - - 0 1", # Mate in 1
                            "rnb1kbnr/pppp1ppp/8/4p3/6P4/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2" # Fool's mate pattern
                        ],
                        inputs=input_fen
                    )
                    
                with gr.Column(scale=1):
                    output_board = gr.HTML(label="Board Visualization")
                    output_value = gr.Textbox(label="Value Head Evaluation")
                    output_moves = gr.Dataframe(headers=["Move (UCI)", "Probability"], label="Top Policy Predictions")
            
            analyze_btn.click(
                fn=analyze_position, 
                inputs=[input_fen], 
                outputs=[output_board, output_moves, output_value]
            )

        # Tab 2: Benchmarks
        with gr.Tab("Structural Benchmarks"):
            gr.Markdown("Run the standard test suite against the currently loaded model.")
            run_bench_btn = gr.Button("Run Benchmark Suite", variant="primary")
            output_report = gr.Markdown()
            
            run_bench_btn.click(fn=run_benchmarks_ui, inputs=[], outputs=[output_report])

if __name__ == "__main__":
    demo.launch()
