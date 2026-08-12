"""Pullbot inference - loads model and generates responses"""
import torch, json, os, sys, time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from model import Pullbot1B, Tokenizer

def load_model():
    checkpoint = torch.load(os.path.join(REPO_ROOT, 'models', 'pullbot_1b.pt'), map_location='cpu')
    config = checkpoint['config']
    
    tokenizer = Tokenizer()
    tokenizer.word_to_id = checkpoint['tokenizer']
    tokenizer.id_to_word = {v: k for k, v in tokenizer.word_to_id.items()}
    
    model = Pullbot1B(
        vocab_size=config['vocab_size'],
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads']
    )
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")
    return model, tokenizer

def respond(question):
    model, tokenizer = load_model()
    tokens = tokenizer.encode(question)
    input_ids = torch.tensor([tokens])
    
    start = time.time()
    response = model.generate(input_ids, tokenizer, max_len=100, temperature=0.8)
    elapsed = time.time() - start
    
    return {
        'question': question,
        'response': response,
        'time': elapsed
    }

if __name__ == '__main__':
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "What is Python?"
    result = respond(q)
    print(f"Q: {result['question']}")
    print(f"A: {result['response']}")
    print(f"Time: {result['time']:.1f}s")
