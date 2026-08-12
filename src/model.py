"""
Pullbot 1B Model - AI Trained From Scratch
Starts with 1 billion random parameters.
AI trainer (free GPT-4o) teaches it using wordbank as reference.
Your optimization methods refine every weight.
"""

import torch
import torch.nn as nn
import json, os, time, random, requests, math

# ============================================
# 1B PARAMETER MODEL
# ============================================

class Pullbot1B(nn.Module):
    def __init__(self, vocab_size=50000, embed_dim=1024, hidden_dim=4096, num_layers=24, num_heads=16):
        super().__init__()
        
        # Token embeddings
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(2048, embed_dim)
        
        # Transformer blocks (24 layers for ~1B params)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Output
        self.ln_final = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Initialize random weights
        self._init_weights()
        
    def _init_weights(self):
        """Random initialization - the AI will teach these values"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids):
        batch, seq = input_ids.shape
        
        # Position embeddings
        positions = torch.arange(seq, device=input_ids.device).unsqueeze(0)
        x = self.embedding(input_ids) + self.pos_embedding(positions)
        
        # Through all transformer layers
        for layer in self.layers:
            x = layer(x)
        
        x = self.ln_final(x)
        logits = self.output(x)
        
        return logits
    
    def generate(self, input_ids, tokenizer, max_len=100, temperature=0.8):
        """Generate text from the model"""
        self.eval()
        generated = list(input_ids[0].tolist())
        
        with torch.no_grad():
            for _ in range(max_len):
                # Get last 512 tokens
                context = generated[-512:]
                context_tensor = torch.tensor([context])
                
                # Forward pass
                logits = self(context_tensor)
                next_logits = logits[0, -1, :] / temperature
                
                # Sample
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                generated.append(next_token)
                
                # Stop token
                if next_token == tokenizer.word_to_id.get('<end>', 3):
                    break
        
        return tokenizer.decode(generated)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
    
    def forward(self, x):
        # Attention with residual
        attn_out, _ = self.attention(x, x, x, need_weights=False)
        x = self.ln1(x + attn_out)
        
        # MLP with residual
        mlp_out = self.mlp(x)
        x = self.ln2(x + mlp_out)
        
        return x


# ============================================
# AI TRAINER (Free GPT-4o via Puter.js)
# ============================================

class AITrainer:
    """Calls free AI to generate training data and grade responses"""
    
    def __init__(self):
        self.api_url = "https://api.puter.com/ai/chat"
    
    def generate_qa(self, wordbank_entry, num_pairs=5):
        """Generate Q&A pairs from wordbank definitions"""
        word = wordbank_entry.get('word', '')
        definition = wordbank_entry.get('definition', '')
        
        prompt = f"""Using this definition: "{word}: {definition}"
Generate {num_pairs} question-answer pairs that test understanding.
Return as JSON: [{{"question": "...", "answer": "..."}}]"""
        
        try:
            r = requests.post(self.api_url, json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7
            }, timeout=60)
            
            if r.status_code == 200:
                text = r.json()["choices"][0]["message"]["content"]
                # Extract JSON
                import re
                match = re.search(r'\[.*\]', text, re.DOTALL)
                if match:
                    return json.loads(match.group())
        except:
            pass
        return []
    
    def grade_response(self, question, answer):
        """Grade Pullbot's response"""
        prompt = f"""Grade this AI response from 1-5:
Q: {question}
A: {answer}
Return: {{"score": 1-5, "correction": "better version if score < 4"}}"""
        
        try:
            r = requests.post(self.api_url, json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3
            }, timeout=30)
            
            if r.status_code == 200:
                text = r.json()["choices"][0]["message"]["content"]
                import re
                match = re.search(r'\{.*\}', text, re.DOTALL)
                if match:
                    return json.loads(match.group())
        except:
            pass
        return {"score": 0, "correction": ""}


# ============================================
# TOKENIZER
# ============================================

class Tokenizer:
    def __init__(self):
        self.word_to_id = {'<pad>': 0, '<unk>': 1, '<start>': 2, '<end>': 3}
        self.id_to_word = {0: '<pad>', 1: '<unk>', 2: '<start>', 3: '<end>'}
    
    def add_word(self, word):
        if word not in self.word_to_id:
            idx = len(self.word_to_id)
            self.word_to_id[word] = idx
            self.id_to_word[idx] = word
    
    def build_from_wordbank(self, wordbank_path='data/wordbank.json'):
        """Build vocabulary from wordbank"""
        if not os.path.exists(wordbank_path):
            return
        
        with open(wordbank_path) as f:
            bank = json.load(f)
        
        for word in bank.get('words', {}):
            self.add_word(word)
            info = bank['words'][word]
            if isinstance(info, dict):
                definition = info.get('definition', '')
                for w in definition.lower().split():
                    self.add_word(w)
        
        print(f"Vocabulary: {len(self.word_to_id)} tokens")
    
    def encode(self, text, max_len=512):
        words = text.lower().split()[:max_len]
        tokens = [self.word_to_id.get(w, 1) for w in words]
        tokens = [2] + tokens + [3]
        while len(tokens) < max_len:
            tokens.append(0)
        return tokens[:max_len]
    
    def decode(self, tokens):
        words = []
        for t in tokens:
            if t in [0, 2, 3]:
                continue
            word = self.id_to_word.get(t, '')
            if word:
                words.append(word)
        return ' '.join(words)


# ============================================
# TRAINING LOOP
# ============================================

def train_pullbot():
    print("=" * 50)
    print("🧠 TRAINING PULLBOT 1B (AI-Taught)")
    print("=" * 50)
    
    # Build tokenizer from wordbank
    tokenizer = Tokenizer()
    tokenizer.build_from_wordbank()
    
    # Create model (1B params, random start)
    model = Pullbot1B(
        vocab_size=len(tokenizer.word_to_id),
        embed_dim=1024,
        hidden_dim=4096,
        num_layers=24,
        num_heads=16
    )
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {params:,} parameters ({params*4/(1024**3):.1f}GB)")
    
    # AI Trainer
    ai = AITrainer()
    
    # Load wordbank
    with open('data/wordbank.json') as f:
        bank = json.load(f)
    
    # Get defined words
    defined = []
    for word, info in bank.get('words', {}).items():
        if isinstance(info, dict) and info.get('has_definition'):
            defined.append({'word': word, 'definition': info['definition']})
    
    print(f"Training on {len(defined)} defined words")
    
    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    model.train()
    
    for epoch in range(3):  # 3 epochs
        print(f"\n=== Epoch {epoch+1} ===")
        random.shuffle(defined)
        total_loss = 0
        
        for i, entry in enumerate(defined[:200]):  # 200 words per epoch
            # AI generates Q&A for this word
            qa_pairs = ai.generate_qa(entry)
            
            for qa in qa_pairs[:3]:  # Use top 3 pairs
                q_tokens = tokenizer.encode(qa['question'])
                a_tokens = tokenizer.encode(qa['answer'])
                
                input_ids = torch.tensor([q_tokens + a_tokens])
                
                # Forward
                logits = model(input_ids)
                target = input_ids[:, 1:]
                logits = logits[:, :-1, :]
                
                loss = criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            if (i+1) % 20 == 0:
                print(f"   {i+1}/{len(defined[:200])} - loss: {total_loss/(i+1):.4f}")
            
            time.sleep(0.5)  # Rate limit for AI calls
    
    # Save
    os.makedirs('models', exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'tokenizer': tokenizer.word_to_id,
        'config': {
            'vocab_size': len(tokenizer.word_to_id),
            'embed_dim': 1024,
            'hidden_dim': 4096,
            'num_layers': 24,
            'num_heads': 16
        }
    }, 'models/pullbot_1b.pt')
    
    print(f"\n✅ Model saved!")

if __name__ == '__main__':
    train_pullbot()
