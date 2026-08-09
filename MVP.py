import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------------------------------------------
# 1. Target Vector Attention & Decomposer (동일 모듈)
# ---------------------------------------------------------------
class SingleStepAttention(nn.Module):
    def __init__(self, embed_dim: int, target_dim: int):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, target_dim)
        self.k_proj = nn.Linear(embed_dim, target_dim)
        self.v_proj = nn.Linear(embed_dim, target_dim)
        self.scale = target_dim**-0.5

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        if query.dim() == 2:
            query = query.unsqueeze(1)
        Q = self.q_proj(query)
        K = self.k_proj(key_value)
        V = self.v_proj(key_value)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        target_x = torch.matmul(attn_weights, V).squeeze(1)
        return target_x


class VectorDecomposer(nn.Module):
    def __init__(self, N: int, dim: int, alpha: float = 1.0):
        super().__init__()
        self.N = N
        self.dim = dim
        self.alpha = alpha
        self.w = nn.Parameter(torch.randn(N, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_bar = torch.mean(self.w, dim=0, keepdim=True)
        w_diff = self.w - w_bar
        std_w = torch.std(self.w, correction=0) + 1e-8

        x_term = x.unsqueeze(1) / self.N
        w_term = self.alpha * (w_diff / std_w).unsqueeze(0)
        return x_term + w_term


# ---------------------------------------------------------------
# 2. Non-Autoregressive One-Shot AI Pipeline
# ---------------------------------------------------------------
class OneShotParallelDecomposedLLM(nn.Module):
    def __init__(self, model_name: str = "gpt2", N_tokens: int = 4, alpha: float = 1.0):
        super().__init__()
        self.N = N_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_llm = AutoModelForCausalLM.from_pretrained(model_name)

        self.vocab_size = base_llm.config.vocab_size
        self.embed_dim = base_llm.config.hidden_size

        # 사전 학습된 Embedding 및 LM Head 가져오기
        self.embedding = base_llm.get_input_embeddings()
        self.lm_head = base_llm.get_output_embeddings()

        # 학습을 위해 사전 학습된 가중치는 동결(Freeze)하고 새로 추가한 모듈만 학습시킵니다.
        for param in self.embedding.parameters():
            param.requires_grad = False
        for param in self.lm_head.parameters():
            param.requires_grad = False

        self.attention_target = SingleStepAttention(self.embed_dim, self.embed_dim)
        self.decomposer = VectorDecomposer(N=N_tokens, dim=self.embed_dim, alpha=alpha)

    def forward(self, prompt_ids: torch.Tensor):
        """
        prompt_ids: (batch_size, seq_len) -> 입력 프롬프트
        returns: logits (batch_size, N, vocab_size) -> 한번에 예측된 N개 토큰의 Logit
        """
        embeddings = self.embedding(prompt_ids)
        query = embeddings[:, -1:, :]  # 프롬프트의 마지막 토큰을 Query로 사용

        # 1) Attention 1회 적용 -> Target Vector x 단 한번 생성
        target_x = self.attention_target(query=query, key_value=embeddings)

        # 2) Attention 없이 수식으로 N개 벡터 v_1 ~ v_N 동시 분해
        v = self.decomposer(target_x)  # (batch_size, N, embed_dim)

        # 3) N개의 벡터를 동시에 LM Head로 통과시켜 한번에 예측
        logits = self.lm_head(v)  # (batch_size, N, vocab_size)

        return logits


# ---------------------------------------------------------------
# 3. 실제 원샷(One-Shot) 학습 루프 실행
# ---------------------------------------------------------------
if __name__ == "__main__":
    # N=4: 한 번의 Target Vector 생성으로 4개의 토큰을 동시에 출력하도록 설정
    N_TOKENS = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = OneShotParallelDecomposedLLM(
        model_name="gpt2", N_tokens=N_TOKENS, alpha=1.0
    ).to(device)

    # 미세 조정할 파라미터 (Attention 모듈과 Decomposer w 파라미터)
    trainable_params = list(model.attention_target.parameters()) + list(
        model.decomposer.parameters()
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # 학습용 미니 데이터셋 (프롬프트 -> 타겟 N개 연속 단어)
    training_data = [
        ("Artificial Intelligence is", " getting smarter every day."),
        ("Deep learning models can", " solve complex human problems."),
        ("Python is a popular", " programming language for AI."),
        ("Natural language processing allows", " computers to understand text."),
    ]

    print("=== 원샷(One-Shot) 타겟 분해 모델 학습 시작 ===")
    model.train()

    epochs = 100
    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        for prompt_text, target_text in training_data:
            # 1) 프롬프트 토큰화
            prompt_ids = model.tokenizer.encode(prompt_text, return_tensors="pt").to(
                device
            )

            # 2) 정답 target_text를 정확히 N_TOKENS 크기로 토큰화 (Parallel Ground Truth)
            target_ids = model.tokenizer.encode(target_text, return_tensors="pt")[
                :, :N_TOKENS
            ].to(device)

            optimizer.zero_grad()

            # 3) Forward: 한 번에 N개 토큰에 대한 logits 예측 (batch_size, N, vocab_size)
            logits = model(prompt_ids)

            # 4) N개 토큰 전체 손실을 한 번에 계산 (Parallel Cross-Entropy)
            loss = loss_fn(logits.view(-1, model.vocab_size), target_ids.view(-1))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{epochs} | Loss: {total_loss / len(training_data):.4f}"
            )

    # ---------------------------------------------------------------
    # 4. 학습 후 추론(Inference) 테스트
    # ---------------------------------------------------------------
    print("\n=== 학습 완료 후 원샷 생성 테스트 ===")
    model.eval()
    with torch.no_grad():
        test_prompt = "Artificial Intelligence is"
        test_ids = model.tokenizer.encode(test_prompt, return_tensors="pt").to(device)

        # 단 한 번의 Forward Pass로 N개 토큰 생성
        logits = model(test_ids)
        pred_ids = torch.argmax(logits, dim=-1).squeeze(0)  # (N,)

        gen_text = model.tokenizer.decode(pred_ids, skip_special_tokens=True)
        print(f"입력 프롬프트: '{test_prompt}'")
        print(f"원샷 동시 생성 결과 ({N_TOKENS}개 토큰): '{gen_text}'")
