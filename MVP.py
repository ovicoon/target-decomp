import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


# ===============================================================
# 1. Target Vector 생성 모듈 (Attention 단 1회 사용)
# ===============================================================
class SingleStepAttention(nn.Module):
    def __init__(self, embed_dim: int, target_dim: int):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, target_dim)
        self.k_proj = nn.Linear(embed_dim, target_dim)
        self.v_proj = nn.Linear(embed_dim, target_dim)
        self.scale = target_dim**-0.5

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        if query.dim() == 2:
            query = query.unsqueeze(1)  # (batch_size, 1, embed_dim)

        Q = self.q_proj(query)  # (batch_size, 1, target_dim)
        K = self.k_proj(key_value)  # (batch_size, seq_len, target_dim)
        V = self.v_proj(key_value)  # (batch_size, seq_len, target_dim)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)

        target_x = torch.matmul(attn_weights, V).squeeze(1)  # (batch_size, target_dim)
        return target_x


# ===============================================================
# 2. Vector Decomposer (Attention 미사용 수식 분해)
# ===============================================================
class VectorDecomposer(nn.Module):
    def __init__(self, max_n: int, dim: int, alpha: float = 1.0):
        super().__init__()
        self.max_n = max_n
        self.dim = dim
        self.alpha = alpha
        # 최대 MAX_N개 위치에 대한 슬롯 가중치 w
        self.w = nn.Parameter(torch.randn(max_n, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, dim)
        returns: v -> (batch_size, max_n, dim)
        """
        w_bar = torch.mean(self.w, dim=0, keepdim=True)
        w_diff = self.w - w_bar
        std_w = torch.std(self.w, correction=0) + 1e-8

        x_term = x.unsqueeze(1) / self.max_n
        w_term = self.alpha * (w_diff / std_w).unsqueeze(0)

        v = x_term + w_term
        return v


# ===============================================================
# 3. 토큰 개수 예측 모듈 (Attention 미사용, 가벼운 Linear Head)
# ===============================================================
class LengthPredictor(nn.Module):
    def __init__(self, embed_dim: int, max_n: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, max_n),  # 1 ~ max_n 개수 범위 예측 (Class Index: 0 ~ max_n-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)  # (batch_size, max_n)


# ===============================================================
# 4. 통합 AI 모델 파이프라인
# ===============================================================
class OneShotDecomposedAI(nn.Module):
    def __init__(
        self, model_name: str = "Qwen/Qwen2.5-0.5B", max_n: int = 16, alpha: float = 1.0
    ):
        super().__init__()
        self.max_n = max_n

        print(f"[{model_name}] 백본 및 Pre-trained Vocab 로드 중...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_llm = AutoModelForCausalLM.from_pretrained(model_name)

        self.vocab_size = base_llm.config.vocab_size
        self.embed_dim = base_llm.config.hidden_size

        # 사전 학습된 Embedding과 LM Head 연동 및 동결(Freeze)
        self.embedding = base_llm.get_input_embeddings()
        self.lm_head = base_llm.get_output_embeddings()

        for param in self.embedding.parameters():
            param.requires_grad = False
        for param in self.lm_head.parameters():
            param.requires_grad = False

        # 학습 가능한 핵심 모듈 구성
        self.attention_target = SingleStepAttention(self.embed_dim, self.embed_dim)
        self.decomposer = VectorDecomposer(max_n=max_n, dim=self.embed_dim, alpha=alpha)
        self.length_predictor = LengthPredictor(embed_dim=self.embed_dim, max_n=max_n)

    def forward(self, prompt_ids: torch.Tensor):
        embeddings = self.embedding(prompt_ids)
        query = embeddings[:, -1:, :]  # 문맥의 마지막 토큰 활용

        # 1) Target Vector x 1회 추출
        target_x = self.attention_target(query=query, key_value=embeddings)

        # 2) 토큰 개수 예측 (Length Logits)
        length_logits = self.length_predictor(target_x)

        # 3) Attention 없이 MAX_N개 벡터 동시 분해
        v = self.decomposer(target_x)  # (batch_size, max_n, embed_dim)

        # 4) LM Head 통과
        logits = self.lm_head(v)  # (batch_size, max_n, vocab_size)

        return logits, length_logits, target_x

    def generate(self, prompt_text: str, device: str = "cpu") -> str:
        self.eval()
        with torch.no_grad():
            prompt_ids = self.tokenizer.encode(prompt_text, return_tensors="pt").to(
                device
            )
            logits, length_logits, _ = self.forward(prompt_ids)

            # 1. 예측된 토큰 개수 K 추출 (1 ~ MAX_N)
            predicted_len = torch.argmax(length_logits, dim=-1).item() + 1

            # 2. 예측된 K개 범위만큼만 슬라이싱하여 원샷 추출
            pred_ids = torch.argmax(logits[0, :predicted_len, :], dim=-1)

            # 3. 사전 완성된 Vocab으로 디코딩
            generated_text = self.tokenizer.decode(pred_ids, skip_special_tokens=True)
            return generated_text, predicted_len


# ===============================================================
# 5. 실제 데이터 토큰화 기반 학습 및 실행 테스트
# ===============================================================
if __name__ == "__main__":
    MAX_N = 16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 모델 생성
    ai = OneShotDecomposedAI(model_name="Qwen/Qwen2.5-0.5B", max_n=MAX_N)

    # 2. 백본 모델(Qwen)의 dtype에 맞춰 전체 모델 dtype 통일
    model_dtype = ai.embedding.weight.dtype  # 보통 bfloat16 또는 float16
    ai = ai.to(device=device, dtype=model_dtype)

    # 학습할 파라미터 (Attention + Decomposer + LengthPredictor)
    trainable_params = (
        list(ai.attention_target.parameters())
        + list(ai.decomposer.parameters())
        + list(ai.length_predictor.parameters())
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)

    loss_token_fn = nn.CrossEntropyLoss(ignore_index=-100)
    loss_length_fn = nn.CrossEntropyLoss()

    # 학습 데이터셋 (프롬프트, 실제 타겟 문장)
    training_data = [
        ("Artificial Intelligence is", " getting smarter every day."),
        ("Deep learning models can", " solve complex problems."),
        ("Python is a popular", " programming language."),
        ("Natural language processing allows", " computers to understand text."),
    ]

    print("\n=== 학습 시작 (Target Vector + Length Predictor 원샷 학습) ===")
    ai.train()
    epochs = 60

    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        for prompt_text, target_text in training_data:
            optimizer.zero_grad()

            prompt_ids = ai.tokenizer.encode(prompt_text, return_tensors="pt").to(
                device
            )

            # Tokenizer가 실제로 변환한 진짜 토큰 목록 추출 (단어 수 X)
            target_ids_raw = ai.tokenizer.encode(target_text, add_special_tokens=False)
            actual_length = len(target_ids_raw)  # 예: 5개 토큰

            # MAX_N 길이에 맞춰 정답 토큰 텐서 생성 및 패딩(-100은 Loss 계산 시 제외)
            padded_target_ids = torch.full(
                (1, MAX_N), -100, dtype=torch.long, device=device
            )
            padded_target_ids[0, :actual_length] = torch.tensor(
                target_ids_raw, device=device
            )

            # Forward
            logits, length_logits, _ = ai(prompt_ids)

            # 1) 토큰 손실 계산
            token_loss = loss_token_fn(
                logits.view(-1, ai.vocab_size), padded_target_ids.view(-1)
            )

            # 2) 길이 예측 손실 계산 (인덱스는 0 ~ MAX_N-1 이므로 actual_length - 1)
            length_target = torch.tensor([actual_length - 1], device=device)
            length_loss = loss_length_fn(length_logits, length_target)

            # 총 손실 = 토큰 손실 + 0.1 * 길이 손실
            loss = token_loss + 0.1 * length_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:2d}/{epochs} | Loss: {total_loss / len(training_data):.4f}"
            )

    # ===============================================================
    # 6. 추론 테스트 (원샷 자동 길이 측정 및 생성)
    # ===============================================================
    print("\n=== 추론 테스트 ===")
    test_prompt = "Artificial Intelligence is"
    gen_text, pred_len = ai.generate(test_prompt, device=device)

    print(f"입력 프롬프트 : '{test_prompt}'")
    print(f"예측된 토큰 수 : {pred_len}개 토큰")
    print(f"원샷 생성 결과 : '{gen_text}'")
