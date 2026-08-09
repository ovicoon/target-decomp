import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


# ===============================================================
# 1. Target Vector 생성을 위한 Single-head/Multi-head Attention
# ===============================================================
class SingleStepAttention(nn.Module):
    """
    Query, Key, Value를 받아 '단 1번의 Attention'으로 Target Vector x를 추출하는 모듈
    """

    def __init__(self, embed_dim: int, target_dim: int):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, target_dim)
        self.k_proj = nn.Linear(embed_dim, target_dim)
        self.v_proj = nn.Linear(embed_dim, target_dim)
        self.scale = target_dim**-0.5

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        query: (batch_size, 1, embed_dim) 또는 (batch_size, embed_dim)
        key_value: (batch_size, seq_len, embed_dim)
        returns: Target Vector x -> (batch_size, target_dim)
        """
        if query.dim() == 2:
            query = query.unsqueeze(1)  # (batch_size, 1, embed_dim)

        Q = self.q_proj(query)  # (batch_size, 1, target_dim)
        K = self.k_proj(key_value)  # (batch_size, seq_len, target_dim)
        V = self.v_proj(key_value)  # (batch_size, seq_len, target_dim)

        # Scaled Dot-Product Attention (1회 수행)
        attn_scores = (
            torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        )  # (batch_size, 1, seq_len)
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Weighted Sum을 통한 Target Vector x 추출
        target_x = torch.matmul(attn_weights, V).squeeze(1)  # (batch_size, target_dim)
        return target_x


# ===============================================================
# 2. Vector Decomposer (Attention 미사용)
# ===============================================================
class VectorDecomposer(nn.Module):
    """
    Attention을 사용하지 않고, 수학적 정규화 및 파라미터 기반으로
    Target Vector x를 N개의 v_i 벡터로 분해하는 모듈
    """

    def __init__(self, N: int, dim: int, alpha: float = 1.0):
        super().__init__()
        self.N = N
        self.dim = dim
        self.alpha = alpha
        # 학습 가능한 Codebook 매개변수 w (N x dim)
        self.w = nn.Parameter(torch.randn(N, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, dim)
        returns: v -> (batch_size, N, dim)
        """
        w_bar = torch.mean(self.w, dim=0, keepdim=True)  # (1, dim)
        w_diff = self.w - w_bar  # (N, dim)
        std_w = torch.std(self.w, correction=0) + 1e-8  # 통계적 표준편차

        x_term = x.unsqueeze(1) / self.N  # (batch_size, 1, dim)
        w_term = self.alpha * (w_diff / std_w).unsqueeze(0)  # (1, N, dim)

        v = x_term + w_term  # (batch_size, N, dim)
        return v


# ===============================================================
# 3. 실전 동작 통합 AI Pipeline (Pre-trained Vocab 연동)
# ===============================================================
class DecomposedAIPipeline(nn.Module):
    def __init__(self, model_name: str = "gpt2", N_tokens: int = 4, alpha: float = 1.0):
        super().__init__()
        self.N = N_tokens

        # 1) 사전 학습된 Tokenizer & LLM Head / Embedding 로드
        print(f"[{model_name}] 모델 및 사전 완성된 Vocab 로드 중...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_llm = AutoModelForCausalLM.from_pretrained(model_name)

        # Vocab Size 및 Embedding Dim 추출
        self.vocab_size = base_llm.config.vocab_size
        self.embed_dim = base_llm.config.hidden_size

        # 실제 사전 학습된 Embedding Layer와 LM Head 추출 및 동결(Freeze - 선택사항)
        self.embedding = base_llm.get_input_embeddings()
        self.lm_head = base_llm.get_output_embeddings()

        # 2) 조건 1: Target Vector x를 만드는 '단 1회의 Attention'
        self.attention_target = SingleStepAttention(
            embed_dim=self.embed_dim, target_dim=self.embed_dim
        )

        # 3) 조건 2: Attention을 안 쓰는 Vector Decomposer
        self.decomposer = VectorDecomposer(N=N_tokens, dim=self.embed_dim, alpha=alpha)

    def forward(self, input_ids: torch.Tensor):
        """
        input_ids: (batch_size, seq_len)
        """
        # 입력 텍스트를 사전 정의된 Vocab Embedding 벡터로 변환
        embeddings = self.embedding(input_ids)  # (batch_size, seq_len, embed_dim)

        # Query: 입력 문장의 마지막 토큰 임베딩 활용
        query = embeddings[:, -1:, :]  # (batch_size, 1, embed_dim)

        # [단계 1] Attention을 단 1회만 적용하여 Target Vector x 산출
        target_x = self.attention_target(
            query=query, key_value=embeddings
        )  # (batch_size, embed_dim)

        # [단계 2] Attention 없이 수식/파라미터로 N개의 Soft-Prompt v_i 분해
        v = self.decomposer(target_x)  # (batch_size, N, embed_dim)

        # [단계 3] 사전 완성된 Vocab LM Head를 통해 최종 토큰 로짓 출력
        logits = self.lm_head(v)  # (batch_size, N, vocab_size)

        return logits, target_x, v

    def generate_text(self, text_prompt: str) -> str:
        """
        입력 텍스트를 받아 모델을 거쳐 예측된 N개의 토큰을 문장으로 복원하는 함수
        """
        self.eval()
        with torch.no_grad():
            # 사전 완성된 Tokenizer로 토큰화
            inputs = self.tokenizer(text_prompt, return_tensors="pt")
            input_ids = inputs["input_ids"]

            # Forward
            logits, _, _ = self.forward(input_ids)  # (1, N, vocab_size)

            # 가장 확률이 높은 토큰 ID 선택 (Greedy Decoding)
            predicted_ids = torch.argmax(logits, dim=-1).squeeze(0)  # (N,)

            # 사전 완성된 Vocab으로 디코딩
            generated_text = self.tokenizer.decode(
                predicted_ids, skip_special_tokens=True
            )
            return generated_text


# ===============================================================
# 4. 실행 및 학습 테스트
# ===============================================================
if __name__ == "__main__":
    # 모델 초기화 (GPT-2의 완성된 Vocab과 구조 사용)
    N_TOKENS = 3  # 분해할 토큰 개수
    ai_model = DecomposedAIPipeline(model_name="gpt2", N_tokens=N_TOKENS)

    # 1. 추론(Inference) 테스트
    prompt = "Artificial Intelligence is"
    print(f"\n[입력 프롬프트]: '{prompt}'")
    output_text = ai_model.generate_text(prompt)
    print(f"[예측 생성 텍스트 ({N_TOKENS}개 토큰)]: '{output_text}'")

    # 2. 수식 검증 (sum(v_i) == Target x)
    inputs = ai_model.tokenizer(prompt, return_tensors="pt")
    logits, target_x, v = ai_model(inputs["input_ids"])
    v_sum = torch.sum(v, dim=1)

    print("\n=== 수식 및 역전파 검증 ===")
    print(
        f"Target x 와 sum(v_i) 일치 여부: {torch.allclose(target_x, v_sum, atol=1e-5)}"
    )

    # 3. 실제 학습(Backprop) 동작 테스트
    # 가상의 타겟 토큰 정답
    dummy_labels = torch.tensor([[262, 379, 1517]])  # GPT-2 vocab id 샘플
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits.view(-1, ai_model.vocab_size), dummy_labels.view(-1))

    optimizer = torch.optim.AdamW(ai_model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Loss 계산 완료: {loss.item():.4f}")
    print("✅ 역전파 및 가중치 업데이트가 성공적으로 수행되었습니다.")
