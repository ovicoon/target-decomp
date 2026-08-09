import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------
# 1. Custom Layer: 수식을 수행하는 정규화 및 분해 모듈
# ---------------------------------------------------------------
class VectorDecomposer(nn.Module):
    def __init__(self, N, dim, alpha=1.0):
        super().__init__()
        self.N = N
        self.dim = dim
        self.alpha = alpha

        # w_i를 학습 가능한 파라미터(Codebook)로 설정 (N x dim)
        self.w = nn.Parameter(torch.randn(N, dim))

    def forward(self, x):
        """
        x: Target Vector (batch_size, dim)
        returns: v (batch_size, N, dim)
        """
        batch_size = x.size(0)

        # [식 1] w의 평균 (N개의 벡터에 대한 평균) -> (1, dim)
        w_bar = torch.mean(self.w, dim=0, keepdim=True)

        # [식 2 분모] 표준편차(RMS) 계산 -> scalar
        # (w_j - w_bar)^2 의 평균 후 sqrt (eps로 0 나누기 방지)
        w_diff = self.w - w_bar  # (N, dim)
        std_w = torch.sqrt(torch.mean(torch.sum(w_diff**2, dim=1)) + 1e-8)

        # [식 2] v_i 계산
        # (1/N) * x  -> (batch_size, 1, dim)
        x_term = (1.0 / self.N) * x.unsqueeze(1)

        # alpha * (w_i - w_bar) / std_w -> (1, N, dim)
        w_term = self.alpha * (w_diff / std_w).unsqueeze(0)

        # v_i들 생성 -> (batch_size, N, dim)
        v = x_term + w_term
        return v


# ---------------------------------------------------------------
# 2. 전체 Pipeline (Dummy Attention + Decomposer + Dummy LLM)
# ---------------------------------------------------------------
class TokenDecomposedLLMPipeline(nn.Module):
    def __init__(self, input_dim, target_dim, N, vocab_size):
        super().__init__()
        self.N = N
        self.target_dim = target_dim

        # 1) Query를 받아 Target Vector x를 만드는 Network (FFN)
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, target_dim)
        )

        # 2) Target x를 N개의 v_i로 분해하는 모듈
        self.decomposer = VectorDecomposer(N=N, dim=target_dim, alpha=1.0)

        # 3) Soft-Prompt v_i를 받아 토큰을 예측하는 가상의 간단한 LLM
        #    (실제로는 HuggingFace의 LLaMA, GPT 등의 Embedding Layer로 대체됨)
        self.llm_head = nn.Linear(target_dim, vocab_size)

    def forward(self, query):
        # Step 1: Target Vector x 생성
        x = self.network(query)  # (batch_size, target_dim)

        # Step 2: x를 N개의 v_i 벡터로 분해
        v = self.decomposer(x)  # (batch_size, N, target_dim)

        # Step 3: v_i 벡터들을 LLM에 Soft-Prompt로 전달하여 각 위치의 토큰 로짓 계산
        logits = self.llm_head(v)  # (batch_size, N, vocab_size)

        return logits, x, v


# ---------------------------------------------------------------
# 3. 역전파(Backprop) 동작 및 검증 테스트
# ---------------------------------------------------------------
if __name__ == "__main__":
    # 하이퍼파라미터 설정
    BATCH_SIZE = 2
    INPUT_DIM = 16
    TARGET_DIM = 8
    N_TOKENS = 5
    VOCAB_SIZE = 100

    # 모델 및 임의의 입력 데이터 생성
    model = TokenDecomposedLLMPipeline(INPUT_DIM, TARGET_DIM, N_TOKENS, VOCAB_SIZE)
    dummy_query = torch.randn(BATCH_SIZE, INPUT_DIM)

    # 임의의 Target Token 정답 (Batch x N_Tokens)
    dummy_target_tokens = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, N_TOKENS))

    # 순전파 (Forward Pass)
    logits, x, v = model(dummy_query)

    # 손실 함수 계산 (CrossEntropy Loss)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits.view(-1, VOCAB_SIZE), dummy_target_tokens.view(-1))

    # 역전파 (Backward Pass)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    loss.backward()

    # -----------------------------------------------------------
    # 결과 출력 및 검증
    # -----------------------------------------------------------
    print("=== 1. 수식 검증 (sum(v_i) == x 인가?) ===")
    v_sum = torch.sum(v, dim=1)  # 각 샘플별 sum(v_i)
    is_close = torch.allclose(x, v_sum, atol=1e-5)
    print(f"Target x 와 sum(v_i) 일치 여부: {is_close}")
    print(f"오차 차이 최대값: {torch.max(torch.abs(x - v_sum)).item():.8f}\n")

    print("=== 2. 역전파(Backpropagation) 기울기 검증 ===")
    print(f"Loss 값: {loss.item():.4f}")
    print(
        f"1) Network Layer 기울기 존재 여부 : {model.network[0].weight.grad is not None}"
    )
    print(f"2) Decomposer w_i 기울기 존재 여부 : {model.decomposer.w.grad is not None}")

    # 가중치 업데이트 수행
    optimizer.step()
    print("\n✅ 파라미터가 기울기를 받아 성공적으로 업데이트되었습니다!")
