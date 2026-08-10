import re
import ast
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.backends.cudnn.benchmark = True


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
            query = query.unsqueeze(1)

        Q = self.q_proj(query)
        K = self.k_proj(key_value)
        V = self.v_proj(key_value)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)

        target_x = torch.matmul(attn_weights, V).squeeze(1)
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
        self.w = nn.Parameter(torch.empty(max_n, dim, dim))
        nn.init.xavier_uniform_(self.w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = torch.einsum("bd, ndk -> bnk", x, self.w)
        return v


# ===============================================================
# 3. 토큰 개수 예측 모듈
# ===============================================================
class LengthPredictor(nn.Module):

    def __init__(self, embed_dim: int, max_n: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, max_n),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ===============================================================
# 4. 통합 AI 모델 파이프라인
# ===============================================================
class OneShotDecomposedAI(nn.Module):

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        max_n: int = 16,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.max_n = max_n

        print(f"[{model_name}] 백본 및 Pre-trained Vocab 로드 중...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_llm = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)

        self.vocab_size = base_llm.config.vocab_size
        self.embed_dim = base_llm.config.hidden_size

        self.embedding = base_llm.get_input_embeddings()
        self.lm_head = base_llm.get_output_embeddings()

        for param in self.embedding.parameters():
            param.requires_grad = False
        for param in self.lm_head.parameters():
            param.requires_grad = False

        self.attention_target = SingleStepAttention(self.embed_dim, self.embed_dim)
        self.decomposer = VectorDecomposer(max_n=max_n, dim=self.embed_dim, alpha=alpha)
        self.length_predictor = LengthPredictor(embed_dim=self.embed_dim, max_n=max_n)

    def forward(self, prompt_ids: torch.Tensor):
        embeddings = self.embedding(prompt_ids)
        query = embeddings[:, -1:, :]

        target_x = self.attention_target(query=query, key_value=embeddings)
        length_logits = self.length_predictor(target_x)
        v = self.decomposer(target_x)
        logits = self.lm_head(v)

        return logits, length_logits, target_x

    def generate(self, prompt_text: str, device: str = "cpu") -> str:
        self.eval()
        with torch.no_grad():
            messages = [{"role": "user", "content": prompt_text}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            prompt_ids = self.tokenizer.encode(
                formatted_prompt, return_tensors="pt"
            ).to(device)
            logits, length_logits, _ = self.forward(prompt_ids)

            predicted_len = torch.argmax(length_logits, dim=-1).item() + 1
            pred_ids = torch.argmax(logits[0, :predicted_len, :], dim=-1)

            generated_text = self.tokenizer.decode(pred_ids, skip_special_tokens=True)
            return generated_text, predicted_len


# ===============================================================
# 5. CSV 데이터 로드 및 학습
# ===============================================================
if __name__ == "__main__":
    MAX_N = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ai = OneShotDecomposedAI(model_name="Qwen/Qwen2.5-0.5B", max_n=MAX_N)
    ai = torch.compile(ai, mode="default")
    ai = ai.to(device=device)

    # 1. train.csv 불러오기 및 데이터 전처리
    # ===============================================================
    # 데이터 제한 설정
    # ===============================================================
    MAX_SAMPLES = 100

    ai.tokenizer.padding_side = "left"
    if ai.tokenizer.pad_token is None:
        ai.tokenizer.pad_token = ai.tokenizer.eos_token

    print("\n'train.csv' 로딩 및 파싱 중...")
    df = pd.read_csv("train.csv")

    if len(df) > MAX_SAMPLES:
        df = df.iloc[:MAX_SAMPLES]
        print(f"데이터셋을 상위 {MAX_SAMPLES}개로 제한했습니다.")

    formatted_prompts = []
    targets = []

    for idx, row in df.iterrows():
        try:
            dialog_str = str(row["dialog"])

            # [핵심] 정규식으로 따옴표 형태에 구애받지 않고 개별 문장만 추출
            # 큰따옴표/작은따옴표로 둘러싸인 텍스트 영역을 정밀하게 캡처합니다.
            utterances = re.findall(r"['\"]+(.*?)['\"]+", dialog_str, re.DOTALL)

            # 불필요한 공백 제거 및 빈 문장 필터링
            utterances = [u.strip() for u in utterances if u.strip()]

            # 대화 문장이 최소 2개 이상이어야 (질문-답변) 학습 가능
            if len(utterances) < 2:
                continue

            # 짝수번째=user, 홀수번째=assistant 지정
            messages = []
            for i, text in enumerate(utterances[:-1]):
                role = "user" if i % 2 == 0 else "assistant"
                messages.append({"role": role, "content": text})

            # Chat Template 적용
            formatted = ai.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            formatted_prompts.append(formatted)
            targets.append(utterances[-1])  # 마지막 발화를 AI의 정답(Target)으로 설정

        except Exception as e:
            continue

    print(f"총 {len(formatted_prompts)}개의 대화 데이터셋 구성 완료!")

    # 데이터가 제대로 파싱되었는지 방어 코드
    if len(formatted_prompts) == 0:
        raise ValueError(
            "데이터가 0개 수집되었습니다! CSV 파일의 'dialog' 컬럼 형식을 다시 확인해주세요."
        )

    # 2. 토크나이징 및 Tensor 구축
    batch_prompt_ids = ai.tokenizer(
        formatted_prompts, return_tensors="pt", padding=True
    ).input_ids.to(device)

    batch_target_list = [ai.tokenizer.encode(t) for t in targets]

    # [핵심] CUDA Assert 에러 방지 (MAX_N 상한선 클램핑)
    actual_lengths = [min(len(t), MAX_N) for t in batch_target_list]

    padded_targets = torch.full(
        (len(formatted_prompts), MAX_N), -100, dtype=torch.long, device=device
    )
    for i, t_ids in enumerate(batch_target_list):
        length = actual_lengths[i]
        padded_targets[i, :length] = torch.tensor(t_ids[:length], device=device)

    length_targets = torch.tensor([l - 1 for l in actual_lengths], device=device)

    # 3. 학습 설정
    trainable_params = (
        list(ai.attention_target.parameters())
        + list(ai.decomposer.parameters())
        + list(ai.length_predictor.parameters())
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)

    loss_token_fn = nn.CrossEntropyLoss(ignore_index=-100)
    loss_length_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler()

    # 4. 학습 시작
    print("\n=== CSV 대화 데이터 기반 원샷 학습 시작 ===")
    ai.train()
    epochs = 2000

    # Warmup / Compile
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda"):
        logits_w, length_logits_w, _ = ai(batch_prompt_ids)
        loss_w = loss_token_fn(
            logits_w.view(-1, ai.vocab_size), padded_targets.view(-1)
        ) + 0.2 * loss_length_fn(length_logits_w, length_targets)

    scaler.scale(loss_w).backward()
    scaler.step(optimizer)
    scaler.update()

    torch.cuda.synchronize()
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda"):
            logits, length_logits, _ = ai(batch_prompt_ids)

            token_loss = loss_token_fn(
                logits.view(-1, ai.vocab_size), padded_targets.view(-1)
            )
            length_loss = loss_length_fn(length_logits, length_targets)
            loss = token_loss + 0.2 * length_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}/{epochs} - Loss: {loss.item():.4f}")

    torch.cuda.synchronize()
    end_time = time.time()
    print(f"학습 소요 시간: {end_time - start_time:.4f}초")

    # 5. 추론 테스트
    print("\n=== 대화 테스트 ===")
    test_prompt = input("User 입력 >>>: ")
    gen_text, pred_len = ai.generate(test_prompt, device=device)
    print(f"예측된 토큰 수 : {pred_len}개")
    print(f"AI 원샷 답변 : '{gen_text}'")
