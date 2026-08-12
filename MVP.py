import re
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
        self.norm = nn.LayerNorm(target_dim)  # LayerNorm 추가

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if query.dim() == 2:
            query = query.unsqueeze(1)

        Q = self.q_proj(query)
        K = self.k_proj(key_value)
        V = self.v_proj(key_value)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            mask = attn_mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e4)

        attn_weights = F.softmax(attn_scores, dim=-1)
        target_x = torch.matmul(attn_weights, V).squeeze(1)

        # 단순 target_x에 LayerNorm만 적용하거나 alpha 비율 조절
        target_x = self.norm(target_x + 0.3 * query.squeeze(1))
        return target_x


class DecomposerBlock(nn.Module):

    def __init__(self, dim: int, expansion_factor: int = 2):
        super().__init__()
        # 1. 토큰 간 맥락 교류를 위한 Self-Attention 추가
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=4, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)

        # 2. 기존 FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * expansion_factor),
            nn.GELU(),
            nn.Linear(dim * expansion_factor, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-Attention 연산 (N개의 토큰 벡터끼리 정보 교환)
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)

        # FFN 연산
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class VectorDecomposer(nn.Module):
    def __init__(self, max_n: int, dim: int, alpha: float = 2.0, num_layers: int = 4):
        super().__init__()
        self.max_n = max_n  # N
        self.dim = dim  # D
        self.alpha = alpha  # \alpha

        # \mathbf w_i 역할: 학습 가능한 위치별 벡터 (N, D)
        self.w = nn.Parameter(torch.randn(max_n, dim) * 0.02)

        self.blocks = nn.ModuleList(
            [DecomposerBlock(dim=dim) for _ in range(num_layers)]
        )
        self.out_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, D)
        B, D = x.shape
        N = self.max_n

        # 1. \bar{\mathbf w} 계산: \frac{1}{N} \sum w_j -> shape: (1, D)
        w_bar = self.w.mean(dim=0, keepdim=True)

        # 2. \mathbf w_i - \bar{\mathbf w} (Zero-mean 설정) -> shape: (N, D)
        w_diff = self.w - w_bar

        # 3. 분모의 L2 Norm 평균 계산: \sqrt{ \frac{1}{N} \sum ||w_j - \bar{w}||^2 }
        # w_diff.pow(2).sum(dim=-1) 은 각 위치별 ||w_i - \bar{w}||^2
        norm_std = torch.sqrt(w_diff.pow(2).sum(dim=-1).mean() + 1e-8)

        # 4. \mathbf v_i 계산 수식 집행
        # (1/N)*x  -> (B, 1, D)
        x_base = (x / N).unsqueeze(1)

        # \alpha * (w_i - w_bar) / norm_std -> (1, N, D)
        fluctuation = self.alpha * (w_diff / norm_std).unsqueeze(0)

        # \mathbf v_i = (1/N)\mathbf x + \alpha * ( ... )
        v = x_base + fluctuation  # Broadcast 되어서 (B, N, D) 형태가 됨

        # --- 수식적으로 \sum_{i=1}^N v_i == x 가 보장됨! ---

        # 5. Decomposer 깊은 블록 통과
        for block in self.blocks:
            v = block(v)

        return self.out_proj(v)


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
        base_llm = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32
        )

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

    def forward(self, prompt_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        embeddings = self.embedding(prompt_ids)

        if attention_mask is not None:
            # (B, L, 1) 형태로 확장하여 PAD 토큰을 계산에서 제외
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(embeddings)

            # 실제 토큰 위치의 임베딩만 다 더함 (B, D)
            sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)

            # 실제 토큰의 개수로 나눔 (0으로 나누기 방지 clamp)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)

            # 평균 계산 후 (B, 1, D) 형태로 차원 변경
            query = (sum_embeddings / sum_mask).unsqueeze(1)
        else:
            # 마스크가 없는 경우 전체 시퀀스 평균
            query = embeddings.mean(dim=1, keepdim=True)

        # 이제 맥락 전체가 압축된 Query로 Attention 수행
        target_x = self.attention_target(
            query=query, key_value=embeddings, attn_mask=attention_mask
        )
        length_logits = self.length_predictor(target_x)
        v = self.decomposer(target_x)
        logits = self.lm_head(v)

        return logits, length_logits, target_x

    def generate(self, prompt_text: str, device: str = "cpu"):
        self.eval()
        with torch.no_grad():
            messages = [{"role": "user", "content": prompt_text}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            prompt_inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(
                device
            )
            prompt_ids = prompt_inputs.input_ids
            attn_mask = prompt_inputs.attention_mask

            logits, length_logits, target_x = self.forward(
                prompt_ids, attention_mask=attn_mask
            )

            # [디버깅 용] target_x 벡터가 달라지는지 눈으로 직접 확인!
            sample_val = target_x[0, :3].detach().cpu().numpy()
            print(
                f"[DEBUG x 벡터 샘플]: {sample_val[0]:.4f}, {sample_val[1]:.4f}, {sample_val[2]:.4f}"
            )

            predicted_len = torch.argmax(length_logits, dim=-1).item() + 1
            pred_ids = torch.argmax(logits[0, :predicted_len, :], dim=-1)

            generated_text = self.tokenizer.decode(pred_ids, skip_special_tokens=True)
            return generated_text, predicted_len


# ===============================================================
# 5. CSV 데이터 로드 및 학습
# ===============================================================
if __name__ == "__main__":
    MAX_N = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ai = OneShotDecomposedAI(model_name="Qwen/Qwen2.5-0.5B", max_n=MAX_N)
    ai = torch.compile(ai, mode="default")
    ai = ai.to(device=device)

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
            utterances = re.findall(r"['\"]+(.*?)['\"]+", dialog_str, re.DOTALL)
            utterances = [u.strip() for u in utterances if u.strip()]

            if len(utterances) < 2:
                continue

            messages = []
            for i, text in enumerate(utterances[:-1]):
                role = "user" if i % 2 == 0 else "assistant"
                messages.append({"role": role, "content": text})

            formatted = ai.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            formatted_prompts.append(formatted)
            targets.append(utterances[-1])

        except Exception as e:
            continue

    print(f"총 {len(formatted_prompts)}개의 대화 데이터셋 구성 완료!")

    if len(formatted_prompts) == 0:
        raise ValueError("데이터가 0개 수집되었습니다!")

    # 2. 토크나이징 및 Tensor 구축
    tokenized_batch = ai.tokenizer(formatted_prompts, return_tensors="pt", padding=True)
    batch_prompt_ids = tokenized_batch.input_ids.to(device)
    batch_attention_mask = tokenized_batch.attention_mask.to(device)

    batch_target_list = [ai.tokenizer.encode(t) for t in targets]
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
    epochs = 300

    # Warmup / Compile
    with torch.amp.autocast("cuda"):
        logits, length_logits, _ = ai(
            batch_prompt_ids, attention_mask=batch_attention_mask
        )

        token_loss = loss_token_fn(
            logits.view(-1, ai.vocab_size), padded_targets.view(-1)
        )
        length_loss = loss_length_fn(length_logits, length_targets)

        total_loss = token_loss + 1.0 * length_loss

    optimizer.zero_grad(set_to_none=True)

    # 단 1번만 backward 실행! (retain_graph 불필요)
    scaler.scale(total_loss).backward()

    scaler.step(optimizer)
    scaler.update()

    torch.cuda.synchronize()
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda"):
            logits, length_logits, _ = ai(
                batch_prompt_ids, attention_mask=batch_attention_mask
            )

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
    while True:
        test_prompt = input("User 입력 >>>: ")

        if test_prompt == "exit":
            break

        gen_text, pred_len = ai.generate(test_prompt, device=device)
        print(f"예측된 토큰 수 : {pred_len}개")
        print(f"AI 원샷 답변 : '{gen_text}'\n")
