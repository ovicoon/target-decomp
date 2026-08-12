import re
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset

torch.backends.cudnn.benchmark = True


# ===============================================================
# 1. Target Vector 생성 모듈
# ===============================================================
class SingleStepAttention(nn.Module):
    def __init__(self, embed_dim: int, target_dim: int):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, target_dim)
        self.k_proj = nn.Linear(embed_dim, target_dim)
        self.v_proj = nn.Linear(embed_dim, target_dim)
        self.scale = target_dim**-0.5
        self.norm = nn.LayerNorm(target_dim)

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

        target_x = self.norm(target_x + 0.3 * query.squeeze(1))
        return target_x


class DecomposerBlock(nn.Module):
    def __init__(self, dim: int, expansion_factor: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expansion_factor),
            nn.GELU(),
            nn.Linear(dim * expansion_factor, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class VectorDecomposer(nn.Module):
    def __init__(self, max_n: int, dim: int, alpha: float = 0.5, num_layers: int = 4):
        super().__init__()
        self.max_n = max_n
        self.dim = dim
        self.alpha = alpha

        self.w = nn.Parameter(torch.randn(max_n, dim) * 0.02)
        self.blocks = nn.ModuleList(
            [DecomposerBlock(dim=dim) for _ in range(num_layers)]
        )
        self.out_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D = x.shape
        N = self.max_n

        w_bar = self.w.mean(dim=0, keepdim=True)
        w_diff = self.w - w_bar
        norm_std = torch.sqrt(w_diff.pow(2).sum(dim=-1).mean() + 1e-8)

        x_base = (x / N).unsqueeze(1)
        fluctuation = self.alpha * (w_diff / norm_std).unsqueeze(0)
        v = x_base + fluctuation

        for block in self.blocks:
            v = block(v)

        return self.out_proj(v)


# ===============================================================
# 3. 토큰 개수 예측 모듈 (Regression 기반)
# ===============================================================
class LengthPredictor(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 1),  # 연속적인 scalar 출력
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ===============================================================
# 4. 통합 AI 모델 파이프라인
# ===============================================================
class OneShotDecomposedAI(nn.Module):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        max_n: int = 16,
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
            param.requires_grad = True

        self.attention_target = SingleStepAttention(self.embed_dim, self.embed_dim)
        self.decomposer = VectorDecomposer(max_n=max_n, dim=self.embed_dim)
        # [수정] max_n 인자 제거
        self.length_predictor = LengthPredictor(embed_dim=self.embed_dim)

    def forward(self, prompt_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        embeddings = self.embedding(prompt_ids)

        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(embeddings)
            sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            query = (sum_embeddings / sum_mask).unsqueeze(1)
        else:
            query = embeddings.mean(dim=1, keepdim=True)

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
            messages = [{"role": "user", "content": prompt_text.strip()}]
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

            sample_val = target_x[0, :3].detach().cpu().numpy()
            print(
                f"[DEBUG x 벡터 샘플]: {sample_val[0]:.4f}, {sample_val[1]:.4f}, {sample_val[2]:.4f}"
            )

            # [수정] Regression 예측값을 정수로 변환하여 사용
            raw_len = length_logits.squeeze().item()
            predicted_len = int(round(raw_len))
            predicted_len = max(1, min(predicted_len, self.max_n))

            pred_ids = torch.argmax(logits[0, :predicted_len, :], dim=-1)
            generated_text = self.tokenizer.decode(pred_ids, skip_special_tokens=True)
            return generated_text, predicted_len, raw_len


# ===============================================================
# 5. CSV 데이터 로드 및 Regression 기반 학습
# ===============================================================
if __name__ == "__main__":
    TARGET_LOSS = 0
    PATIENCE = 3
    patience_counter = 0
    best_loss = float("inf")
    MAX_N = 32

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ai = OneShotDecomposedAI(model_name="Qwen/Qwen2.5-0.5B", max_n=MAX_N)

    ai = torch.compile(ai, mode="default")
    ai = ai.to(device=device)

    MAX_SAMPLES = 10

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

    length_targets_float = torch.tensor(
        actual_lengths, dtype=torch.float32, device=device
    )

    dataset = TensorDataset(
        batch_prompt_ids, batch_attention_mask, padded_targets, length_targets_float
    )
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    trainable_params = (
        list(ai.attention_target.parameters())
        + list(ai.decomposer.parameters())
        + list(ai.length_predictor.parameters())
        + list(ai.lm_head.parameters())
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)

    loss_token_fn = nn.CrossEntropyLoss(ignore_index=-100)
    loss_length_fn = nn.SmoothL1Loss()
    scaler = torch.amp.GradScaler("cuda")

    print("\n=== 미니배치 기반 원샷 학습 시작 (Regression Length Predictor) ===")
    ai.train()
    epochs = 500

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0

        for b_prompt, b_mask, b_target, b_len_target_float in train_loader:
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                logits, length_logits, _ = ai(b_prompt, attention_mask=b_mask)

                token_loss = loss_token_fn(
                    logits.view(-1, ai.vocab_size), b_target.view(-1)
                )

                pred_len_float = length_logits
                length_loss = loss_length_fn(pred_len_float, b_len_target_float)

                loss = token_loss + 0.5 * length_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        if epoch % 1 == 0:
            print(f"Epoch {epoch}/{epochs} - Avg Loss: {avg_loss:.4f}")

        if avg_loss <= TARGET_LOSS:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(
                    f"\n🎯 목표 Loss ({TARGET_LOSS}) 이하로 {PATIENCE}회 지속되었습니다."
                )
                print(
                    f"과적합 방지를 위해 Epoch {epoch}에서 학습을 조기 종료(Early Stopping)합니다!"
                )
                print(f"최종 Loss: {avg_loss:.4f}")
                break
        else:
            patience_counter = 0

    torch.cuda.synchronize()
    end_time = time.time()
    print(f"학습 소요 시간: {end_time - start_time:.4f}초")

    print("\n=== 대화 테스트 (Regression Predictor) ===")
    while True:
        test_prompt = input("User 입력 >>>: ")

        if test_prompt == "exit":
            break

        gen_text, pred_len, raw_len = ai.generate(test_prompt, device=device)
        print(f"예측된 토큰 수 : {pred_len}개 (Raw float: {raw_len:.2f})")
        print(f"AI 원샷 답변 : '{gen_text}'\n")
