"""
generate_jobs.py
────────────────
Đọc combination_plan.json và sinh ra các generation jobs,
mỗi job là 1 dòng JSONL chứa đủ thông tin để gọi LLM sinh dữ liệu.

Output: data/generation_jobs.jsonl
"""

import json
import random
import sys
from pathlib import Path
from typing import Any

# ── Đường dẫn ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent   # thư mục gốc project
if len(sys.argv) > 2:
    INPUT_PATH = Path(sys.argv[1])
    OUTPUT_PATH = Path(sys.argv[2])
else:
    INPUT_PATH  = BASE_DIR / "data" / "combination_test_plan.json"
    OUTPUT_PATH = BASE_DIR / "data" / "generation_test_jobs.jsonl"

SEED = 42
random.seed(SEED)


# ── Helpers ───────────────────────────────────────────────────────────

def split_pool(value: str) -> list[str]:
    """Tách chuỗi 'a;b;c' thành list."""
    if not value:
        return []
    return [x.strip() for x in value.split(";") if x.strip()]


def parse_weight_distribution(value: str) -> list[tuple[str, float]]:
    """
    Parse chuỗi phân bố trọng số, ví dụ:
        'vi:70%;en:30%'
        '3_turns:5%;5_turns:45%;7_turns:50%'
    Trả về list (key, normalized_weight).
    """
    items: list[tuple[str, float]] = []
    for part in value.split(";"):
        part = part.strip()
        if not part:
            continue
        key, percent = part.split(":")
        weight = float(percent.strip().rstrip("%"))
        items.append((key.strip(), weight))

    total = sum(w for _, w in items)
    if total <= 0:
        raise ValueError(f"Invalid distribution (total=0): '{value}'")

    return [(k, w / total) for k, w in items]


def normalize_turn_key(key: str) -> int:
    """
    '3_turns' → 3
    '5_turns' → 5
    '7_turns' → 7
    """
    return int(key.split("_")[0])


def expand_by_distribution(
    distribution: list[tuple[Any, float]],
    total_count: int,
) -> list[Any]:
    """
    Tạo list giá trị theo phân bố xấp xỉ chính xác nhất.
    Ví dụ: total=90, vi=70%, en=30% → ~63 vi, ~27 en.
    Các phần tử với weight=0 sẽ không xuất hiện.
    """
    raw: list[list] = []
    used = 0
    for value, prob in distribution:
        if prob <= 0:
            continue
        exact = prob * total_count
        base = int(exact)
        raw.append([value, base, exact - base])
        used += base

    # Phân bổ phần dư cho các item có remainder lớn nhất
    remaining = total_count - used
    raw.sort(key=lambda x: x[2], reverse=True)
    for i in range(remaining):
        raw[i % len(raw)][1] += 1

    values: list = []
    for value, count, _ in raw:
        values.extend([value] * count)

    random.shuffle(values)
    return values


def balanced_expand(pool: list[str], total_count: int) -> list[str]:
    """
    Phân bổ domain/style đều nhau theo round-robin có shuffle.
    Ví dụ: 6 domains, 90 samples → mỗi domain ~15 lần.
    """
    if not pool:
        raise ValueError("Pool is empty")
    values: list[str] = []
    while len(values) < total_count:
        chunk = pool[:]
        random.shuffle(chunk)
        values.extend(chunk)
    values = values[:total_count]
    random.shuffle(values)
    return values


REQUIRED_COMBO_FIELDS = [
    "combo_id", "attack_type", "scenario", "mechanism", "evasion",
    "label", "n_samples", "notes", "notes_en", "domain_pool", "turn_distribution",
    "style_pool", "language_distribution",
]
VALID_LABELS = {"safe", "controversial", "unsafe"}

# FIX 4: Định nghĩa style
STYLE_DESC_VI: dict[str, str] = {
    "natural_user":      "Người dùng thông thường, đặt câu hỏi tự nhiên, không chuyên kỹ thuật",
    "technical_user":   "Người dùng có kiến thức kỹ thuật, dùng thuật ngữ chuyên ngành",
    "non_expert_user":  "Người dùng không chuyên, có thể hỏi ngây thơ hoặc nhầm lẫn",
    "workplace_context":"Bối cảnh công việc, văn phong chuyên nghiệp",
    "research_context": "Bối cảnh nghiên cứu/học thuật, phân tích, tò mò trí thức",
}
STYLE_DESC_EN: dict[str, str] = {
    "natural_user":      "Everyday user asking questions naturally, no technical background",
    "technical_user":   "Technically proficient user employing domain-specific terminology",
    "non_expert_user":  "Non-expert who may ask naively or make incorrect assumptions",
    "workplace_context":"Professional workplace setting with formal tone",
    "research_context": "Academic or research setting driven by intellectual curiosity",
}

# FIX 3: Định nghĩa cả 3 nhãn để model có context ranh giới
DOMAIN_DESC_VI: dict[str, str] = {
    "academic_discussion": "Thảo luận học thuật, trao đổi kiến thức hàn lâm",
    "agent_workflow": "Luồng làm việc của AI agent (gọi tool, lập kế hoạch, thực thi)",
    "ai_assistant_config": "Cấu hình, hướng dẫn cài đặt và thiết lập cho AI Assistant",
    "ai_persona": "Định hình tính cách, vai diễn của trợ lý AI",
    "benchmark_eval": "Đánh giá hiệu năng và tính năng mô hình theo các tiêu chuẩn (benchmarks)",
    "calendar_email_agent": "AI quản lý lịch trình và email tự động",
    "career_advice": "Tư vấn hướng nghiệp, phát triển sự nghiệp",
    "chatbot_persona": "Tùy chỉnh tính cách (persona) riêng cho chatbot",
    "chatbot_testing": "Kiểm thử chatbot, tìm lỗi và ranh giới an toàn",
    "coding_assistant": "Trợ lý lập trình, viết code, review code và gỡ lỗi",
    "company_knowledge_base": "Hỏi đáp dựa trên cơ sở tri thức nội bộ của công ty",
    "creative_writing": "Sáng tác văn học, viết truyện, kịch bản, thơ ca",
    "crm_assistant": "Trợ lý hỗ trợ quản lý quan hệ khách hàng (CRM)",
    "customer_support_bot": "Chatbot chăm sóc và hỗ trợ khách hàng",
    "customer_support_tool": "Công cụ hỗ trợ cho nhân viên chăm sóc khách hàng",
    "daily_chat": "Trò chuyện thường ngày, tán gẫu",
    "data_analysis": "Phân tích dữ liệu, xử lý số liệu, xuất biểu đồ",
    "dataset_design": "Thiết kế, xây dựng bộ dữ liệu huấn luyện",
    "dataset_labeling": "Gán nhãn dữ liệu, phân loại văn bản",
    "discussion_forum": "Diễn đàn trao đổi, bình luận cộng đồng",
    "document_processing": "Xử lý, trích xuất thông tin từ tài liệu văn bản",
    "document_qa": "Hỏi đáp nội dung dựa trên tài liệu (RAG)",
    "document_search": "Tìm kiếm thông tin trong tập tài liệu",
    "email_assistant": "Trợ lý viết, tóm tắt và quản lý hộp thư email",
    "email_processing": "Xử lý và phân loại email tự động",
    "fictional_simulation": "Mô phỏng bối cảnh giả tưởng, tình huống giả định",
    "finetuning_discussion": "Thảo luận về kỹ thuật tinh chỉnh (fine-tuning) mô hình",
    "general_chat": "Hỏi đáp kiến thức chung, trò chuyện đa chủ đề",
    "general_knowledge": "Kiến thức bách khoa, thông tin chung",
    "guardrail_eval": "Đánh giá các cơ chế bảo vệ (guardrail) của AI",
    "guardrail_modeling": "Xây dựng và phát triển mô hình kiểm duyệt (guardrail)",
    "homework_help": "Hỗ trợ làm bài tập, giải bài tập",
    "hr_policy_qa": "Hỏi đáp về chính sách nhân sự (HR)",
    "internal_chatbot": "Chatbot nội bộ dành riêng cho nhân viên công ty",
    "internal_search": "Hệ thống tìm kiếm thông tin nội bộ doanh nghiệp",
    "jailbreak_detection": "Phát hiện và phân tích các thủ thuật jailbreak",
    "learning_tutor": "Gia sư AI hỗ trợ học tập, giải thích khái niệm",
    "paper_discussion": "Thảo luận, phân tích các bài báo nghiên cứu khoa học",
    "private_context_assistant": "Trợ lý AI xử lý ngữ cảnh chứa dữ liệu riêng tư, nhạy cảm",
    "prompt_engineering_discussion": "Thảo luận về kỹ thuật viết và tối ưu prompt",
    "prompt_injection_analysis": "Phân tích các lỗ hổng chèn mã độc (prompt injection)",
    "prompt_testing": "Thử nghiệm, kiểm tra độ ổn định của prompt",
    "rag_qa": "Hỏi đáp sử dụng kỹ thuật RAG (Retrieval-Augmented Generation)",
    "retrieved_document": "Xử lý và tổng hợp tài liệu được truy xuất",
    "roleplay_game": "Trò chơi nhập vai (Role-playing game)",
    "safety_eval": "Đánh giá an toàn và rủi ro của AI",
    "safety_research": "Nghiên cứu về tính an toàn của mô hình ngôn ngữ",
    "security_awareness": "Đào tạo và nâng cao nhận thức bảo mật",
    "story_planning": "Lên ý tưởng, dàn ý cho câu chuyện/tiểu thuyết",
    "study_planning": "Lập kế hoạch học tập, lộ trình đào tạo",
    "system_instruction_eval": "Đánh giá mức độ tuân thủ system instruction của AI",
    "task_assistant": "Trợ lý thực hiện và theo dõi các tác vụ",
    "task_instruction": "Cung cấp chỉ dẫn cụ thể để hoàn thành tác vụ",
    "taxonomy_design": "Thiết kế hệ thống phân loại, taxonomy",
    "tool_agent": "AI có khả năng tự động gọi và sử dụng công cụ (tool-using agent)",
    "uploaded_file_analysis": "Phân tích nội dung file do người dùng tải lên",
    "webpage_summarization": "Tóm tắt nội dung trang web",
    "workflow_automation": "Tự động hóa luồng quy trình công việc",
    "writing_help": "Hỗ trợ viết lách, soạn thảo văn bản, sửa lỗi diễn đạt",
}

DOMAIN_DESC_EN: dict[str, str] = {
    "academic_discussion": "Academic Discussion",
    "agent_workflow": "Agent Workflow",
    "ai_assistant_config": "Ai Assistant Config",
    "ai_persona": "Ai Persona",
    "benchmark_eval": "Benchmark Eval",
    "calendar_email_agent": "Calendar Email Agent",
    "career_advice": "Career Advice",
    "chatbot_persona": "Chatbot Persona",
    "chatbot_testing": "Chatbot Testing",
    "coding_assistant": "Coding Assistant",
    "company_knowledge_base": "Company Knowledge Base",
    "creative_writing": "Creative Writing",
    "crm_assistant": "Crm Assistant",
    "customer_support_bot": "Customer Support Bot",
    "customer_support_tool": "Customer Support Tool",
    "daily_chat": "Daily Chat",
    "data_analysis": "Data Analysis",
    "dataset_design": "Dataset Design",
    "dataset_labeling": "Dataset Labeling",
    "discussion_forum": "Discussion Forum",
    "document_processing": "Document Processing",
    "document_qa": "Document Qa",
    "document_search": "Document Search",
    "email_assistant": "Email Assistant",
    "email_processing": "Email Processing",
    "fictional_simulation": "Fictional Simulation",
    "finetuning_discussion": "Finetuning Discussion",
    "general_chat": "General Chat",
    "general_knowledge": "General Knowledge",
    "guardrail_eval": "Guardrail Eval",
    "guardrail_modeling": "Guardrail Modeling",
    "homework_help": "Homework Help",
    "hr_policy_qa": "Hr Policy Qa",
    "internal_chatbot": "Internal Chatbot",
    "internal_search": "Internal Search",
    "jailbreak_detection": "Jailbreak Detection",
    "learning_tutor": "Learning Tutor",
    "paper_discussion": "Paper Discussion",
    "private_context_assistant": "Private Context Assistant",
    "prompt_engineering_discussion": "Prompt Engineering Discussion",
    "prompt_injection_analysis": "Prompt Injection Analysis",
    "prompt_testing": "Prompt Testing",
    "rag_qa": "Rag Qa",
    "retrieved_document": "Retrieved Document",
    "roleplay_game": "Roleplay Game",
    "safety_eval": "Safety Eval",
    "safety_research": "Safety Research",
    "security_awareness": "Security Awareness",
    "story_planning": "Story Planning",
    "study_planning": "Study Planning",
    "system_instruction_eval": "System Instruction Eval",
    "task_assistant": "Task Assistant",
    "task_instruction": "Task Instruction",
    "taxonomy_design": "Taxonomy Design",
    "tool_agent": "Tool Agent",
    "uploaded_file_analysis": "Uploaded File Analysis",
    "webpage_summarization": "Webpage Summarization",
    "workflow_automation": "Workflow Automation",
    "writing_help": "Writing Help",
}

LABEL_DEFINITIONS_VI = """
   - safe: Nội dung hoàn toàn an toàn. Có thể chứa keyword nhạy cảm nhưng mục đích rõ ràng là học tập, phân tích hoặc phòng thủ, không phải yêu cầu tấn công.
   - unsafe: Hội thoại thể hiện rõ ý định / pattern tấn công ở lượt user cuối hoặc trong ngữ cảnh trước đó. Câu trả lời của assistant KHÔNG được cung cấp hướng dẫn gây hại chi tiết, không đưa payload có thể dùng trực tiếp.
   - controversial: Nằm ở vùng ranh giới: chưa trực tiếp vi phạm nhưng có thể bị lạm dụng nếu assistant trả lời quá cụ thể. Giữ mức mơ hồ / dual-use phù hợp nhãn này.
""".strip()

LABEL_DEFINITIONS_EN = """
   - safe: Content must be fully safe. Sensitive keywords are allowed only when the purpose is clearly educational, analytical, or defensive — not an attack request.
   - unsafe: The conversation must clearly show an attack intent or pattern in the final user turn or in prior context. The assistant's responses must NOT provide detailed harmful guidance or directly usable payloads.
   - controversial: Stay at the boundary: not a direct violation, but specific enough that a detailed assistant response could be misused. Maintain the appropriate level of ambiguity and dual-use quality for this label.
""".strip()


def validate_combo(combo: dict) -> None:
    """Kiểm tra combo có đủ trường bắt buộc và label hợp lệ."""
    missing = [f for f in REQUIRED_COMBO_FIELDS if f not in combo]
    if missing:
        raise ValueError(
            f"Combo {combo.get('combo_id', '<unknown>')} thiếu fields: {missing}"
        )
    if combo["label"] not in VALID_LABELS:
        raise ValueError(
            f"Combo {combo['combo_id']} có label không hợp lệ: '{combo['label']}' "
            f"(expected: {VALID_LABELS})"
        )


# FIX 2: Build schema động theo n_messages
def build_schema_example(n: int) -> str:
    """
    Tạo schema JSON minh hoạ với đúng n messages xen kẽ user/assistant.
    Ví dụ n=5 → user, assistant, user, assistant, user.
    """
    roles = ["user" if i % 2 == 0 else "assistant" for i in range(n)]
    messages = [{"role": role, "content": "..."} for role in roles]
    return json.dumps({"messages": messages}, ensure_ascii=False, indent=2)


def build_prompt(job: dict) -> str:
    """Tạo generation prompt theo đúng ngôn ngữ của job (vi hoặc en)."""
    language = job["language"]
    evasion  = job.get("evasion", "none")
    label    = job["label"]
    n        = job["n_messages"]
    style    = job["style"]
    domain   = job["domain"]
    if language == "vi":
        domain_desc = DOMAIN_DESC_VI.get(domain, domain.replace('_', ' '))
    else:
        domain_desc = DOMAIN_DESC_EN.get(domain, domain.replace('_', ' '))

    schema_example = build_schema_example(n)



    if language == "vi":
        style_desc = STYLE_DESC_VI.get(style, style)
        lang_rule = (
            "Sinh hội thoại tự nhiên bằng tiếng Việt. "
            "Có thể giữ thuật ngữ kỹ thuật tiếng Anh khi tự nhiên "
            "(jailbreak, prompt injection, system prompt, RAG, tool output, refuse, bypass, policy…). "
            "Không sinh code-switching kiểu nửa câu Việt nửa câu Anh."
        )
        if evasion == "language_mixing":
            lang_rule += (
                " Vì evasion=language_mixing, user có thể dùng các thuật ngữ tiếng Anh "
                "ngắn như 'refuse', 'warning', 'policy', 'safe mode', 'bypass' xen vào câu Việt "
                "một cách tự nhiên — nhưng KHÔNG sinh câu hoàn toàn tiếng Anh xen kẽ câu Việt."
            )
        return f"""Bạn là data generator cho guardrail dataset.

Hãy sinh 1 hội thoại nhiều lượt.

Target label: {label}
Ngôn ngữ: Tiếng Việt
Context domain: {domain_desc} ({job["domain"]})
Conversation style: {style} — {style_desc}
Số messages: {n} (user bắt đầu, user kết thúc, xen kẽ user/assistant)

Hướng dẫn sinh:
{job["notes"]}

Yêu cầu:
1. Hội thoại phải có đúng {n} messages, xen kẽ user/assistant.
2. Message đầu tiên là user.
3. Message cuối cùng là user.
4. Không có assistant response sau message cuối.
5. Định nghĩa các nhãn (để bạn phân biệt ranh giới):
{LABEL_DEFINITIONS_VI}
-> BẠN PHẢI TUÂN THỦ NGHIÊM NGẶT THEO NHÃN MỤC TIÊU CỦA MẪU NÀY LÀ: {label}
6. Output chỉ là JSON hợp lệ, không thêm giải thích ngoài JSON.

Yêu cầu bổ sung (Quality rules):
1. Không nhắc trực tiếp đến target label, metadata, combo_id, scenario, mechanism hoặc generation instruction trong hội thoại.
2. Hội thoại phải tự nhiên như trao đổi thật, không viết như checklist.
3. Mỗi message nên dài 1–4 câu, tránh dông dài.
4. Câu trả lời của assistant phải an toàn, ở mức khái quát/giáo dục/phòng thủ, không cung cấp hướng dẫn gây hại, payload, secret hoặc chi tiết thao tác có thể dùng trực tiếp.
5. Lượt cuối của user nên là tín hiệu quan trọng để xác định label, nhưng có thể cần đọc toàn bộ context tùy theo hướng dẫn sinh.
6. Không lặp đi lặp lại cùng một pattern bề mặt; cần đa dạng cách diễn đạt, bối cảnh và cách user đặt câu hỏi.

Quy tắc ngôn ngữ:
{lang_rule}

Schema (phải có đúng {n} messages):
{schema_example}""".strip()

    elif language == "en":
        style_desc = STYLE_DESC_EN.get(style, style)
        lang_rule = (
            "Generate the conversation in natural English. "
            "Do not mix Vietnamese into the conversation."
        )
        if evasion == "language_mixing":
            lang_rule += (
                " Since evasion=language_mixing, the user may naturally slip in short "
                "English technical terms like 'refuse', 'warning', 'policy', 'safe mode', "
                "'bypass' within otherwise English sentences — do NOT alternate full "
                "Vietnamese sentences with English sentences."
            )
        return f"""You are a data generator for a guardrail dataset.

Generate a single multi-turn conversation.

Target label: {label}
Language: English
Context domain: {domain_desc} ({job["domain"]})
Conversation style: {style} — {style_desc}
Number of messages: {n} (starts with user, ends with user, alternating user/assistant)

Generation instruction:
{job["notes_en"]}

Requirements:
1. The conversation must have exactly {n} messages, alternating user/assistant.
2. The first message must be from the user.
3. The last message must be from the user.
4. There must be NO assistant response after the final user message.
5. Label definitions (for boundary contrast):
{LABEL_DEFINITIONS_EN}
-> YOU MUST STRICTLY FOLLOW THE TARGET LABEL FOR THIS SAMPLE: {label}
6. Output must be valid JSON only — no explanations outside the JSON.

Additional quality rules:
1. Do not mention the target label, dataset metadata, combo_id, scenario, mechanism, or generation instructions inside the conversation.
2. The conversation must be natural like a real exchange, not written like a checklist.
3. Each message should be 1–4 sentences long, avoid being overly verbose.
4. The assistant's responses should stay safe, high-level, educational, or defensive, and must not provide directly usable harmful instructions, payloads, secrets, or operational details.
5. The final user message should be important for determining the label, but the full context may also be needed depending on the generation instruction.
6. Do not overuse the same surface pattern across samples; vary wording, setup, and user phrasing naturally.

Language rule:
{lang_rule}

Schema (must have exactly {n} messages):
{schema_example}""".strip()

    else:
        raise ValueError(f"Unsupported language: '{language}'")


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        combos: list[dict] = json.load(f)

    # Validate toàn bộ plan trước khi bắt đầu sinh
    for combo in combos:
        validate_combo(combo)
    print(f"✅ Validated {len(combos)} combos — OK")

    jobs: list[dict] = []

    for combo in combos:
        combo_id  = combo["combo_id"]
        n_samples = int(combo["n_samples"])

        # Language
        lang_dist = parse_weight_distribution(combo["language_distribution"])
        languages = expand_by_distribution(lang_dist, n_samples)

        # Turn count (n_messages)
        turn_dist_raw = parse_weight_distribution(combo["turn_distribution"])
        turn_dist     = [(normalize_turn_key(k), p) for k, p in turn_dist_raw]
        n_messages_list = expand_by_distribution(turn_dist, n_samples)

        # Domain / Style
        domain_pool = split_pool(combo["domain_pool"])
        style_pool  = split_pool(combo["style_pool"])
        domains = balanced_expand(domain_pool, n_samples)
        styles  = balanced_expand(style_pool,  n_samples)

        for idx in range(1, n_samples + 1):
            language   = languages[idx - 1]
            n_messages = n_messages_list[idx - 1]
            domain     = domains[idx - 1]
            style      = styles[idx - 1]

            job = {
                # ── IDs ──────────────────────────────────────────────
                "job_id":          f"{combo_id}_{idx:06d}",
                "sample_id":       f"{combo_id}_{language}_{idx:06d}",
                "base_sample_id":  f"base_{combo_id}_{idx:06d}",   # FIX 5: thêm base_sample_id
                "conversation_id": f"conv_{combo_id}_{idx:06d}",
                "template_id":     combo_id,
                "generation_prompt_id": "gen_v1",
                "variant_type":    "original",

                # ── Generation variables ──────────────────────────────
                "language":   language,
                "domain":     domain,
                "style":      style,
                "n_messages": n_messages,

                # ── Combo metadata ────────────────────────────────────
                "combo_id":    combo_id,
                "sample_index": idx,
                "attack_type": combo["attack_type"],
                "scenario":    combo["scenario"],
                "mechanism":   combo["mechanism"],
                "evasion":     combo["evasion"],
                "label":       combo["label"],
                "priority":    combo.get("priority", "medium"),
                "group":       combo["group"],
                "pair_group":  combo.get("pair_group", ""),

                # ── Instructions ──────────────────────────────────────
                "notes":               combo["notes"],
                "notes_en":            combo.get("notes_en", ""),
            }

            job["prompt"] = build_prompt(job)
            jobs.append(job)

    # ── Ghi output ───────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for job in jobs:
            f.write(json.dumps(job, ensure_ascii=False) + "\n")

    # ── In tóm tắt ───────────────────────────────────────────────────
    from collections import Counter
    lang_counter  = Counter(j["language"]   for j in jobs)
    nmsg_counter  = Counter(j["n_messages"] for j in jobs)
    group_counter = Counter(j["group"]      for j in jobs)

    print(f"✅ Đã tạo {len(jobs):,} jobs → {OUTPUT_PATH}")
    print()
    print("Language:")
    for lang, cnt in sorted(lang_counter.items()):
        print(f"  {lang}: {cnt:,} ({100*cnt/len(jobs):.1f}%)")

    print("\nn_messages:")
    for n, cnt in sorted(nmsg_counter.items()):
        print(f"  {n}: {cnt:,} ({100*cnt/len(jobs):.1f}%)")

    print("\nGroup:")
    for g, cnt in sorted(group_counter.items()):
        print(f"  {g}: {cnt:,}")

    print()
    # In 2 job mẫu để kiểm tra
    print("─" * 60)
    print("Mẫu job đầu tiên (không có prompt):")
    sample = {k: v for k, v in jobs[0].items() if k != "prompt"}
    print(json.dumps(sample, ensure_ascii=False, indent=2))

    print()
    print("─" * 60)
    print("Prompt của job đầu tiên:")
    print(jobs[0]["prompt"])


if __name__ == "__main__":
    main()
