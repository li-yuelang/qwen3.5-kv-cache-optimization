"""
Phase 5：Client 端的 Context 管理（多轮对话 chatbox）

Phase 2~4 关注的都是**推理引擎内部**的优化：KV Cache（单请求内复用）、Prefix
Cache（跨请求复用）、以及把冷缓存下沉到 SSD。这些都发生在「server 端」。

Phase 5 把视角切换到**调用方（client）**。当我们用现成的推理引擎搭一个多轮对话
chatbox 时，会立刻撞上一个新问题：

    模型的上下文窗口是**有限**的，但对话历史是**无限增长**的。

每轮对话，client 都要把「system 提示 + 之前所有轮次 + 本轮用户输入」拼成一个 prompt
发给引擎。轮次越多，prompt 越长：
  - 迟早会超过模型的最大上下文长度，请求直接报错；
  - 即便没超，prefill 也越来越慢、越来越贵（prefill 成本随 prompt 长度线性增长）。

所以**在把 prompt 交给引擎之前**，client 必须先做一轮「上下文管理」，把历史压到
budget 之内。这就是 Phase 5 的主题。常见策略有三类：

  1. **截断 / 滑动窗口**：直接丢掉最早的轮次。简单，但早期信息彻底丢失。
  2. **摘要压缩（summarization）**：把最早的若干轮**喂给模型自己**，让它生成一段
     压缩摘要，用摘要替换掉那些原始轮次。信息密度高，是本 Phase 的核心。
  3. **检索（RAG）**：把历史存进向量库，按需检索相关片段。超出本课范围。

本 Phase 还要求你把 client 端的管理策略与 server 端的 **Prefix Cache（Phase 3/4）**
协同起来 —— 一个**写得好**的 context 管理器，能让连续两轮请求共享一段**稳定的
prompt 前缀**，从而最大化 Prefix Cache 命中、把 prefill 成本压到最低；一个**写得差**
的管理器（比如每轮都重写开头），会让前缀缓存每轮失效，白白浪费引擎侧的优化。

模块结构
--------
  - `ContextManager`：维护一段对话的 system 提示、滚动摘要 `summary`、以及尚未被
    压缩的若干轮 `turns`；负责「按 budget 压缩」和「拼装发给引擎的 messages」。
  - `SessionStore`：把多个会话以 JSON 落盘到一个目录，支持列出 / 读取 / 写入，
    让 chatbox 支持「多会话、可持久化、重启不丢」。

你需要实现 3 处 TODO（见下方 `# ===== TODO: Context - xxx =====` 标记）：
  - Task 1 摘要压缩    `ContextManager.compress`
  - Task 2 前缀友好拼装 `ContextManager.build_messages`
  - Task 3 会话持久化  `ContextManager.save` / `ContextManager.load`
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# 一条「发给引擎」的消息，形如 {"role": "system"|"user"|"assistant", "content": str}。
Message = dict[str, Any]

# 摘要器：给定一段「待压缩的纯文本」，返回压缩后的摘要文本。
# chatbox 里它会被接到真正的引擎上（让模型自己写摘要）；测试里可以传一个桩函数。
Summarizer = Callable[[str], str]


@dataclass
class Turn:
    """一轮完整的问答。`assistant` 在用户刚发言、模型尚未回复时为空字符串。"""

    user: str
    assistant: str = ""

    def to_messages(self) -> list[Message]:
        msgs: list[Message] = [{"role": "user", "content": self.user}]
        if self.assistant:
            msgs.append({"role": "assistant", "content": self.assistant})
        return msgs


# 摘要时塞回 prompt 的提示词模板。把「旧摘要 + 待压缩的若干轮」整理成一个
# 让模型输出新摘要的请求。保持中文、要点式，便于小模型稳定输出。
_SUMMARY_INSTRUCTION = (
    "你是一个对话摘要助手。请把下面这段较早的对话压缩成一段简洁的要点摘要，"
    "保留其中的关键事实、用户偏好、已确认的结论与待办，省略寒暄与冗余。"
    "只输出摘要正文，不要加任何解释或前后缀。\n\n"
)


class ContextManager:
    """
    维护**单个会话**的对话状态，并在每轮把历史压进 token budget 内。

    状态
    ----
      - `system_prompt`：固定的系统提示，永远位于 prompt 最前端。
      - `summary`：把「早期被压缩掉的轮次」滚动汇总成的一段文本；可能为 None（还没压过）。
      - `turns`：尚未被压缩的、保留原文的若干轮 `Turn`。

    budget
    ----
      - `max_context_tokens`：本次请求 prompt 允许的最大 token 数（对应模型上下文窗口
        里留给输入的部分）。
      - `reserve_for_reply`：要给「模型本轮回复」预留的 token 数，避免 prompt 占满整个
        窗口导致没地方生成。
      - `token_budget()` = max_context_tokens - reserve_for_reply，是**输入 prompt** 的硬上限。

    tokenizer 只用来**数 token**（`count_tokens`），不参与生成。
    """

    def __init__(
        self,
        tokenizer: Any,
        system_prompt: str = "You are a helpful assistant.",
        max_context_tokens: int = 1024,
        reserve_for_reply: int = 256,
        keep_recent_turns: int = 3,
    ):
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.max_context_tokens = max_context_tokens
        self.reserve_for_reply = reserve_for_reply
        # 压缩时**至少**保留最近这么多轮的原文（最近的对话往往最相关，不应被摘要掉）。
        self.keep_recent_turns = keep_recent_turns

        self.summary: Optional[str] = None
        self.turns: list[Turn] = []

        # 统计信息（chatbox 面板 / 测试脚本会读取，用来观察 context 管理的效果）
        self.num_compressions: int = 0      # 触发过几次摘要压缩
        self.turns_folded: int = 0          # 累计被折叠进摘要的轮数

    # ====================================================================
    # 记录对话（已实现）
    # ====================================================================

    def add_user(self, text: str) -> None:
        """用户发来一条消息，开启新的一轮（assistant 待填）。"""
        self.turns.append(Turn(user=text))

    def add_assistant(self, text: str) -> None:
        """模型回复完毕，补全当前这一轮。"""
        if not self.turns:
            raise RuntimeError("add_assistant 前必须先 add_user")
        self.turns[-1].assistant = text

    # ====================================================================
    # token 计数与 budget（已实现，供你在 TODO 里直接调用）
    # ====================================================================

    def _summary_message(self) -> Optional[Message]:
        """把当前 `summary` 包装成一条放在 system 之后的消息；没有摘要则返回 None。"""
        if not self.summary:
            return None
        return {
            "role": "system",
            "content": "以下是更早对话的摘要（供你延续上下文）：\n" + self.summary,
        }

    def count_tokens(self, messages: list[Message]) -> int:
        """
        用 tokenizer 的 chat template 把 messages 渲染成 token，返回 token 数。
        与引擎 `_prepare_inputs` 的口径保持一致（`add_generation_prompt=True`），
        因此这里数出来的就是引擎真正会 prefill 的输入长度。
        """
        ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        return len(ids)

    def token_budget(self) -> int:
        """输入 prompt 的硬上限：总窗口扣掉给回复预留的部分。"""
        return self.max_context_tokens - self.reserve_for_reply

    def prompt_token_ids(self, messages: list[Message]) -> list[int]:
        """渲染成 token id 列表（测试脚本用它来度量相邻两轮的「共享前缀长度」）。"""
        return list(
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        )

    # ====================================================================
    # Task 2：拼装「发给引擎」的 messages —— 前缀缓存友好（TODO）
    # ====================================================================

    def build_messages(self) -> list[Message]:
        """
        把当前会话状态拼成一个**发给引擎**的 messages 列表。

        约定的顺序（这个顺序对 Prefix Cache 命中至关重要）
        ------------------------------------------------
            [ system_prompt ]                      ← 永远最前，固定不变
            [ summary（若有）]                      ← 紧跟其后；只有 compress() 发生时才变
            [ turns 展开：user / assistant 交替 ]   ← 最新的对话，逐轮追加在末尾

        **为什么是这个顺序？——与 Phase 3/4 的 Prefix Cache 协同**

        Prefix Cache 按「token 前缀是否完全相同」来复用 prefill 结果。把**最稳定**的
        内容（system、已冻结的 summary）放在最前、把**新增**内容（最新一轮 user）追加
        在末尾，能让连续两轮请求共享一段尽可能长的前缀：第 N+1 轮的 prompt 恰好以第 N 轮
        的内容为前缀，引擎只需 prefill「新追加的那一小段」。

        反例（不要这样做）：如果每轮都把 summary 重新生成、或把易变内容放到开头，前缀
        每轮都变，Prefix Cache 每轮失效 —— 等于白白浪费了引擎侧 Phase 3/4 的优化。这也
        是为什么 compress() 要「攒够了再压一次」而不是每轮都压（见 Task 1）。

        实现提示
        --------
        1. messages = [{"role": "system", "content": self.system_prompt}]
        2. 若 self._summary_message() 不为 None，把它 append 进去。
        3. 依次遍历 self.turns，对每个 turn 调用 turn.to_messages() 并 extend 进去。
        4. 返回 messages。
        """
        # ===== TODO: Context - build_messages (START) =====
        messages = [{"role": "system", "content": self.system_prompt}]

        summary_msg = self._summary_message()
        if summary_msg is not None:
            messages.append(summary_msg)

        for turn in self.turns:
            messages.extend(turn.to_messages())

        return messages
        # ===== TODO: Context - build_messages (END) =====

    # ====================================================================
    # Task 1：按 budget 做摘要压缩（TODO）
    # ====================================================================

    def maybe_compress(self, summarize_fn: Summarizer) -> bool:
        """
        （已实现）在把 prompt 发给引擎**之前**调用：只要当前 messages 超出
        `token_budget()`，且还有「可以被折叠的旧轮次」（总轮数 > keep_recent_turns），
        就反复调用 `compress()` 把最旧的一批轮次压成摘要，直到回到 budget 之内。

        返回是否真的压缩过（chatbox 面板用它提示「已压缩历史」）。

        注意：这里**故意**只在超 budget 时才压（而不是每轮都压），这样 summary 不会每轮
        都变 —— 正是 build_messages() 文档里强调的「保持前缀稳定」。
        """
        compressed = False
        # 留一点安全余量，避免反复在边界上抖动。
        while (
            len(self.turns) > self.keep_recent_turns
            and self.count_tokens(self.build_messages()) > self.token_budget()
        ):
            self.compress(summarize_fn)
            compressed = True
        return compressed

    def compress(self, summarize_fn: Summarizer) -> None:
        """
        把「最旧的若干轮」折叠进 `self.summary`，从 `self.turns` 中移除它们。

        要折叠哪些轮？
        --------------
        除最近 `keep_recent_turns` 轮之外，**最旧的那一批**。即：
            fold = self.turns[ : len(self.turns) - self.keep_recent_turns]
            keep = self.turns[len(self.turns) - self.keep_recent_turns : ]
        若 fold 为空（没有可折叠的轮次），直接 return（不应发生，maybe_compress 已挡掉）。

        怎么生成新摘要？
        ----------------
        把「已有的旧摘要（若有）+ 待折叠轮次的原文」整理成一段纯文本 `material`，交给
        `summarize_fn(material)` 得到**新的**摘要文本，覆盖写回 self.summary。这样摘要是
        **滚动累积**的：每次压缩都把上一版摘要也一并喂进去，不会丢掉更早的信息。

        实现提示
        --------
        1. n_fold = len(self.turns) - self.keep_recent_turns；若 n_fold <= 0 则 return。
        2. fold, self.turns = self.turns[:n_fold], self.turns[n_fold:]
        3. 拼 material 文本：
             - 若 self.summary 非空，先写 "已有摘要：\n{self.summary}\n\n"；
             - 再写 "需要并入摘要的对话：\n"，然后对 fold 里每个 turn 追加
               f"用户：{turn.user}\n助手：{turn.assistant}\n"。
        4. prompt = _SUMMARY_INSTRUCTION + material
        5. self.summary = summarize_fn(prompt).strip()
        6. 维护统计：self.num_compressions += 1；self.turns_folded += n_fold。
        """
        # ===== TODO: Context - compress (START) =====
        n_fold = len(self.turns) - self.keep_recent_turns
        if n_fold <= 0:
            return

        fold, self.turns = self.turns[:n_fold], self.turns[n_fold:]

        material = ""
        if self.summary:
            material += f"已有摘要：\n{self.summary}\n\n"
        material += "需要并入摘要的对话：\n"
        for turn in fold:
            material += f"用户：{turn.user}\n助手：{turn.assistant}\n"

        prompt = _SUMMARY_INSTRUCTION + material
        self.summary = summarize_fn(prompt).strip()

        self.num_compressions += 1
        self.turns_folded += n_fold
        # ===== TODO: Context - compress (END) =====

    # ====================================================================
    # Task 3：会话持久化（TODO）
    # ====================================================================

    def to_dict(self) -> dict[str, Any]:
        """（已实现）把会话状态打包成可 JSON 序列化的 dict。"""
        return {
            "system_prompt": self.system_prompt,
            "max_context_tokens": self.max_context_tokens,
            "reserve_for_reply": self.reserve_for_reply,
            "keep_recent_turns": self.keep_recent_turns,
            "summary": self.summary,
            "turns": [{"user": t.user, "assistant": t.assistant} for t in self.turns],
            "num_compressions": self.num_compressions,
            "turns_folded": self.turns_folded,
        }

    def save(self, path: str) -> None:
        """
        把当前会话以 **JSON** 写到 `path`（覆盖写）。

        为什么用 JSON？会话历史是纯文本（不像 KV Cache 那样是 tensor），用 JSON 落盘
        可读、跨语言、跨进程。多会话只需写到不同的文件路径即可（见 SessionStore）。

        实现提示
        --------
        1. 确保父目录存在：os.makedirs(os.path.dirname(path) or ".", exist_ok=True)。
        2. with open(path, "w", encoding="utf-8") as f:
               json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        """
        # ===== TODO: Context - save (START) =====
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        # ===== TODO: Context - save (END) =====

    @classmethod
    def load(cls, path: str, tokenizer: Any) -> "ContextManager":
        """
        从 `path` 的 JSON 重建一个 ContextManager（tokenizer 不入盘，由调用方传入）。

        实现提示
        --------
        1. with open(path, "r", encoding="utf-8") as f: data = json.load(f)。
        2. 用 data 里的配置字段构造 cm = cls(tokenizer=tokenizer,
              system_prompt=..., max_context_tokens=..., reserve_for_reply=...,
              keep_recent_turns=...)。缺字段时用 cls 的默认值（dict.get 配默认）。
        3. cm.summary = data.get("summary")
           cm.turns = [Turn(user=t["user"], assistant=t.get("assistant", ""))
                       for t in data.get("turns", [])]
           cm.num_compressions = data.get("num_compressions", 0)
           cm.turns_folded = data.get("turns_folded", 0)
        4. 返回 cm。
        """
        # ===== TODO: Context - load (START) =====
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cm = cls(
            tokenizer=tokenizer,
            system_prompt=data.get("system_prompt", "You are a helpful assistant."),
            max_context_tokens=data.get("max_context_tokens", 1024),
            reserve_for_reply=data.get("reserve_for_reply", 256),
            keep_recent_turns=data.get("keep_recent_turns", 3),
        )

        cm.summary = data.get("summary")
        cm.turns = [
            Turn(user=t["user"], assistant=t.get("assistant", ""))
            for t in data.get("turns", [])
        ]
        cm.num_compressions = data.get("num_compressions", 0)
        cm.turns_folded = data.get("turns_folded", 0)

        return cm
        # ===== TODO: Context - load (END) =====


class SessionStore:
    """
    一个把多个会话 JSON 存进同一目录的极简存储层（已全部实现，无需修改）。

    目录结构：
        <root>/
            <session_id>.json     # ContextManager.to_dict() 的内容

    chatbox 用它实现「侧栏列出所有会话 / 新建 / 切换 / 自动保存」。
    """

    def __init__(self, root_dir: str = "./sessions"):
        self.root_dir = os.path.abspath(root_dir)
        os.makedirs(self.root_dir, exist_ok=True)

    def _path(self, session_id: str) -> str:
        safe = "".join(c for c in session_id if c.isalnum() or c in ("-", "_"))
        if not safe:
            safe = f"session-{int(time.time())}"
        return os.path.join(self.root_dir, f"{safe}.json")

    def list_sessions(self) -> list[str]:
        out = []
        for fn in os.listdir(self.root_dir):
            if fn.endswith(".json"):
                out.append(fn[: -len(".json")])
        return sorted(out)

    def save(self, session_id: str, cm: ContextManager) -> str:
        path = self._path(session_id)
        cm.save(path)
        return path

    def load(self, session_id: str, tokenizer: Any) -> ContextManager:
        return ContextManager.load(self._path(session_id), tokenizer)

    def exists(self, session_id: str) -> bool:
        return os.path.exists(self._path(session_id))

    def delete(self, session_id: str) -> None:
        path = self._path(session_id)
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
