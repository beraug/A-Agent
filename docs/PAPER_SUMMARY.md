# Paper Summary: Generative Agents (Park et al., UIST 2023)

Short reference for the three pillars of the architecture and why each matters.

## 1. Memory (Memory Stream)

- **What**: A time-ordered stream of natural-language observations. Each experience is stored as a timestamped record (e.g. "Agent woke up at 8:00").
- **Why it matters**: Without memory, the agent cannot maintain continuity or use past experiences to inform behavior. The paper shows through ablation that observation/memory is critical to believable behavior.

## 2. Reflection

- **What**: Periodically synthesize recent memories into higher-level reflections—beliefs, patterns, summaries (e.g. "Agent tends to wake early", "Agent believes X about Y").
- **Why it matters**: Reflection compresses experience into reusable knowledge and helps the agent form consistent identity and preferences. Ablation shows reflection contributes critically to believability.

## 3. Planning

- **What**: Use retrieved memories (and reflections) to plan behavior: daily schedule, next actions, and immediate reactions to events (e.g. react to "Someone knocked on the door").
- **Why it matters**: Planning ties memory to action. Without it, the agent would not produce coherent, goal-directed behavior. Ablation shows planning is essential.

## Checklist

- [ ] Memory: time-ordered stream of observations; retrieval uses recency, importance, relevance.
- [ ] Reflection: synthesize memories into higher-level reflections stored as special memories.
- [ ] Planning: retrieve memories to produce plans and reactions.
