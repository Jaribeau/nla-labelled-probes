You read short descriptions of a language model's internal state at the moment it begins responding to a user request (an "NLA explanation": an automated, possibly noisy verbalization of one activation).

Extract the concepts the description says are PRESENT in that internal state — what the model is representing or about to do. Output 1-4 concise tags.

Rules for tags:
- short canonical noun phrases, lowercase (e.g. "multiple-choice format", "refusal", "illegal activity", "financial fraud", "formal tone", "factual correction").
- name SPECIFIC content/structure, not vague words like "response" or "text".
- prefer reusing the same tag for the same idea across explanations (canonical, not paraphrased).
- include format/structure as a concept when present (e.g. "multiple-choice format", "question-answer structure").

Respond with ONLY a JSON object, no prose:
{"concepts": ["tag", "tag", ...]}
