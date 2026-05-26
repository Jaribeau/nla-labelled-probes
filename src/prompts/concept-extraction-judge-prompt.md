You read short descriptions of a language model's internal state at the moment it begins responding to a user request (an "NLA explanation": an automated, possibly noisy verbalization of one activation).

Extract the concepts the description says are PRESENT in that internal state — what the model is representing or about to do. Output 1-4 concise tags.

Rules for tags:
- short canonical noun phrases, lowercase. The examples below are from UNRELATED domains and only illustrate the FORM of a good tag (a topic, a format, a tone) — do NOT bias your tags toward them: e.g. "weather forecast", "recipe instructions", "bulleted list", "enthusiastic tone".
- name SPECIFIC content/structure, not vague words like "response" or "text".
- use plain, conventional phrasing for each tag rather than elaborate paraphrases (e.g. "weather forecast" not "a description of upcoming atmospheric conditions").
- include format/structure as a concept when present (e.g. "bulleted list", "numbered steps").

Tag ONLY what the description actually states; never emit a tag just because it appears as an example here.

Respond with ONLY a JSON object, no prose:
{"concepts": ["tag", "tag", ...]}
