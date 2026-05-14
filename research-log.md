# Research Log

---

**Instructions**

- New dated entries go at the top. 
- Don't restructure old entries.
- Be concise.

---

### May 14, 2026 — Getting setup

This paper was just released, and opens up an interesting path to look at eval awareness:
https://transformer-circuits.pub/2026/nla/index.html
tldr of that: NLAs (natural language autoencoders) output a text representation of the models latent representation. 
Here’s a demo: https://www.neuronpedia.org/llama3.3-70b-it/nla

The idea with this research:

1. Reproduce Debunova's eval awareness format confound paper (her paper shows that those eval-awareness linear probes respond to prompt format, not the content)
2. Extend it, exposing the same format confound using NLAs
3. Then, if that works well, explore using the output of an NLAs to generate labels to train better linear probes 
- NLA inference is very expensive; however, probes trained off NLA output could be usable production/further research.
- Most likely source of this part to fail is that current NLAs provide a noisy/uncalibrated signal. Step 2 should help get a sense of this. 
- If it’s too noisy, a more complex project (that anthropic is likely already working on) is to train a better NLA (using more than 1 activation layer as input). This would have to be a proof of concept on a small model, as this would be extremely expensive to train.

1 and 2 derisk things. 3 gets much more involved; and potentially much more interesting.
