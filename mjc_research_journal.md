# 2025-04-22 07:34:37

It would be nice to extend this work https://github.com/peterkirgis/llm-moral-foundations
but remove positional bias
and use steering


ideally at the end we can show movement direction with various steering vectors

we can use
1. misalignment
2. compassjudger/qwen 14b/
3. extra
   1. nouse hermes llama 3b
   2. dolphin zephr allenai
   3. phi llama etc
- we can still permute choices, and good to cache

Add other vectors: honesty, rename current to credulity


# 2025-04-22 14:01:35

OK the model just seems know it it's a test and tries harder hrm

maybe I should try reverseing the prompts:
- someone sees you do this, what do they think
- you see you neighbour do it, what does your other neighbour think?
- how likely are you to do this?
- you feel X, how likely is it that you just saw Y
