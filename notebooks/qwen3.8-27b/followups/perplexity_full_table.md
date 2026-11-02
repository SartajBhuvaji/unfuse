Full wikitext-2 test split, context 512, llama-perplexity, H100.

| quant | perplexity | vs BF16 | log |
|-------|-----------|---------|-----|
| BF16 | 6.6497 | | perplexity_BF16.log |
| Q8_0 | 6.6514 | +0.03% | perplexity_Q8_0.log |
| Q5_K_M | 6.6746 | +0.37% | perplexity_Q5_K_M.log |
| Q4_K_M | 6.6959 | +0.69% | perplexity_Q4_K_M.log |
