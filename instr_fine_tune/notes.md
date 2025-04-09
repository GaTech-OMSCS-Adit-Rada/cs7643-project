# Instruction Fine-tuning On Alpaca Dataset

| run_name | model_sz |batch_sz|n_epochs|opt|   lr  |min_lr|warmup|  wd  | loss  | perplex | gemini_eval|
|----------|----------|--------|--------|---|-------|------|------|------|-------|---------|------------|
|   run_1  | M-(355M) |    2   |   1    |adw|0.00005|  -   |  -   | 0.1  | 1.519 | 4.569   |   33.39    |
|   run_2  | L-(774M) |    2   |   1    |adw|0.00005|  -   |  -   | 0.1  | 1.591 | 4.911   |   33.06    |
|   run_3  |XL-(1558M)|    2   |   1    |adw|0.00005|  -   |  -   | 0.1  | 1.602 | 4.963   |   41.55    |
|   run_4  | L-(774M) |    2   |   1    |adw|0.00005| 1e-5 |  0.2 | 0.1  | 1.417 | 4.125   |   51.80    |
|   run_5  | M-(355M) |    2   |   1    |adw|0.00005| 1e-5 |  0.2 | 0.1  | 1.448 | 4.255   |   32.55    |
|   run_6  | L-(774M) |    2   |   1    |adw|0.00005| 1e-5 |  0.2 | 0.1  | 1.436 | 4.203   |   41.91    | (wo/ gradient clipping)


## Model Comparisons
Model Name: gpt2-medium (355M)
	Total Parameters: 406,286,336
	Total Memory Requirement: 1549.86 MB
    Pretrained score: 21.14


Model Name: gpt2-large (774M)
	Total Parameters: 838,359,040
	Total Memory Requirement: 3198.09 MB


Model Name: gpt2-xl (1558M)
	Total Parameters: 1,638,022,400
	Total Memory Requirement: 6248.56 MB
    Pretrained score: 18.58


* run_1
Ep 1 (Step 005925): Train loss 0.873, Val loss 1.529 Train perplexity 2.394, Val perplexity 4.612
Ep 1 (Step 005950): Train loss 0.834, Val loss 1.527 Train perplexity 2.302, Val perplexity 4.603
Ep 1 (Step 005975): Train loss 1.105, Val loss 1.519 Train perplexity 3.020, Val perplexity 4.569

Training completed in 12.76 minutes on A100

* run_2
- Training completed in 21.19 minutes on A100
- Overfitting observed after 0.5 epoch

* run_3:
- Training completed in 34.16 minutes on A100.
- Overfitting observed after 0.5 epoch

* run_4
- Training completed in 22.05 minutes.
- No overfitting happened, learning schedule helps!
- The model even got french translation correct!

* run_5
- Training completed in 13.37 minutes on A100
- Unlike run_4, this didnt get the french translation proving that you need a larger model to learn translations.

* run_6
- With gradient clipping removed. Seems like there is slight overfitting based on the increase in gaps btw. train and val loss from curves.
- The gemini eval is higher than no learning schedule but lower than learning_schedule+clipping. This suggests that clipping is important.

Notes:
* It seems like the models are overfitting (reasons may be small batch size, lack of diverse data, learning schedule, freeze some layers).
* It seems like the models are not learning any *new* info, e.g. km to meters. They are just changing
    how to spit out answer. That is reusing their pretrained base and then selecting a way to output.
    Thus having a very good pretrained model is very helpful.
* Using larger model
    - sometimes leads to worse generations: e.g. CL could not compose a tweet, XL could not spell friend.
        - Is this because of overfitting?
    - some things like turning km to meters can't even be done by XL
    - French translation only produced for XL.

* Evals are hard.

----------------------------------------------------------------------------------------------------------------------------------------------------------------

# Lora

| run_name | model_sz |batch_sz|n_epochs|opt|   lr  |min_lr|warmup| w_dcy| rank |alpha| loss  | perplex | gemini_eval|
|----------|----------|--------|--------|---|-------|------|------|------|------|-----|-------|---------|------------|
|   run_10 | M-(355M) |    2   |   1    |adw|0.00005|  -   |  -   | 0.1  |  16  |  16 |  |    |       |



To Try:
* Learning rate schedulers
* Optimizer type (SGD, AdamW)
* Epochs (1 vs 2)
* Lora for all layers vs last layers
* Lora params: rank and alpha


Resources:
* https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms
