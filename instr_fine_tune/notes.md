# Instruction Fine-tuning On Alpaca Dataset

| run_name | model_sz |batch_sz|n_epochs|opt|   lr  |min_lr|warmup|  wd  | loss  | perplex | gemini_eval|
|----------|----------|--------|--------|---|-------|------|------|------|-------|---------|------------|
|   run_1  | M-(355M) |    2   |   1    |adw|0.00005|  -   |  -   | 0.1  | 1.519 | 4.569   |   33.39    |
|   run_2  | L-(774M) |    2   |   1    |adw|0.00005|  -   |  -   | 0.1  | 1.591 | 4.911   |   33.06    |
|   run_3  |XL-(1558M)|    2   |   1    |adw|0.00005|  -   |  -   | 0.1  | 1.602 | 4.963   |   41.55    |
| **run_4**| L-(774M) |    2   |   1    |adw|0.00005| 1e-5 |  0.2 | 0.1  | 1.417 | 4.125   |   51.80    |
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
- With A100, 38.5/40 GB used, so can't increase batch size.

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

| run_name | model_sz |batch_sz|n_epochs|opt|   lr  |min_lr|warmup| w_dcy| rank |alpha| loss  | perplex | gemini_eval|  time   |
|----------|----------|--------|--------|---|-------|------|------|------|------|-----|-------|---------|------------|---------|
|  run_10  | M-(355M) |    2   |   1    |adw|0.00005| 1e-5 |  0.2 | 0.1  |  16  |  16 | 1.484 |  4.411  |  28.65     |T4(20.02)|
|  run_11  | L-(774M) |    2   |   1    |adw|0.00005| 1e-5 |  0.2 | 0.1  |  16  |  16 | 1.402 |  4.063  |  34.53     |T4(39.75)|
|----------|----------|--------|--------|---|-------|------|------|------|------|-----|-------|---------|------------|---------|
batch_sz, OOM with 8 
|  run_12  | M-(355M) |    4   |   1    |adw|0.00005| 1e-5 |  0.2 | 0.1  |  16  |  16 | 1.388 |  4.007  |  32.46     |T4(19.19)|
|----------|----------|--------|--------|---|-------|------|------|------|------|-----|-------|---------|------------|---------|
r, alpha
|  run_13  | M-(355M) |    2   |   1    |adw|0.00005| 1e-5 |  0.2 | 0.1  |  16  |  32 | 1.509 |  4.523  |  21.22     |T4(20.48)|
|  run_14  | M-(355M) |    2   |   1    |adw|0.00005| 1e-5 |  0.2 | 0.1  |  32  |  32 | 1.498 |  4.472  |  25.27     |T4(20.12)|
|  run_15  | M-(355M) |  **4** |   1    |adw|0.00005| 1e-5 |  0.2 | 0.1  |  64  |  64 | 1.844 |  6.319  |  14.80     |T4(20.70)|
|----------|----------|--------|--------|---|-------|------|------|------|------|-----|-------|---------|------------|---------|
Learning rate schedule
|  run_16  | M-(355M) |    4   |   1    |adw| 2.5e-5| 1e-5 |  0.3 | 0.1  |  64  |  64 | 1.470 |  4.350  |  28.14     |T4(20.70)|
|  run_17  | M-(355M) |    4   |   1    |adw| 3e-5  | 1e-5 |   0  | 0.1  |  64  |  64 | 1.560 |  4.760  |  31.53     |T4(20.70)|
|  run_18  | M-(355M) |    4   |   1    |adw| 1e-4  | 1e-5 |   0  | 0.1  |  64  |  64 | 3.974 |  53.202 |  -----     |T4(20.70)|
|----------|----------|--------|--------|---|-------|------|------|------|------|-----|-------|---------|------------|---------|
|  run_19  | M-(355M) |    4   |   2    |adw|0.00005| 1e-5 |  0.2 | 0.1  |  16  |  16 | 1.378 |  3.967  |  36.35     |T4(19.19)|


* **run_10**
- The generations are pretty similar to run_5 (full fine tuning)
- 7.9/15 GB used so I can probably double the batch size for this RAM size.

* **run_11**
- Here batch_size of 2 is the max. 10.9/15 Gb used.
- For the questions, "Suggest an alternate word for 'happy'.". run_4 says "joyful", but run_4 says "happy".
- Useful to generation comparisons for run_4 and run_11, it seems like two things are being done in finetuning: 1) instruction following, and 2) info learning.
    - In some questions, the lora finetuned model follows instruction, but does not output correct answer. Maybe it didnt learn info due to weights constraints.
- lora model completely fails in tweet query, suggesting lack of knowledge learning. Have to retry with higher rank to get more learning capacity.

* **run_12**
- With batch size 4, 9.7gb/15gb used and reached 14.2gb at one point. So this is max batch size.
- There was a wierd spike in the loss of the training data. Probably due to length mismatch leading to lot of padding tokens?

* **run_13**
- The performance is worse than run_10, so doubling the scaling factors does not help.
- This run also had a training loss spike in the middle. Hmm, this seems to be like lora thing.

* **run_14**
- Largely similar respones btw. run10 and run14. run_14 is slightly better, e.g. tweet composition is better than with run14.
- The gemini eval score is lower than run10, but I think its just due to randomness. It should be the same or slightly higher than run10
- It seems like training has not converged, so training for more epochs might help. I'll try 2 and 3.
- It seems like capacity is limiting factor. Like "Suggest an alternate word for 'happy'."
    - All lora models don't answer w/ "joyful" even though full models do.
    - This might be either due to capcity (so try higer rank) or not training for long enough (try 2 and 3 epochs)

* **run_15**
- There was a massive spike in loss and perplexity at exactly the time when peak_lr is reached. This suggests that new lora params changed too much.
- NExt action: try reduce peak_lr to 2.5e-5 and increase warmup to 30% for more smooth increase.
- Gemini conversation: https://g.co/gemini/share/34932dd66d19

- I'll use this run as a testbed for improving learning schedule for Lora as the effect is most pronounced with r=64.
    - Hopefully it translates to others as well

* **run_16**
- Inherits from run 15 and makes peak_lr to 2.5e-5 and increase warmup to 30% for more smooth increase/
- When I see the loss curve, there is still a spike, but now less pronounced than run15.
    - In addition, there is a smooth improvement in loss after the peak_lr, and very jagged curve before.
    - Thus, next time I'll only do cosine annealing and see what happens.

* **run_17**
- I completely turned off warmup and only did cosine annealing and got better performance (comparing gemini eval andnot final perplexity) than run 16
- What is surprising is that the spike still exists (this may indicate it may also be due to data batch.)
- Given that Raschaka's blog starts with learning rate of 0.1. I'll retry run16, but now starting with a learning rate of 1e-4

- This seems like conssine annealing only did not give good performance, let's go bak to run_12 and now try for more epochs. That is now run_19 is.

* **run_18**
- Horrible performance!

* **run_19**
- It seems like I have pulled all the juice out of lora for medium. Despite the perpleixty and loss falling, the generations are pretty similar to run_12.
- This might possibly be the best we can do with lora.
- It did worse on knowledge tasks like "correct word for friend", "synonym for happy", but it did better on cretive tasks (last half of generations).

To Try:
* Learning rate schedulers
* Optimizer type (SGD, AdamW)
* Epochs (1 vs 2)
* Lora for all layers vs last layers
* Lora params: rank and alpha


Resources:
* https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms

## Model Comparisons
* Model Name: gpt2-medium (355M) r=16
	Total Parameters: 406,286,336
	Total Memory Requirement: 1549.86 MB
Total trainable parameters before: 406,286,336
Total trainable parameters after: 0
Total trainable LoRA parameters: 7,898,384

* Model Name: gpt2-medium (355M) r=32
	Total Parameters: 406,286,336
	Total Memory Requirement: 1549.86 MB
Total trainable parameters before: 406,286,336
Total trainable parameters after: 0
Total trainable LoRA parameters: 15,796,768

* Model Name: gpt2-medium (355M) r=64
	Total Parameters: 406,286,336
	Total Memory Requirement: 1549.86 MB
Total trainable parameters before: 406,286,336
Total trainable parameters after: 0
Total trainable LoRA parameters: 31,593,536

* Model Name: gpt2-large (774M) r=16
	Total Parameters: 838,359,040
	Total Memory Requirement: 3198.09 MB
Total trainable parameters before: 838,359,040
Total trainable parameters after: 0
Total trainable LoRA parameters: 14,095,632
