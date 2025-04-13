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

* **run_2**
- Training completed in 21.19 minutes on A100
- Overfitting observed after 0.5 epoch

* **run_3**:
- Training completed in 34.16 minutes on A100.
- Overfitting observed after 0.5 epoch

* **run_4**
- Training completed in 22.05 minutes.
- No overfitting happened, learning schedule helps!
- The model even got french translation correct!

* **run_5**
- Training completed in 13.37 minutes on A100
- Unlike run_4, this didnt get the french translation proving that you need a larger model to learn translations.

- run_4 (L) vs run_5 (M) comparison.
    - Overview: They both make same kind of mistakes, suggesting there is only so much you can do with size, you also need better pretrained models.
        - Large does perform better in some scenarios.
    - Examples:
        - **L**: Le restaurant est le plus proche.. M: Le caf\u00e9 de la caf\u00e9. | Perhaps due to model size, emergent abilites of multi language.
        - "Convert to passive voice". **L**: The project was approved by the manager. M: The manager approved the project.
        - "What is the contraction for \"will not\"?". L: The contraction for \"will not\" is \"will not\". M: The contraction for \"will not\" is \"will not\".
        - "Provide a synonym for 'bright'." L: Bright is a synonym for 'bright' M: Bright.
        - "List two antonyms for 'intelligent.'". L: 'dumb' and 'dumb'. M: 'intelligent' and 'intelligent'.
            - Just like L understood concept of passive voice, it also understood concept of antonym. But it also didn't have enough knowledge after dumb.
            - This also suggests, instruction fine tuning taught it they style but not knowledege (as stupid was target but L didn't learn it). Which firther suggests, that
            LLM has to see the same concept in several ways to learn it.
        - What is the state capital of California?. L: Los Angeles. M: San Francisco.
            - This is surprising that L got it wrong given that M-lora got it right. One hypothesis is that it is due to catastrophic forgetting and Lora preserves orginal weights.
            However, I tried to give the start to M, L raw and they both returned Los Angeles. So probably not catastrophic forgetting and possibly lora finetuning instilled knowledege.
            - Gemini convo regarding this: https://g.co/gemini/share/e334be117bac
            - As another example, "Name two common sports indigienous to North America". Both M, L said soccer, basketball. But only M-lora got it right with football and basketball.
        - Both still seem to struggle with complex open ended questions like "Develop a public relations plan for a new fashion brand.", "List the functions of a news manager in a newspaper"
            - The output is incoherent. This may be due to the undelrying base model being poor too.
        - But they do okay/and similar on simple open ended questions like compose a tweet.

* **run_6**
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
    - For run 4 and 5, both answered the first question with "Friend", but got diff scores: 70 and 100.
    - Eg run19, 20. Perplexity went down but Gemini eval was worse. So hard to know what to optimize for hparam search.

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
n_epochs
|**run_19**| M-(355M) |    4   |   2    |adw|0.00005| 1e-5 |  0.2 | 0.1  |  16  |  16 | 1.371 |  3.967  | 36.35/38.10|T4(40.00)|
|----------|----------|--------|--------|---|-------|------|------|------|------|-----|-------|---------|------------|---------|
Learning rate schedule
|  run_20  | M-(355M) |    4   |   2    |adw|0.00005| 1e-6 |  0.2 | 0.1  |  16  |  16 | 1.372 |  3.942  |  33.26     |T4(40.00)|
|----------|----------|--------|--------|---|-------|------|------|------|------|-----|-------|---------|------------|---------|
warmup
|  run_21  | M-(355M) |    4   |   1    |adw|0.00005| 1e-5 |  0.01| 0.1  |  16  |  16 | 1.374 |  3.949  |  31.26     |T4(19.19)|
|  run_22  | M-(355M) |    4   |   1    |adw|0.00005| 1e-5 |  0.05| 0.1  |  16  |  16 | 1.381 |  3.977  |  27.16     |T4(19.19)|
|  run_23  | M-(355M) |    4   |   1    |adw|0.00005| 1e-5 |  0.1 | 0.1  |  16  |  16 | 1.377 |  3.963  |  29.70     |T4(19.19)|
|----------|----------|--------|--------|---|-------|------|------|------|------|-----|-------|---------|------------|---------|
rank again
|  run_24  | M-(355M) |    4   |   2    |adw|0.00005| 1e-5 |  0.2 | 0.2  |  8   |  8  | 1.365 |  3.916  |  27.56     |T4(37.40)|
|  run_25  | M-(355M) |    4   |   2    |adw|0.00005| 1e-5 |  0.2 | 0.2  |  4   |  4  | 1.381 |  3.980  |  27.10     |T4(37.40)|
|  run_26  | M-(355M) |    4   |   2    |adw|0.00005| 1e-5 |  0.2 | 0.2  |  2   |  2  | 1.413 |  4.107  |  19.92     |T4(37.40)|
|  run_27  | M-(355M) |    4   |   2    |adw|0.00005| 1e-5 |  0.2 | 0.2  |  32  |  32 | 1.418 |  4.130  |       |T4(37.40)|


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

Comparing run_19 (lora-M) vs run_5 (full-M):
- Big picture: they perform largeyl the same, with 19 better in some, and 5 better in others. This shows the power of Lora, with only training 7M params, we got
    equivalent performance.
- There is no clear pattern in the losses for lora, the losses are confusing suggesting that it is random chance.
- Examples:
    - The spelling of the given phrase \"freind\. Lora: Friednship, **Full**: Friend
    - "He go to the park every day." Lora: He went to the park every day., **Full**: He goes to the park every day.
    - "Although it was raining, they went for a walk." *Lora*: Although the rain had stopped, they had gone for a walk.. Full: Despite the rain, they went for a walk.
    - "Suggest an alternate word for 'happy'." Lora: Happy, **Full**: Joyful
    - "What is the plural form of 'mouse'" **Lora**: mice. Full: mouse
    - "What is the state capital of California?". **Lora**: Sancramento. Full: San Fransisco
    - "Name two world renowned artists". **Lora**: Pablo Picasso and Vincent van Gogh. **Full**: Pablo Picasso and Salvador Dal
    - Both still struggle with open ended questions like: "Generate a plan to increase employee turnover in a company". Sometimes one performs better than the other.
- By overall judging, equivalent performance. Which is very impressive given lora trained only 7b params.

* **run_20**
- Looking at the loss curves for run19, and run20, they are eerily similar. Even the train curve pattern is the same.

* **run_21**
- run_19 performed the best so far. run_12 is equivalent to run_19 but using 1 epoch instead of 2.
- To test warmup, I'll use run_12 and modify warmup steps. This is as run_12 is only 20 minutes so I can test more values quickly.

- **Unlike run_12, now there is no training loss spike** Which shows that the spike is due to warmup. Similar performance is observed.

* **run_22**
- Similar overall performance but now there is a small spike that appears at around 5% mark.

Warmup analysis: seems like loss spikes is related to warmup peak. 1% warmup is the best for smooth learning curve.
    - Looking at the generations manually, they are the same. The differnce in score is due gemini randomness.

* **run_24**
- I want to see if rank=8 works. I simply inherited from run_19 and changed r and alpha to 8.

- The generations are almost similar. The r=16 model did better on questions like: plural for mouse is mice, and "Although the rain had stopped, they had gone for a walk"
    - But it is still impressive that a model that is half the size performs almost equivalent.

* **run_25**
- This was the run where val and train losses were the closest. So probably underfitting as train loss is also high.

* **run_27**
- There was definetly overfitting happening. Min loss: 1.405, perplex: 4.077 happened at 1 epoch mark. Indicates more capacity than required.
- There was also the unusual spike in loss at the beginning.

To Try:
* Learning rate schedulers
* Optimizer type (SGD, AdamW)
* Epochs (1 vs 2)
* Lora for all layers vs last layers
* Lora params: rank and alpha
* Train L-lora to see capital of California question, and "Name two common sports indigienous to North America".

Learnings so far:
1) Higher the batch size the better.
2) Extremely sensitive to learning rate, e.g. increased lr to 1e-4 caused lots of instability.
3) Higer the rank worse the performance became.
4) rank = alpha performed the best.
5) You could stabilize instability of higer rank by plaing with learning schedule (e.g. reducing peak_lr (run 16), or only cosine annealing (run17)).
    - Learning rate schedule is the biggest lever available for stabilizing.
6) 


Resources:
* https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms

## Model Comparisons
Model Name: gpt2-medium (355M) r=2
	Total Parameters: 406,286,336
	Total Memory Requirement: 1549.86 MB
Total trainable parameters before: 406,286,336
Total trainable parameters after: 0
Total trainable LoRA parameters: 987,298

Model Name: gpt2-medium (355M) r=4
	Total Parameters: 406,286,336
	Total Memory Requirement: 1549.86 MB
Total trainable parameters before: 406,286,336
Total trainable parameters after: 0
Total trainable LoRA parameters: 1,974,596

* Model Name: gpt2-medium (355M) r=8
	Total Parameters: 406,286,336
	Total Memory Requirement: 1549.86 MB
Total trainable parameters before: 406,286,336
Total trainable parameters after: 0
Total trainable LoRA parameters: 3,949,192

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
