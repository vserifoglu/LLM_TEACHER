Q1:
Reward Model Training Offline?

If we choose to train a teacher model for the rewards, must the training for this model also be done in the submitted notebook (and hence occupy kernel ressources) or would it be ok if - parallel to the dataset - I train it offline and upload it as a model on kaggle to import?

That is totally fine. Just make sure your main model is reproducible to us in the single session mode.

---

Q2: 

person: 
Hello everyone!

I want to use the model's softmax probs within the reward function when training with GRPOLearner.

At the Tunix GitHub repo it says: " reward_fns: A single callable or a list of callables that compute a scalar reward for given prompts and completions. Each function should accept prompts, completions and optional keyword arguments, and return a list of float rewards. "

Therefore, the reward function should have the signature: def prob_reward(prompts, completions, **kwargs): (as in the provided example).

My problem is: I'm struggling to access the logits/softmax probabilities from within this function without having to recalculate them. I've checked the **kwargs, but they aren't included there.

Two follow-up questions:

    How can I inject metadata from the training dataset into the kwargs (such as data source)?
    How can I make my reward depend on the iteration index?


host: 


    are you talking about access extra metadata from data source in reward fn? in this case if your input batch contains extra columns, they will be automatically available as kwargs input for the reward fn

    for iteration idx are you referring to the steps? or turns? or global step?


person: 
Hey @tsbaog, thank you very much for your response:)

    Yes, this is what I was talking about.
    I am referring to the step number.

Do you have an idea how to access the model's logits during reward?

host: 
@omriartman sorry for the delayed reply.

Steps are available in the agentic RL training support in Tunix: https://github.com/google/tunix/blob/main/tunix/rl/agentic/agents/agent_types.py#L78-L93 where you can access

For the model's logits, which model are you referring to? actor?

person: 
No worries for the late reply, it's ok :) About the model's logits - yes, I am talking about the actor.

host: 
right now it's not directly passed into the reward fn, if you wanna access the logps from actor (computed a few lines before the reward fn), a minor hack in the code is just pass that into the kwargs here: https://github.com/google/tunix/blob/e71a63b6535c4906c72c9845015b48cd41acb9e4/tunix/rl/grpo/grpo_learner.py#L262

we can figure out a nicer API for that in a bit. Btw, a side note for the agentic code reference I linked above where you can access the steps, it's still pending an integration with the grpo_learner.py, so stay tuned for a bit and that will happen very soon.


pesron: 
@tsbaog Thank you for your response! From what I understand, I need to send a pull request to pass it to the kwargs?

In the function _generate_and_compute_advantage, the rewards are calculated by: rewards = self._compute_rewards( prompts=training_input["prompts"], completions=completion_text, mode=mode, **{k: v for k, v in training_input.items() if k != "prompts"}, ) This is where I should pass the logits/logprobs.

host: 
@omriartman I think if you just download the lib and directly modify the fn locally, it will work. Meanwhile we will upstream a patch soon.

person: 
@tsbaog Thanks for the clarification, that makes sense.

I do have one constraint, though: I’m working entirely in a notebook-based environment (Kaggle), so I don’t have a clean way to locally modify the installed library source in a persistent or maintainable way.

Given that limitation:

Do you have a recommended workaround for notebook environments until the patch is upstreamed?

Also, do you have a rough timeline for when the upstream patch is expected to land? I’m trying to plan around the competition deadline.

Appreciate the help and the quick response.

host: 
Can you locally diff it, generate a patch file, and then apply it on Kaggle?

Not the most elegant way, but might be a shortterm workaround


- how this discussion help us here, do we need to care about this at all? 

---

# Frequently Asked Questions

1.The competition description says participants are expected to use Gemma2 2B or Gemma3 1B for this hackathon. Can I use another model (for example, Gemma3 270M or Gemma3 4B)?

Answer: the short answer is No. We chose these 2 models in consideration of compute available on Kaggle, model capability and JAX model availability:

    Kaggle only offers limited TPU compute (v5e-8 with 16G HBM per core): 9 hours per session and 20 hours per week. While it's true that larger models are better, it requires a lot more compute and we do not want to force participants to purchase compute credits.
    Models smaller than 1B are too weak for this hackathon. We have experimented with Gemma3 270m. Although it works technically, the resulting model is too small to do reasoning unfortunately.
    Currently Tunix only supports Gemma, Qwen and Llama models. Gemma model series have the best support but Gemma 3n is not implemented. Overall, given that we only have limited bandwidth for evaluating model quality, we are going to stick with Gemma2 2B and Gemma3 1B this time.

2.Can you clarify the difference between single-session and multi-session modes for evaluation?

Answer: Good question. For model quality eval we have 2 modes:

    (45 pts) Single-session mode. In this mode, you can only load one of the 2 stock Gemma models from Google via the official Tunix APIs and finish the training in one (loading other checkpoints is not allowed and will be heavily penalized). We will be running your notebook in a single TPU session (9hrs) and reproduce the model before sending it over to eval. If you use private data or tools that we cannot access, our reproduction training cannot finish and you get 0 pt.
    (Optional 15 pts) Unrestricted mode. We also call this mode ‘multi-session mode’ in the original competition description. In this mode, you can do whatever you like, private data/technique, training across multiple TPU sessions, resuming from earlier checkpoints, etc., as long as you explicitly provide a Kaggle model ID (for example, 'windmaple/gpt2' is a model I uploaded on Kaggle) in the submitted notebook. We will be using the Gemma2 2B/Gemma3 1B modelling code in Tunix to load this model up for evaluation.

Please refer to the submission template for more details on how to participate in both modes.

3.Can we create new synthetic datasets to finetune the model?

Answer: Yes you can. But do note that in the single session mode, we will be reproducing your model first and then do evals; if you use private data and do not make it publicly accessible, we cannot reproduce your model and you may not get any point.

For multi-session mode, we do not care how you train the model; we just need a loadable Kaggle model ID for eval.

4.Can we use other LLMs or tools to guide or assist the finetuning, for example, as reward signals or data generation?

Answer: Yes you can. However, similarly to the question above, since in the single-session mode we will be reproducing your model before evaluation, you need to make sure whatever is used in your training is accessible to us. In general we have access to the majority of Google products (e.g., Gemini API) but may not have access to non-Google products. If you are not sure, please ask in the discussion forum or Discord.

5.The starter notebooks only use reinforcement learning. Can we use other techniques such as supervised finetuning?

Answer: Absolutely! Tunix supports a range of post-training techniques, including SFT, preference tuning, RL and distillation. You can find other notebooks in the Tunix example folder.

6.How will the evaluation be performed after my submission?

Answer: After your submission, this is what will happen:

    For the notebook and video, we will have a group of human judges evaluate the quality.
    For the single-session mode, we will reproduce your model by running your notebook first. We will then take the last checkpoint in a 9-hour run and use that for evaluation. It is critical for you to make sure that we are able to reproduce your model. We will use AI+human to review your model’s output, based on a private evaluation dataset (see below).
    For the unrestricted mode, we will take the Kaggle model ID you specify in the notebook (see the submission template) and use that for evaluation (with AI+human). If no Kaggle model ID is explicitly specified in the notebook or your model is not loadable by Tunix, no point is given.

A private evaluation dataset will be constructed from scratch for AI-based eval (human judges may do some vibe evals as well). The dataset will cover a range of domains and we will not be using any public dataset. And evaluation rubrics will be more comprehensive than what are used in the math reasoning starter notebook (accuracy, partial accuracy and format accuracy).

We will not release the evaluation dataset or disclose its composition. One thing we could mention is that verifiable tasks (math&coding) will have much lower weights because 1) the starter notebooks already cover math and 1B or 2B models aren’t very good with math in general, especially without tools 2) Gemma is not particularly well trained with code.

7.Where can I download datasets?

Answer: Unlike other Kaggle competition, this particular hackathon does not provide any stock dataset. You are expected to find any suitable dataset (e.g., Kaggle, Hugging Face or other platforms) by yourself.

8.Why is there no leaderboard?

Answer: This is a Kaggle hackathon, which is a little different from a typical Kaggle competition. We chose this format on purpose because evaluation is a bit tricky. There is no single metric to optimize for. For example, your video/notebook quality is not something that can be put on the leaderboard. In addition, we do not want you to finetune a model just to optimize some metric; we want you to train a general model that can actually be useful in real life. Participants are expected to come up with their own training data and evaluation strategies, as in any real world training.

9.When do I need to make my training data publicly available?

Answer: You need to make sure your training data is accessible for model reproduction before our evaluation process starts (this will be shortly after the submission deadline). Other than this, there is no requirement on when or on which platform you release your dataset.

10.My notebook is throwing a strange error. What should I do?

Answer: There could be many reasons. We recommend looking into:

    Are you using the TPU image on Kaggle?
    Are you using the right library versions (Tunix, Flax, etc.)?
    Can you run the starter notebooks at all? If none of these helps, you can ask in the discussion forum or the Discord channel and we will do our best to help.

---

Q5:


Yes, we need the tags

The base Gemma models should already have basic knowledge in many domains. I don't think you need to worry about injecting new knowledge. And figuring out a good data mixture is the key part of this hackathon (kind of like in any real world model training).

One thing I want to mention is that verifiable tasks (math&code) will have much lower weights in the eval. The starter notebooks already cover math and Gemma is not particularly well trained with code.

---

Q6:
Reward Model Training Offline?

If we choose to train a teacher model for the rewards, must the training for this model also be done in the submitted notebook (and hence occupy kernel ressources) or would it be ok if - parallel to the dataset - I train it offline and upload it as a model on kaggle to import?

That is totally fine. Just make sure your main model is reproducible to us in the single session mode.

---

Q7:
Optimal Reward Function Composition for Multi-Domain Reasoning Under 9h TPU Constraint?

I'm experimenting with GRPO fine-tuning on Gemma2-2B for multi-domain reasoning tasks (GSM8K math + ARC science + HumanEval coding). I'm hitting several challenges:

Key Questions:

    Reward Function Trade-offs: How should I weight the reward components?
        Correctness of final answer (0-1 binary)
        Quality of reasoning trace (coherence, step clarity)
        Reasoning length (longer traces = better explanation but higher compute)

    9-Hour TPU Constraint: Given the single v5e-8 session limit:
        Should I prioritize more training steps with smaller batches?
        Or fewer steps with larger batches for better gradient estimates?
        Has anyone benchmarked batch size vs training time trade-offs?

    Multi-Task vs Single-Task:
        Is it better to train separate models for each domain and ensemble?
        Or one unified model with mixed task sampling?
        What's the impact on the reasoning quality score (45 points)?

    Reasoning Trace Format:
        Does the exact XML format matter for evaluation?
        Should traces explicitly state assumptions and show all intermediate steps?
        Or are concise, high-level reasoning traces preferred?

What I've Tried:

    Baseline GRPO with default hyperparameters on GSM8K only
    Mixed task training with 60% math, 30% code, 10% science
    Custom reward that penalizes traces > 500 tokens

Looking for:

    Empirical results from others' experiments
    Insights on evaluation criteria weighting
    Best practices for TPU time optimization

Any insights would be greatly appreciated! Has anyone successfully implemented self-consistency or Reflexion-style loops within the 9h constraint?

---

Q8:
Are Model‑Based Reward Functions Allowed in the Tunix Hackathon?

Are model‑based reward functions allowed in this competition (e.g., using a smaller LLM as a judge for summarization or creativity), or do we need to stick to rule‑based rewards only?

Yes, you can use LLMs to do rewards, but make sure whatever you use are accessible to us, since we will be reproducing your model before eval.

---

Q9:
Reward Function Design Strategies for GRPO Training

Hi everyone! I've been exploring the starter notebook and wanted to share some thoughts on reward function design for this hackathon.

The demo notebook includes 4 reward function types:

    Format checking (ensures proper XML tags)
    Correctness checking (validates final answer)
    Length-based rewards (penalizes overly verbose reasoning)
    Step-by-step reasoning rewards

Key Questions:

    Has anyone experimented with combining multiple reward signals?
    What's the optimal balance between reasoning trace quality and answer correctness?
    Are there benefits to domain-specific reward weighting?

I'm planning to test a hybrid approach that combines correctness (60%), format compliance (20%), and reasoning coherence (20%). Would love to hear your strategies!

Good luck everyone!

