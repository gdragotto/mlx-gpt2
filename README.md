# GPT-2 MLX with OpenAI GPT-2 Output Dataset
This is a fork of [MLX GPT2 from pranavjad](https://github.com/pranavjad/mlx-gpt2) to train a GPT-2 model with the [OpenAI GPT-2 Output Dataset](https://github.com/openai/gpt-2-output-dataset).
To download the data used to train (small-117M.test.jsonl and small-117M.train.jsonl), please follow [the instructions here](https://github.com/openai/gpt-2-output-dataset) and put the dataset in a new folder called "data".

### Estimated Training Time
On a Mac Pro M3, 11 cores active and with the small-117M training set, an epoch takes around 2h and 45' to train. 
To train the model for roughly 20 epochs, it will take roughly 57 hours. It's a nice weekend task you can perform while you're doing something else 😇
