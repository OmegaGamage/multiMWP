# A simple example for finetuning T5. 

To set up, you'll need to pip install transformers and all that normal stuff.

# The important details:
## Dataset
You'll have a dataset of some sort mapping source => target. To use this you'll want to set up a data_dir (which you will specify in train.py) to have
- train.source
- train.target
- val.source
- val.target
- test.source
- test.target

Where each source/train pair has one example per line. So, e.g., for QA you
would have
- questions in .source
- answers in .target
- with one line for each of them

You can generate a split using sklearn

## Config
You'll want to tweak the k_* parameters at the top of train.py

## Tensorboard
To run tensorboard, just pip install tensorboard and then
tensorboard --logdir=<your save dir>
  
  
# Notes
- Test is not implemented, so if you want to test on a holdout dataset, you'll want to tweak the code to generate a dataset on test.source and test.target and evaluate the metrics you want.
- If your task is similar to one of the originally trained tasks like summarization, you might benefit from prepending a task label to your inputs, like "summarize: " or "translate English to Russian: "

# Citation
If you use this dataset in your research, please cite our paper:

**O. Gamage et al., "A Multilingual Dataset (MultiMWP) and Benchmark for Math Word Problem Generation," in IEEE Transactions on Audio, Speech and Language Processing, doi: [10.1109/TASLPRO.2025.3552936](https://doi.org/10.1109/TASLPRO.2025.3552936).**

### BibTeX:
```bibtex
@ARTICLE{10933586,
  author={Gamage, Omega and Ranathunga, Surangika and Lee, Annie and Sun, Xiao and Singh, Aryaveer and Skenduli, Marjana Prifti and Alam, Mehreen and Nayak, Ajit Kumar and Gao, Haonan and Deori, Barga and Ji, Jingwen and Zhang, Qiyue and Zeng, Yuchen and Tian, Muxin and Mao, Yanke and Trico, Endi and Nako, Danja and Shqezi, Sonila and Hoxha, Sara and Imami, Dezi and Doksani, Dea and Pandey, Virat Kumar and Ananya, Ananya and Aggarwal, Nitisha and Hussain, Naiyarah and Dwivedi, Vandana and Sinha, Rajkumari Monimala and Kalita, Dhrubajyoti},
  journal={IEEE Transactions on Audio, Speech and Language Processing},
  title={A Multilingual Dataset (MultiMWP) and Benchmark for Math Word Problem Generation},
  year={2025},
  volume={},
  number={},
  pages={1-13},
  keywords={Translation;Multilingual;Mathematical models;Arithmetic;Natural language processing;Speech processing;Data models;Data mining;Artificial neural networks;Training;Benchmark;low-resource languages;math word problem generation;multi-way parallel dataset;multilingual dataset},
  doi={10.1109/TASLPRO.2025.3552936}
}
