<div align="center">
  <a href="#">
  	<img src="https://media.giphy.com/media/JIX9t2j0ZTN9S/giphy-downsized.gif" alt="Logo project" height="160" />
  </a>
  <br>
  <br>
  <p>
    <b>MSSA</b>
  </p>
  <p>
     <i>MSSA is designed to detect binary code similarities. MSSA not only uses the CFG structure information in the upper binary code, but also uses the Bi-GRU network to extract the semantic information between the binary code instructions.</i>
  </p>
</div>

---

**Content**

* [Description](##description)
* [Install](##install)
* [Usage](##usage)
* [Exemples](##exemples)
* [Documentation](##documentation)
* [Datasets](##datasets)
* [Evaluation](##evaluation)
* [Maintainers](##maintainers)

## Description ✨
MSSA is a binary code clone detection method.
the final **AUC** value can reach about **99.7%**. For more specific information, please refer to paper------.
### The MSSA Model 
![MSSA Model](https://github.com/zhouCQJ/MSSA/assets/170287774/8c34eae0-055c-42aa-b9ad-3240314672e9)

## Install 🐙
It is recommended that you install a conda environment and then install the dependent packages with the following command：
```
conda create -n DeepBCCD37 -y python==3.7.16 && conda activate DeepBCCD37
pip install -r requirements.txt
```

## Usage 💡
1. git clone the project.
```
git clone https://github.com/zhouCQJ/MSSA.git -d your_profile
```
2. Go inside the project folder(IDE) and open your terminal.
3. See  [Install](##install) to install the environment.
4. run the command `python run.py --train true --test true` to start.

## Exemples 🖍
We trained with the **A5000** GPU.
```
python run.py --train true --test true --mrr true --w2v_dim 100 --batch_size 10024--max_block_seq 20--num_block 20 --iter_level 5 
```

## Documentation 📄
For a more detailed description of the contents of MSSA, please refer to our paper-----

## Datasets 👩‍💻
For the datasets, we used the datasets **BinaryCorp-3M**（https://github.com/vul337/jTrans) in the Jtrans paper. To conform to the input format according to the DeepBCCD model, We re-extracted the binary function set from the source binary and formed the **dataset_train.csv** and **dataset_test.csv** datasets，which are also essentially derived from Binarycorp-3M.
For the **BinaryCrop-26M** dataset, we will try it in the future because it requires a larger training resource。					

The dataset used in MSSA [download](https://efss.qloud.my/index.php/s/a2B2S9rNwdXkmBo).

## Maintainers 👷
* @zhouCQJ

## License ⚖️
GPL

---
<div align="center">
	<b>
		<a href="https://www.npmjs.com/package/get-good-readme">File generated with get-good-readme module</a>
	</b>
</div>
