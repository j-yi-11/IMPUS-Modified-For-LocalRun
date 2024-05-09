# [IMPUS]-Modified-For-LocalRun
Since code in original repo(https://github.com/GoL2022/IMPUS) is only with jupyter. I modify it so as to run on my machine.
## step
```bash
conda create -n impus python=3.8 -y
conda activate impus
```
Then install some dependencies as follows:
```bash
pip install -r requirements.txt
pip install diffusers==0.19.2
```
Finally run:
```bash
python main.py
```

