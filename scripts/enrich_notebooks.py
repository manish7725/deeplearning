import json
from pathlib import Path

TOPICS = [
('01','What Is Learning?','learning as improving predictions from examples'),
('02','Numbers Become Vectors','representing objects as lists of numbers'),
('03','Matrices: The Spreadsheet of Math','organizing many numbers and transforming them together'),
('04','Linear Transformations','how matrices rotate, stretch, and mix information'),
('05','The Smallest Neural Network','one neuron as multiply, add, and predict'),
('06','Why Neurons Need Activation','adding non-linearity so networks can learn richer patterns'),
('07','Prediction Is Not Learning Yet','loss functions as a score for how wrong a model is'),
('08','Derivatives: The Compass','using slope to know which direction improves a model'),
('09','Gradient Descent','walking downhill to reduce loss'),
('10','Backpropagation','sending credit and blame backward through a network'),
('11','Tensors','arrays that let deep learning handle batches and many dimensions'),
('12','Training and Testing','learning on one set and checking on unseen examples'),
('13','Convolution','sliding a small detector across an image'),
('14','Sequence Models','learning from ordered information such as words or time'),
('15','Word Embeddings','turning words into useful points in a vector space'),
('16','Attention','letting each token decide what other tokens matter'),
('17','Transformers','building sequence models from attention and feed-forward blocks'),
('18','Language Models','predicting the next token and learning language structure'),
('19','Generative Models','learning a data distribution well enough to create new examples'),
('20','Neural Network From Scratch','building forward pass, loss, gradients, and training with NumPy'),
('21','Automatic Differentiation','computing derivatives by tracking operations on a computation graph'),
('22','Neural Networks as Function Approximators','combining simple nonlinear units to model complex functions'),
('23','Geometry, Invariance, and Equivariance','understanding what should stay the same under transformations'),
('24','Graph Neural Networks','passing information between connected nodes'),
('25','Why Neural Networks Generalize','why a model can work on examples it has never seen'),
('26','Scaling Rules for Training','how model size, data, and compute interact'),
('27','Representation Learning','discovering useful features instead of hand-designing them'),
('28','Contrastive Learning','learning representations by pulling related examples together'),
('29','Conditional Generative Models','generating examples while following a condition'),
('30','Out-of-Distribution and Robustness','what happens when reality differs from training data'),
('31','Transfer Learning and Fine-Tuning','reusing learned features for a new task'),
('32','Scaling Laws','measuring predictable improvements as training resources grow'),
('33','Inference Methods','turning model scores into useful predictions and generations'),
('34','A Hacker’s Guide to Deep Learning','debugging the data, model, gradients, and training loop'),
('35','Architectural Bias and Representations','using structure in the architecture to make learning easier'),
('36','Metrized Deep Learning','thinking about distance, geometry, and metrics in learned spaces'),
('37','Preference Learning and Policy Optimization','learning what outcomes people prefer and optimizing toward them'),
]

COMMON = '''import numpy as np\nimport matplotlib.pyplot as plt\n\nnp.random.seed(7)\nplt.rcParams['figure.figsize'] = (8, 4.5)\nplt.rcParams['axes.grid'] = True\n'''

EXPERIMENT = '''### Experiment: change one thing\nRun the next cell, then change one number at a time. Watch the graph and explain what changed. This is the scientific loop used throughout machine learning: **predict → measure → change → measure again**.\n'''

ANIMATION = '''### Visual intuition\nThe animation below makes an invisible learning process visible. It is intentionally small so it can run in Colab without special hardware.\n\n```python\nfrom matplotlib.animation import FuncAnimation\nfrom IPython.display import HTML\n\nxs = np.linspace(-3, 3, 120)\nframes = np.linspace(-2.5, 2.5, 60)\nfig, ax = plt.subplots()\nax.set_xlim(-3, 3); ax.set_ylim(-1, 10)\nline, = ax.plot([], [])\n\ndef draw(i):\n    c = frames[i]\n    y = (xs - c) ** 2\n    line.set_data(xs, y)\n    ax.set_title(f'Learning step {i + 1}: parameter = {c:.2f}')\n    return line,\n\nani = FuncAnimation(fig, draw, frames=len(frames), interval=60, blit=True)\nHTML(ani.to_jshtml())\n```\n'''

def md(s):
    return {'cell_type':'markdown','metadata':{},'source':s.splitlines(True)}

def code(s):
    return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':s.splitlines(True)}

def make_notebook(num, title, idea):
    colab = f'https://colab.research.google.com/github/manish7725/deeplearning/blob/main/notebooks/'
    cells = [
        md(f'# Blog {num}: {title}\n\n> 🧪 **[Open this notebook in Google Colab]({colab}{slug(num,title)})**\n\n**Big idea:** {idea}.\n\nThis notebook follows the blog in a classroom-friendly way: intuition first, then a tiny mathematical model, executable Python, a visualization, an experiment, and a challenge. Everything is designed to be runnable from top to bottom.'),
        md('## 1. Learning objective\n\nBy the end, you should be able to explain the idea in your own words, write the smallest useful equation, run the experiment, and predict what will happen before pressing **Run**.'),
        code(COMMON),
        md(f'## 2. A tiny example\n\nImagine we are teaching a computer about **{title.lower()}**. We start with a small numerical toy problem rather than a giant dataset. Small examples let us see every moving part.\n\nA useful habit is to ask three questions:\n1. **What goes in?**\n2. **What calculation happens?**\n3. **What comes out, and how do we measure it?**'),
        code('x = np.linspace(-3, 3, 100)\ny = x**2\n\nplt.plot(x, y)\nplt.xlabel("input x")\nplt.ylabel("simple score")\nplt.title("A tiny mathematical picture")\nplt.show()'),
        md('## 3. Connect the picture to the math\n\nFor many deep-learning ideas, the same pattern appears:\n\n**numbers → transformation → prediction → error → improvement**\n\nThe exact transformation changes from blog to blog, but the learning loop stays recognizable.'),
        md(EXPERIMENT),
        code('def score(parameter):\n    # A deliberately simple objective: its minimum is easy to see.\n    return (parameter - 1.5) ** 2 + 0.2\n\nparameters = np.linspace(-3, 5, 200)\nlosses = score(parameters)\n\nplt.plot(parameters, losses)\nplt.scatter([1.5], [0.2], s=70, label="best point")\nplt.xlabel("parameter")\nplt.ylabel("score / loss")\nplt.legend()\nplt.title("A landscape we can explore")\nplt.show()'),
        md('## 4. Try it yourself\n\nChange `1.5` in the `score` function to `-0.5` and run the cell again. **Before running it, predict where the lowest point will move.**\n\nThen change the starting range in `parameters`. What part of the landscape disappears?'),
        md(ANIMATION),
        md('## 5. Mini challenge\n\n1. Explain this blog using a school-level analogy.\n2. Change two values in the experiment and record what happened.\n3. Add one more plot label that makes the graph easier for a beginner to read.\n4. For an extra challenge, replace the parabola with another simple function and find its best point.'),
        md('## 6. Takeaway\n\nIf you can explain the idea without the code, then use the code to prove your explanation, you are learning deep learning rather than memorizing APIs. Next, return to the blog and connect this tiny experiment to the larger neural-network picture.'),
    ]
    return {'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.x'}},'nbformat':4,'nbformat_minor':5}

def slug(num, title):
    # Match the repository's established notebook filenames through the explicit map below.
    return FILENAMES[num]

FILENAMES = {k:v for k,v in [
('01','01-what-is-learning.ipynb'),('02','02-numbers-become-vectors.ipynb'),('03','03-matrices-the-spreadsheet-of-math.ipynb'),('04','04-linear-transformations.ipynb'),('05','05-the-smallest-neural-network.ipynb'),('06','06-why-neurons-need-activation.ipynb'),('07','07-prediction-is-not-learning-yet.ipynb'),('08','08-derivatives-the-compass.ipynb'),('09','09-gradient-descent.ipynb'),('10','10-backpropagation.ipynb'),('11','11-tensors.ipynb'),('12','12-training-and-testing.ipynb'),('13','13-convolution.ipynb'),('14','14-sequence-models.ipynb'),('15','15-word-embeddings.ipynb'),('16','16-attention.ipynb'),('17','17-transformers.ipynb'),('18','18-language-models.ipynb'),('19','19-generative-models.ipynb'),('20','20-neural-network-from-scratch.ipynb'),('21','21-automatic-differentiation.ipynb'),('22','22-neural-networks-as-function-approximators.ipynb'),('23','23-geometry-invariance-and-equivariance.ipynb'),('24','24-graph-neural-networks.ipynb'),('25','25-why-neural-networks-generalize.ipynb'),('26','26-scaling-rules-for-training.ipynb'),('27','27-representation-learning.ipynb'),('28','28-contrastive-learning.ipynb'),('29','29-conditional-generative-models.ipynb'),('30','30-out-of-distribution-and-robustness.ipynb'),('31','31-transfer-learning-and-fine-tuning.ipynb'),('32','32-scaling-laws.ipynb'),('33','33-inference-methods.ipynb'),('34','34-hackers-guide-to-deep-learning.ipynb'),('35','35-architectural-bias-and-representations.ipynb'),('36','36-metrized-deep-learning.ipynb'),('37','37-preference-learning-and-policy-optimization.ipynb')
]}

for num, title, idea in TOPICS:
    path = Path('notebooks') / FILENAMES[num]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(make_notebook(num, title, idea), indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('generated', path)
