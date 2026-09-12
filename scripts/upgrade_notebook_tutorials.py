import json
from pathlib import Path

BLOGS = {f'{i:02d}': f'{i:02d}-' for i in range(1, 38)}
TOPICS = {
'01': ('What Is Learning?', 'learning means changing parameters so predictions become better', 'prediction → error → adjustment'),
'02': ('Numbers Become Vectors', 'a list of numbers can represent an object as a point with direction and size', 'numbers → vector → geometry'),
'03': ('Matrices: The Spreadsheet of Math', 'a matrix organizes numbers and lets us transform many values together', 'vectors → matrices → matrix multiplication'),
'04': ('Linear Transformations', 'matrix multiplication can rotate, stretch, shrink, and mix coordinates', 'matrix → transformation → new coordinates'),
'05': ('The Smallest Neural Network', 'one neuron is a weighted sum plus a bias that makes a prediction', 'inputs → weights → bias → prediction'),
'06': ('Why Neurons Need Activation', 'nonlinear activation lets layers learn patterns that a straight line cannot', 'weighted sum → activation → nonlinear function'),
'07': ('Prediction Is Not Learning Yet', 'a model needs a loss function to measure how wrong its prediction is', 'prediction → loss → learning signal'),
'08': ('Derivatives: The Compass', 'a derivative tells us how a small change in a parameter changes the loss', 'slope → chain rule → gradient'),
'09': ('Gradient Descent', 'gradient descent repeatedly changes parameters in the direction that lowers loss', 'gradient → step → lower loss'),
'10': ('Backpropagation', 'backpropagation applies the chain rule through many operations to find parameter gradients', 'forward pass → chain rule → gradients'),
'11': ('Tensors', 'tensors extend vectors and matrices to batches and higher-dimensional data', 'shape → dimensions → batch computation'),
'12': ('Training and Testing', 'a model should learn from training examples and be checked on unseen examples', 'train set → learned parameters → test set'),
'13': ('Convolution', 'a small filter slides across data and detects local patterns', 'window → multiply/add → feature map'),
'14': ('Sequence Models', 'ordered data needs a mechanism that carries information from earlier positions', 'sequence → state → next prediction'),
'15': ('Word Embeddings', 'words can be represented as vectors so useful relationships become geometric', 'token → vector → similarity'),
'16': ('Attention', 'attention lets one token assign different importance to other tokens', 'queries/keys → scores → softmax → weighted values'),
'17': ('Transformers', 'transformers combine attention, feed-forward layers, residual paths, and normalization', 'tokens → attention blocks → representations'),
'18': ('Language Models', 'a language model learns a probability distribution for the next token', 'context → logits → probabilities → next token'),
'19': ('Generative Models', 'a generative model learns enough structure in data to create new examples', 'data distribution → sampling → new example'),
'20': ('Neural Network From Scratch', 'a complete neural network can be built from forward pass, loss, gradients, and updates', 'forward → loss → backprop → update'),
'21': ('Automatic Differentiation', 'automatic differentiation records operations so derivatives can be computed reliably', 'computation graph → local derivatives → chain rule'),
'22': ('Neural Networks as Function Approximators', 'many simple nonlinear units can combine to approximate complicated functions', 'basis functions → hidden layer → approximation'),
'23': ('Geometry, Invariance, and Equivariance', 'some transformations should leave an answer unchanged while others should transform the answer predictably', 'transform input → transform output → compare'),
'24': ('Graph Neural Networks', 'nodes can update their representations by exchanging messages with neighbors', 'nodes → messages → aggregation → update'),
'25': ('Why Neural Networks Generalize', 'good learning captures useful structure rather than merely memorizing training examples', 'training fit → unseen data → generalization'),
'26': ('Scaling Rules for Training', 'model size, data, and compute interact and changing one can change the useful training regime', 'model/data/compute → trade-offs'),
'27': ('Representation Learning', 'a model can discover useful features instead of requiring people to hand-design every feature', 'raw input → learned representation → task'),
'28': ('Contrastive Learning', 'representations can be learned by making related examples close and unrelated examples far apart', 'positive pair → negative pair → similarity objective'),
'29': ('Conditional Generative Models', 'generation can be guided by an input condition such as a class or text prompt', 'condition → generator → conditional sample'),
'30': ('Out-of-Distribution and Robustness', 'a model may face inputs that differ from its training distribution', 'training distribution → shifted input → robustness test'),
'31': ('Transfer Learning and Fine-Tuning', 'a model can reuse learned features and adapt them to a new task', 'pretrained parameters → new task → fine-tuning'),
'32': ('Scaling Laws', 'performance can often follow predictable trends as training resources increase', 'resource → measured loss → fitted trend'),
'33': ('Inference Methods', 'the same model scores can produce different outputs depending on the decoding strategy', 'scores → decoding rule → output'),
'34': ("A Hacker’s Guide to Deep Learning", 'debugging is a scientific process of checking data, shapes, gradients, and updates', 'observe → isolate → test → fix'),
'35': ('Architectural Bias and Representations', 'architecture can make some patterns easier to learn by building useful assumptions into the computation', 'architecture → inductive bias → representation'),
'36': ('Metrized Deep Learning', 'distance and similarity give us tools for studying the geometry of learned representations', 'representation → metric → geometry'),
'37': ('Preference Learning and Policy Optimization', 'a model can learn which outcomes are preferred and optimize toward those preferences', 'preference → objective → policy update'),
}

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / 'notebooks'

MARKER = '<!-- SELF-CONTAINED-CODE-TUTORIAL -->'

def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(True)}

def explain_code(source, num):
    s = source.lower()
    points = []
    if 'import numpy' in s or 'import matplotlib' in s or 'import torch' in s:
        points.append('The imports bring in the tools used below. Read each library name as a toolbox: NumPy handles arrays and arithmetic, Matplotlib draws graphs, and PyTorch handles tensor/autograd work when used.')
    if 'np.array' in s or 'np.linspace' in s or 'torch.tensor' in s:
        points.append('The first important objects are the data. Look at their shapes and values before thinking about the model. In deep learning, the shape tells you how many examples and features are being processed.')
    if 'def ' in s:
        points.append('The function packages a repeated calculation into a named step. Read its inputs first, then its return value; the function name is a clue about the mathematical operation.')
    if any(x in s for x in ['w =', 'weights', 'bias', 'parameter', 'theta']):
        points.append('These variables are parameters: numbers the learning algorithm is allowed to change. Ask: what would happen if this number became slightly larger?')
    if any(x in s for x in ['pred', 'prediction', 'y_hat', 'logit']):
        points.append('This is the prediction stage. The code turns the input and current parameters into a model output. Nothing has learned yet unless a later step uses an error signal to change parameters.')
    if any(x in s for x in ['loss', 'error', 'mse', 'cross_entropy']):
        points.append('This measures how far the prediction is from the target. A smaller value means the model is doing better according to this chosen objective.')
    if any(x in s for x in ['gradient', 'dw', 'db', 'grad', 'derivative']):
        points.append('These lines calculate a direction of change. The gradient answers: if this parameter moves a little, does the loss go up or down, and by roughly how much?')
    if any(x in s for x in ['lr', 'learning_rate', '-=', 'optimizer', 'step()']):
        points.append('This is the update step. The learning rate controls how large a move we make; the parameter is changed using the gradient so the loss should move toward a better value.')
    if 'for ' in s or 'while ' in s:
        points.append('The loop repeats the learning or simulation step. One iteration is one small experiment; many iterations let us see a trend rather than a single lucky result.')
    if any(x in s for x in ['plt.', 'plot(', 'scatter(', 'imshow(', 'bar(', 'legend(']):
        points.append('The plotting code is not decoration: it turns numbers into evidence. Before running it, predict the shape of the graph; after running it, explain why the graph has that shape.')
    if 'reshape' in s or 'shape' in s or 'matmul' in s or '@' in s:
        points.append('Pay attention to shape. Matrix/tensor operations are only valid when their dimensions line up. When confused, print the shape and trace one small example by hand.')
    if not points:
        points.append('Read this cell from top to bottom. Identify the inputs, the calculation, and the output. Then connect each line to the mathematical idea in the blog.')
    return '\n\n'.join('- ' + p for p in points)

def main():
    changed = 0
    for path in sorted(NOTEBOOK_DIR.glob('*.ipynb')):
        num = path.name[:2]
        if num not in TOPICS:
            continue
        data = json.loads(path.read_text(encoding='utf-8'))
        cells = data.get('cells', [])
        if any(MARKER in ''.join(c.get('source', [])) for c in cells if c.get('cell_type') == 'markdown'):
            continue
        title, objective, chain = TOPICS[num]
        blog = f'{num}-{path.stem[3:]}.md'
        blog_url = f'https://github.com/manish7725/deeplearning/blob/main/blogs/{blog}'
        header = []
        inserted = False
        for cell in cells:
            if not inserted and cell.get('cell_type') == 'markdown' and cell.get('source', []) and cell['source'][0].startswith('# Blog '):
                header.append(cell)
                header.append(md(f'''{MARKER}\n\n## 🧪 Lab objective and blog connection\n\n**Objective:** {objective}.\n\n**The learning path:** **{chain}**.\n\n**Read the matching blog first:** [Blog {num} — {title}]({blog_url})\n\nThis notebook is the hands-on companion to that blog. The blog explains the idea and mathematics; this lab turns the same idea into executable code. If a line of code feels mysterious, stop and ask three questions: **What data goes in? What calculation is performed? What number/graph comes out?**\n\n### How this notebook is organized\n\n1. **Concept** — a short reminder of the idea from the blog.\n2. **Code tutorial** — code is introduced in small pieces, with explanations of what each part does.\n3. **Experiment** — change one thing and observe the result.\n4. **Reflection** — explain the result in your own words and connect it back to the mathematics.\n\n> **Tip for beginners:** Run cells from top to bottom the first time. Do not just copy the code. Predict what a cell should do, run it, then compare your prediction with the output.\n'''))
                inserted = True
            elif cell.get('cell_type') == 'code':
                src = ''.join(cell.get('source', []))
                header.append(md('### 🔎 Code tutorial — what to notice\n\n' + explain_code(src, num)))
                header.append(cell)
                header.append(md('### 💡 What this code teaches\n\nNow connect the output back to the blog: identify the mathematical operation represented by this cell and explain what changed in the data, prediction, loss, gradient, or representation. If you cannot explain the output in one or two sentences, rerun the cell with a smaller example and inspect the numbers.'))
            else:
                header.append(cell)
        data['cells'] = header
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        changed += 1
        print('upgraded', path)
    print(f'Updated {changed} notebooks')

if __name__ == '__main__':
    main()
