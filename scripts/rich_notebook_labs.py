import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / 'notebooks'
MARK = '<!-- RICH-MASTERY-LAB -->'

TOPICS = {
1: ('What Is Learning?', 'learning means changing parameters so predictions improve', 'y=wx+b', 'fit a line and watch loss fall'),
2: ('Numbers Become Vectors', 'numbers can become points, arrows, distances, and directions', 'v=[3,4], ||v||=5', 'change one coordinate and observe distance'),
3: ('Matrices', 'a matrix collects many linear combinations into one operation', 'A=[[1,2],[3,4]], x=[1,1]', 'change one matrix entry and observe Ax'),
4: ('Linear Transformations', 'matrix multiplication moves points in predictable geometric ways', 'A=diag(2,1), x=[1,2]', 'change one scale factor and plot the grid'),
5: ('The Smallest Neural Network', 'one neuron is a weighted sum plus a bias', 'y_hat=wx+b', 'change one weight and see the line rotate'),
6: ('Activation Functions', 'nonlinearity lets stacked layers represent curved rules', 'ReLU(x)=max(0,x)', 'change the activation and compare curves'),
7: ('Prediction Is Not Learning Yet', 'a prediction becomes learning only when an objective measures error', 'L=(y-y_hat)^2', 'change the target and compare loss'),
8: ('Derivatives', 'a derivative measures local sensitivity', 'd/dx x^2=2x', 'change the evaluation point and compare slopes'),
9: ('Gradient Descent', 'repeated downhill steps can minimize a loss', 'w_next=w-eta*2(w-3)', 'change eta and compare trajectories'),
10: ('Backpropagation', 'the chain rule carries credit backward through a computation', 'y=(wx+b)^2', 'change one upstream value and compare gradients'),
11: ('Tensors', 'higher-dimensional arrays let models process batches and structured data', 'X.shape=(batch,features)', 'change batch size and inspect shapes'),
12: ('Training and Testing', 'generalization means performing well on unseen examples', 'train/test split', 'change the split and compare test error'),
13: ('Convolution', 'a local filter slides and detects a pattern', 'sum(x[i:i+3]*k)', 'change the kernel and compare feature maps'),
14: ('Sequence Models', 'a state carries information across ordered inputs', 'h_t=tanh(Wx_t+Uh_prev)', 'change sequence order and compare states'),
15: ('Word Embeddings', 'tokens can be represented by vectors whose geometry carries meaning', 'cosine(a,b)', 'change one embedding and observe similarity'),
16: ('Attention', 'queries score keys and mix values according to those scores', 'softmax(QK^T/sqrt(d))V', 'change one key and watch attention weights move'),
17: ('Transformers', 'attention, residual paths, feed-forward layers, and normalization form a reusable block', 'x+Attention(x)', 'change one attention score and compare output'),
18: ('Language Models', 'a language model turns context into next-token probabilities', 'softmax(logits)', 'change one logit and observe probability mass shift'),
19: ('Generative Models', 'generation samples new points from a learned distribution', 'z~N(0,1)', 'change noise scale and compare samples'),
20: ('Neural Network From Scratch', 'a complete network can be built from forward pass, loss, gradients, and updates', 'XW+b', 'change hidden width and compare fit'),
21: ('Automatic Differentiation', 'a computation graph lets software apply the chain rule automatically', 'z=(x*y)+sin(x)', 'change x and inspect dz/dx'),
22: ('Function Approximators', 'many nonlinear units can build complicated functions', 'y=sum_i a_i sigma(w_i x+b_i)', 'change hidden units and compare approximation'),
23: ('Invariance and Equivariance', 'some transformations should leave outputs unchanged or transform them predictably', 'f(Tx)=f(x) or f(Tx)=Tf(x)', 'change the input transformation'),
24: ('Graph Neural Networks', 'nodes learn by aggregating messages from neighbors', 'h_i=sum_j A_ij h_j', 'change one edge and compare node states'),
25: ('Generalization', 'the goal is useful structure, not memorization', 'train loss vs test loss', 'change model complexity and compare the gap'),
26: ('Scaling Rules', 'model, data, and compute interact', 'loss=f(model,data,compute)', 'change one resource while holding others fixed'),
27: ('Representation Learning', 'a network can learn features useful for downstream tasks', 'z=f_theta(x)', 'change the representation dimension'),
28: ('Contrastive Learning', 'related examples should be close while unrelated ones separate', 'L=-log exp(sim_pos)/sum exp(sim)', 'change temperature and compare similarities'),
29: ('Conditional Generation', 'a condition guides the generated output', 'p(x|c)', 'change the condition and compare samples'),
30: ('Distribution Shift and Robustness', 'performance can change when deployment inputs differ from training', 'P_train(x) != P_test(x)', 'change noise or shift magnitude'),
31: ('Transfer Learning', 'pretrained features can be reused and adapted', 'theta_new <- theta_pretrained', 'change frozen layers and compare learning'),
32: ('Scaling Laws', 'measured performance can follow smooth resource trends', 'L(N)=aN^-alpha+c', 'change exponent or data range and refit'),
33: ('Inference Methods', 'the same scores can produce different outputs under different decoding rules', 'argmax vs sampling', 'change temperature and compare outputs'),
34: ('Hacker’s Guide', 'debugging is hypothesis testing over data, shapes, gradients, and updates', 'assert invariants', 'change one assumption and catch the failure'),
35: ('Architectural Bias', 'architecture encodes assumptions about which patterns are easy to learn', 'local vs global mixing', 'change connectivity and compare learning'),
36: ('Metrized Deep Learning', 'metrics turn learned representations into measurable geometry', 'd(a,b)=||a-b||', 'change the metric and compare neighborhoods'),
37: ('Preference Learning', 'a policy can optimize an objective built from relative preferences', 'maximize E[r(x)]', 'change reward strength and compare policy updates'),
}

def md(s):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': s.splitlines(True)}

def code(s):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': s.splitlines(True)}

def archetype(n):
    if n <= 10: return 'foundation'
    if n in (11, 12): return 'tensor'
    if n in (13, 14): return 'sequence'
    if n in (15, 16, 17, 18): return 'attention'
    if n in (19, 29): return 'generation'
    if n in (20, 21, 22): return 'optimization'
    if n in (23, 24, 35): return 'structure'
    if n in (25, 26, 30, 32): return 'experiment'
    if n in (27, 28, 31, 36): return 'representation'
    return 'systems'

def lab_cells(n, title, intuition, math, experiment):
    cells = [md(f'''{MARK}\n\n# 🧪 Mastery Laboratory — {title}\n\nThis notebook is the **laboratory companion** to the blog. The blog tells the story and develops the intuition; this section makes you *do* the mathematics. A strong learning cycle is: **read → predict → calculate → run → change → break → explain**.\n\n## 0. Story and intuition checkpoint\n\n**Core intuition:** {intuition}.\n\n**Mathematical anchor:** `{math}`\n\nBefore running anything, explain the idea to a Class 7 student using one analogy. Then explain the same idea using the equation above. If you cannot do both, return to the matching blog.\n\n### Learning contract\n\n1. Recalculate the smallest example by hand.\n2. Run the matching notebook cells.\n3. Implement the idea with NumPy.\n4. Implement it with PyTorch.\n5. Visualize what the mathematics predicts.\n6. Change exactly one variable.\n7. Break the experiment intentionally and explain why.\n8. Complete the mini-project.\n9. At advanced levels, reproduce a paper and formulate a research question.\n'''),
        md('''## 1. ✍️ Recalculate the smallest numerical example by hand\n\nDo this **before** pressing Run. Write the intermediate numbers, not only the final answer. Then use the next cell as a calculator to verify your work.\n\n**Scientist habit:** prediction first, computation second, explanation third.'''),
        code(f'''# Hand-calculation checker for Chapter {n:02d}\n# Start with tiny numbers so every intermediate value is visible.\nimport numpy as np\n\nexample = np.array([1.0, 2.0, 3.0])\nprint("tiny example:", example)\nprint("sum:", example.sum())\nprint("mean:", example.mean())\nprint("squared values:", example**2)\n'''),
        md('''## 2. 🧮 NumPy — build the idea yourself\n\nNumPy is deliberately low-level here. Do not hide the mathematics behind a high-level library. Print the intermediate arrays and connect each one to a symbol in the blog.'''),
        code(f'''# NumPy practice scaffold — Chapter {n:02d}\n# {experiment}\nimport numpy as np\nimport matplotlib.pyplot as plt\n\nrng = np.random.default_rng(7)\nx = np.linspace(-3, 3, 101)\ny = np.sin(x)\n\nz = x**2\nprint("x shape:", x.shape)\nprint("first five x:", x[:5])\nprint("first five z:", z[:5])\n\nplt.figure(figsize=(8,4))\nplt.plot(x, y, label="reference")\nplt.plot(x, z / z.max(), label="normalized experiment")\nplt.title("Chapter {n:02d}: predict, run, explain")\nplt.legend(); plt.show()\n'''),
        md('''## 3. 🔥 PyTorch — express the same mathematics with tensors\n\nNow repeat the calculation with PyTorch. The point is not to replace mathematics with a framework. The point is to prove that the framework is executing the mathematics you already understand.'''),
        code('''import torch\n\ntorch.manual_seed(7)\ntx = torch.linspace(-3, 3, 101)\ntx.requires_grad_(True)\ntz = tx**2\nprint("tensor shape:", tuple(tx.shape))\nprint("first five values:", tz[:5])\n\nprobe = tz.mean()\nprobe.backward()\nprint("gradient exists:", tx.grad is not None)\nprint("first five gradients:", tx.grad[:5])\n'''),
        md('''## 4. 📈 Visualize what the mathematics predicts\n\nA graph is an experiment, not decoration. Before running the next cell, predict: **where should the curve be high, low, steep, or flat?** Then compare your prediction with the plot.'''),
        code('''plt.figure(figsize=(8,4))\nplt.plot(tx.detach().numpy(), tz.detach().numpy())\nplt.xlabel("input")\nplt.ylabel("mathematical output")\nplt.title("What the equation predicts")\nplt.grid(True)\nplt.show()\n'''),
        md('''## 5. 🔬 Change exactly one variable\n\nChange only one number in the next experiment. Good choices are learning rate, scale, temperature, matrix entry, sequence length, noise level, hidden width, or threshold depending on this chapter. **Predict the direction of change first.**'''),
        code('''scale = 1.0  # <-- change ONLY this value first\ny_changed = np.sin(x * scale)\nplt.figure(figsize=(8,4))\nplt.plot(x, y, label="original")\nplt.plot(x, y_changed, label=f"scale={scale}")\nplt.legend(); plt.grid(True); plt.show()\n'''),
        md('''## 6. 💥 Break it intentionally\n\nA scientist learns from failure. Change the experiment so that it becomes wrong or unstable. Examples: use a huge learning rate, reverse a sequence, swap tensor dimensions, remove normalization, use the wrong target, or make the noise enormous.\n\nThen answer: **What assumption did I violate? What symptom appeared? What mathematical reason explains the failure?**'''),
        code('''# Intentional stress test: extreme scale creates rapid oscillation.\nbroken_scale = 25.0\ny_broken = np.sin(x * broken_scale)\nprint("broken_scale =", broken_scale)\nprint("range =", (y_broken.min(), y_broken.max()))\nplt.figure(figsize=(8,4))\nplt.plot(x, y_broken)\nplt.title("Intentional failure / stress test")\nplt.grid(True); plt.show()\n'''),
        md(f'''## 7. 🛠️ Mini-project — {title}\n\nBuild a small experiment that answers a question about this chapter rather than merely reproducing the tutorial.\n\n**Minimum deliverable**\n- one clearly stated hypothesis;\n- one hand calculation;\n- one NumPy implementation;\n- one PyTorch implementation;\n- one visualization;\n- one controlled variable change;\n- one intentional failure;\n- a 5–10 sentence conclusion.\n\n**Starter question:** {experiment}.\n\nKeep an experiment log with columns: `hypothesis | variable | value | result | explanation`.'''),
        md('''## 8. 🎓 Advanced research bridge\n\nFor Chapters 1–20, treat this as a research-style extension. For advanced chapters, turn it into a real reproduction exercise.\n\n**Paper reproduction protocol**\n1. Choose one paper connected to this chapter.\n2. Write the claim you are trying to reproduce.\n3. Record dataset, model, metric, seed, and compute budget.\n4. Reproduce the simplest baseline first.\n5. Change one experimental factor at a time.\n6. Report uncertainty and failures, not only the best run.\n\n**Research-question template:**\n> If I change **X**, while holding **Y** and **Z** fixed, does **M** change? Why would the mathematics or architecture predict that?\n\nA good research question should be falsifiable. “Can I make it better?” is too vague; “Does doubling X improve M under the same compute budget?” is testable.'''),
        md('''## 9. ✅ Mastery check\n\nYou are done only when you can answer all of these without looking at the blog:\n\n- What problem does this chapter solve?\n- What is the smallest numerical example?\n- What does every symbol in the main equation mean?\n- Can I calculate it by hand?\n- Can I implement it in NumPy?\n- Can I implement it in PyTorch?\n- What graph should I expect before running it?\n- What happens when I change one variable?\n- What failure mode did I create deliberately?\n- What mini-project would I build next?\n\n**Final rule:** never accept a graph or a loss value without being able to explain *why* it looks that way.''')]
    return cells

def main():
    changed = 0
    for p in sorted(NOTEBOOKS.glob('*.ipynb')):
        try: n = int(p.name[:2])
        except ValueError: continue
        if n not in TOPICS: continue
        data = json.loads(p.read_text(encoding='utf-8'))
        if any(MARK in ''.join(c.get('source', [])) for c in data.get('cells', [])): continue
        title, intuition, math, experiment = TOPICS[n]
        data['cells'].extend(lab_cells(n, title, intuition, math, experiment))
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        changed += 1
    print(f'Rich mastery labs added: {changed}')

if __name__ == '__main__':
    main()
