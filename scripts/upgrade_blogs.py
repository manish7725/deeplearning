"""Upgrade short placeholder blogs into complete Class-8-to-PhD lessons.

Only blogs that are still short placeholders are replaced. Existing richer lessons
are preserved. The generated lesson keeps one consistent teaching contract:
intuition -> math -> hand calculation -> NumPy experiment -> failure mode ->
research bridge.
"""
from pathlib import Path
import re
import textwrap

ROOT = Path(__file__).resolve().parents[1]
BLOGS = ROOT / "blogs"

DETAILS = {
"21": ("Automatic Differentiation", "A computer can turn a chain of tiny operations into a chain of local derivatives.", "For z = x^2 and L = 3z, dz/dx = 2x and dL/dx = 3(2x). The chain rule multiplies local slopes.", "Use finite differences to check an analytical derivative, then compare the same derivative with PyTorch autograd.", "Change the finite-difference step size. Very large or very tiny steps can make the numerical estimate worse."),
"22": ("Neural Networks as Function Approximators", "A network can combine many simple bends to draw a complicated curve.", "A one-hidden-layer network has f(x)=sum_j a_j sigma(w_j x+b_j)+c. Each hidden unit contributes a reusable nonlinear shape.", "Fit sin(2x) with polynomial features and then compare with a tiny neural network; plot predictions and error.", "Too few hidden units underfit; too many units can fit noise instead of the underlying rule."),
"23": ("Geometry, Invariance, and Equivariance", "Some transformations should not change an answer, while others should change the answer in a predictable way.", "Invariance means f(Tx)=f(x). Equivariance means f(Tx)=T'f(x). Rotation is a simple transformation matrix example.", "Rotate a set of points and compare distances before and after transformation. Test a quantity that should be invariant.", "A model can memorize positions instead of learning the underlying symmetry."),
"24": ("Graph Neural Networks", "When data is a network, a node learns from the information carried by its neighbors.", "A simple message-passing layer is H' = sigma(AHW), where A describes connections, H node features, and W learnable weights.", "Build a three-node graph, compute one aggregation step by hand, and inspect how changing an edge changes a node representation.", "If the graph is disconnected or the aggregation loses important information, useful relationships may disappear."),
"25": ("Why Neural Networks Generalize", "Learning is useful only if the model works beyond the examples it memorized.", "Training minimizes empirical risk R_hat(f)=1/n sum_i L(f(x_i),y_i); generalization asks how close this is to population risk R(f)=E[L].", "Create a train/test split and plot both losses across training. Watch for a growing generalization gap.", "A very low training loss is not evidence of good performance on unseen data."),
"26": ("Scaling Rules for Training", "Model size, data, batch size, and compute interact; bigger is not a single magic button.", "A simplified scaling view writes loss as L(N,D,C), where N is parameters, D data, and C compute. Holding one resource fixed can make another resource less useful.", "Simulate noisy gradient estimates for several batch sizes and plot variance against batch size.", "Increasing batch size can reduce gradient noise but can also change optimization dynamics and compute efficiency."),
"27": ("Representation Learning", "Instead of hand-writing every useful feature, a model can learn an internal coordinate system.", "An encoder maps x to z=f_theta(x), and a task consumes z. A useful representation keeps information that matters while discarding irrelevant variation.", "Project a small dataset into two dimensions and measure whether simple clusters become easier to separate.", "A representation is not automatically good: it is good relative to a task, invariance, and downstream use."),
"28": ("Contrastive Learning", "The model learns by being told which examples should be close and which should be far apart.", "For an anchor i and positive j, InfoNCE uses L_i=-log exp(sim(z_i,z_j)/tau) / sum_k exp(sim(z_i,z_k)/tau).", "Create positive and negative vector pairs, calculate cosine similarities, and see how temperature changes the softmax competition.", "False negatives can teach the model to push two genuinely related examples apart."),
"29": ("Conditional Generative Models", "Generation becomes controllable when the model receives a condition describing what it should produce.", "The target is a conditional distribution p(x|c). A useful model learns the relationship between condition c and generated sample x rather than only p(x).", "Generate numbers from two different conditional groups and compare their empirical means and variances.", "A generator may ignore the condition; this is called condition leakage or conditioning failure depending on the setup."),
"30": ("Out-of-Distribution and Robustness", "The world can change after training, so a model must be tested where the data distribution changes.", "Training data follow P_train(x,y); deployment may follow P_test(x,y). Robustness asks how performance changes when P_test differs.", "Train a simple threshold rule on one distribution and test it after shifting the input distribution.", "Random test splits can hide distribution shift because train and test may be too similar."),
"31": ("Transfer Learning and Fine-Tuning", "A model can reuse useful structure learned from an earlier task instead of learning everything from zero.", "Write theta_new = theta_pretrained + delta. Freezing sets selected delta values to zero; fine-tuning learns them from the new task.", "Compare a frozen representation with a small fine-tuned representation on a shifted toy task.", "Fine-tuning can destroy useful pretrained features when the new dataset is small or the learning rate is too large."),
"32": ("Scaling Laws", "Instead of guessing whether more compute helps, we can measure the curve and fit a mathematical law.", "A common toy form is L(N)=aN^{-b}+c. Taking logs can turn the power-law part into an approximately linear relationship.", "Generate controlled measurements, fit the exponent b with linear regression in log space, and compare fitted versus observed loss.", "Fitting a power law to too few points can create a convincing but unreliable exponent."),
"33": ("Inference Methods", "A trained model produces scores; a decoding rule decides what answer actually comes out.", "Temperature changes probabilities by p_i=softmax(z_i/T). Greedy chooses argmax; top-k keeps k candidates; top-p keeps the smallest probability mass whose cumulative probability reaches p.", "Use the same logits with several temperatures and implement greedy and top-k choices.", "Sampling is not the same as improving the model. A decoding trick can change outputs without changing learned parameters."),
"34": ("A Hacker's Guide to Deep Learning", "Most training bugs are ordinary software bugs wearing mathematical clothing.", "A training loop is a composition: data -> forward pass -> loss -> gradient -> update. A bug in any edge can make the final metric meaningless.", "Instrument a tiny loop with shape checks, finite-value checks, gradient norms, and loss logging.", "If the data, labels, loss, or update is wrong, changing the optimizer rarely fixes the root cause."),
"35": ("Architectural Bias and Representations", "Architecture is a set of assumptions about which patterns are easy to learn.", "An inductive bias changes the hypothesis space or optimization geometry. Convolution assumes locality and weight sharing; attention allows content-dependent interactions.", "Compare a global average with a local window statistic to see how locality changes the computation.", "An architectural bias helps when it matches reality and hurts when the assumption is wrong."),
"36": ("Metrized Deep Learning", "Metrics turn vague questions about representation quality into measurable quantities.", "For vectors u and v, cosine similarity is u dot v/(||u|| ||v||); Euclidean distance is ||u-v||_2. Different metrics can rank neighbors differently.", "Compute a pairwise distance matrix, inspect nearest neighbors, and compare Euclidean and cosine rankings.", "A metric is a measurement choice, not a universal truth; normalization can change conclusions."),
"37": ("Preference Learning and Policy Optimization", "A model can be trained from choices such as 'answer A is better than answer B'.", "A simple pairwise preference model uses L=-log sigma(r_winner-r_loser), where r is a learned reward and sigma is the logistic function.", "Fit rewards to a small set of pairwise preferences and inspect whether the preferred item receives the larger score.", "Optimizing a proxy reward can exploit weaknesses in the reward model; preference data can also be noisy or inconsistent."),
}


def title(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    m = re.search(r"^# Blog \d{2} — (.+)$", text, re.M)
    return m.group(1).strip() if m else path.stem


def bridge(num: int, current: str, previous: str | None, following: str | None) -> str:
    prev = f"**Previous lesson:** Blog {num-1:02d} — {previous}." if previous else "This is the starting point of the course."
    nxt = f"**Next lesson:** Blog {num+1:02d} — {following}." if following else "You have reached the end of the current learning path."
    return f"## 🧭 Where this lesson fits\n\n{prev}\n\n**Today:** Blog {num:02d} — {current}.\n\n{nxt}\n\n**Student rule:** understand the story first, then the equation, then the code."


def build(path: Path, previous: str | None, following: str | None) -> str:
    num = int(path.name[:2])
    current, intuition, math, experiment, failure = DETAILS[f"{num:02d}"]
    notebook = path.name.replace('.md', '.ipynb')
    branch = 'reorg/class8-to-phd-curriculum'
    return textwrap.dedent(f'''\
    # Blog {num:02d} — {current}

    <!-- NOTEBOOK-LAB-NAV -->

    ## 🧪 Interactive Lab

    The matching notebook is the hands-on laboratory for this lesson. It should be run after reading the mathematical example below.

    **[📓 Open the notebook on GitHub](https://github.com/manish7725/deeplearning/blob/{branch}/notebooks/{notebook})**  · **[▶ Open in Google Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/{branch}/notebooks/{notebook})**

    {bridge(num, current, previous, following)}

    ## 1. The idea in one sentence

    {intuition}

    ## 2. Explain it like I am in Class 8

    Imagine you are learning a new game. You first notice a pattern, then describe that pattern with a few numbers, and finally test whether your description still works when the example changes. Deep learning does the same thing, but the description can contain millions of learned numbers.

    **Three questions:**
    1. What information goes in?
    2. What transformation turns it into a useful representation?
    3. What evidence tells us whether the transformation worked?

    ## 3. The mathematics

    {math}

    Do not memorize the symbols yet. Translate every symbol into a sentence. In the notebook, we use tiny arrays so every multiplication, average, similarity, or probability can be inspected.

    ## 4. Hand calculation

    Before running Python, make a two- or three-number example yourself. Write every intermediate value. Then run the notebook and compare your answer with the program.

    This is an important rule for this curriculum: **the computer checks your reasoning; it does not replace your reasoning.**

    ## 5. Build the smallest experiment

    {experiment}

    The matching notebook implements this as a reproducible experiment with NumPy and, where appropriate, PyTorch. Change exactly one variable first. Predict the result before running the cell.

    ## 6. What can go wrong?

    **Failure mode:** {failure}

    Also check:
    - shapes and dimensions;
    - numerical stability and finite values;
    - whether the metric actually measures the intended behavior;
    - whether the result changes when the random seed or data split changes.

    ## 7. From intuition to research

    A research question is a question whose answer is not known from the lesson alone. Examples:

    - Which assumption in this method matters most?
    - What happens when that assumption is deliberately broken?
    - Which metric changes first when the model fails?
    - Does the result survive a different dataset, seed, or architecture?

    For graduate work, turn one question into a falsifiable hypothesis, define a baseline, predefine the metric, run controlled ablations, and report uncertainty rather than only the best run.

    ## 8. Mini-project

    Build a toy version using at least two competing methods. Keep the data fixed, change only the method, and record the result in a small table containing **hypothesis, setup, metric, result, interpretation**.

    ## 9. Mastery gate

    You are ready to continue when you can:

    - explain the idea without using jargon;
    - calculate the tiny example by hand;
    - run and modify the notebook;
    - explain one failure mode;
    - connect the lesson to the next chapter;
    - propose one experiment that could prove your explanation wrong.

    ## What to remember

    > **Deep learning is not a collection of APIs. It is mathematics, computation, experiments, and evidence connected into one learning process.**
    ''').strip() + '\n'


def main() -> None:
    blogs = sorted(BLOGS.glob('*.md'))
    for i, path in enumerate(blogs):
        num = path.name[:2]
        if num not in DETAILS:
            continue
        if len(path.read_text(encoding='utf-8')) >= 2500:
            continue
        previous = title(blogs[i-1]) if i else None
        following = title(blogs[i+1]) if i + 1 < len(blogs) else None
        path.write_text(build(path, previous, following), encoding='utf-8')
        print('upgraded', path)


if __name__ == '__main__':
    main()
