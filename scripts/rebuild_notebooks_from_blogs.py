import json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; BLOGS=ROOT/'blogs'; NOTEBOOKS=ROOT/'notebooks'
RECIPES={
'01':('y=2x+1; L=mean((yhat-y)^2)','learn a tiny line','import numpy as np\nx=np.array([1.,2.,3.]); y=np.array([3.,5.,7.]); w,b=0.,0.\nfor _ in range(500):\n e=w*x+b-y; w-=.05*np.mean(2*e*x); b-=.05*np.mean(2*e)\nprint(w,b)'),
'02':('v=[x1,x2]; ||v||; u·v','represent objects as points','import numpy as np\nu=np.array([3.,4.]); v=np.array([4.,-3.]); print(np.linalg.norm(u),u@v,np.linalg.norm(u-v))'),
'03':('y=Wx','perform several dot products at once','import numpy as np\nW=np.array([[1.,2.],[3.,4.]]); x=np.array([5.,6.]); print(W@x)'),
'04':('T(x)=Ax','see a matrix transform points','import numpy as np\nA=np.array([[0.,-1.],[1.,0.]]); P=np.array([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]]); print(P@A.T)'),
'05':('z=w·x+b','build the smallest neuron','import numpy as np\nx=np.array([2.,3.]); w=np.array([.5,-1.]); b=2.; print(w@x+b)'),
'06':('a=activation(w·x+b)','compare linear and nonlinear functions','import numpy as np\nz=np.linspace(-4,4,9); print("z",z); print("ReLU",np.maximum(0,z)); print("sigmoid",1/(1+np.exp(-z)))'),
'07':('L(y,yhat)','see how loss scores predictions','import numpy as np\ny=5.; p=np.linspace(0,10,11); print(np.c_[p,(p-y)**2])'),
'08':('dL/dtheta','compare derivative and finite difference','import numpy as np\nf=lambda w:(w-3)**2; w=1.; h=1e-5; print(2*(w-3),(f(w+h)-f(w-h))/(2*h))'),
'09':('theta_next=theta-eta*grad','move downhill on L(w)=(w-3)^2','import numpy as np\ndef run(eta):\n w=0.; hist=[]\n for _ in range(30): hist.append((w,(w-3)**2)); w-=eta*2*(w-3)\n return np.array(hist)\nfor eta in [.01,.1,.5,1.1]: print(eta,run(eta)[-1])'),
'10':('dL/dw=dL/dyhat·dyhat/dw','follow the chain rule','import torch\nw=torch.tensor(2.,requires_grad=True); x=torch.tensor(3.); y=(w*x+1); L=(y-7)**2; L.backward(); print(y.item(),L.item(),w.grad.item())'),
'11':('tensor shape','extend vectors and matrices to batches','import torch\nx=torch.arange(24.).reshape(2,3,4); print(x.shape); print(x.mean(dim=-1))'),
'12':('train set vs test set','evaluate on unseen examples','import numpy as np\nx=np.arange(1,6); y=2*x+1; xt=np.array([6,8]); yt=2*xt+1; w,b=0.,0.\nfor _ in range(1000):\n e=w*x+b-y; w-=.01*np.mean(2*e*x); b-=.01*np.mean(2*e)\nprint(np.mean((w*x+b-y)**2),np.mean((w*xt+b-yt)**2))'),
'13':('output[i]=sum(window*kernel)','slide a local filter','import numpy as np\nx=np.array([1.,2.,3.,4.,5.]); k=np.array([1.,0.,-1.]); print([np.sum(x[i:i+3]*k) for i in range(3)])'),
'14':('h_t=f(x_t,h_t-1)','carry information through a sequence','h=0.; states=[]\nfor x in [1.,2.,3.,4.]: h=.5*h+x; states.append(h)\nprint(states)'),
'15':('cos(u,v)=u·v/(||u||||v||)','make similarity geometric','import numpy as np\ndef c(a,b): return a@b/(np.linalg.norm(a)*np.linalg.norm(b))\nprint(c([1,.8],[.9,.9]),c([1,.8],[-1,.2]))'),
'16':('softmax(QK^T/sqrt(d))V','turn attention scores into weighted values','import torch\nQ=torch.tensor([[1.,0.]]); K=torch.tensor([[1.,0.],[0.,1.]]); V=torch.tensor([[10.,0.],[0.,20.]]); s=Q@K.T; a=torch.softmax(s,1); print(s,a,a@V)'),
'17':('tokens→attention→residual→feed-forward','assemble a tiny transformer-style block','import torch\nx=torch.tensor([[1.,2.],[3.,1.]]); A=torch.softmax(x@x.T,1); print(A); print(x+A@x)'),
'18':('p(next|context)=softmax(logits)','inspect next-token probabilities','import torch\nl=torch.tensor([2.,1.,0.,-1.]); p=torch.softmax(l,0); print(p,p.sum())'),
'19':('sample x~p_theta','sample from a learned distribution','import numpy as np\nr=np.random.default_rng(7); print(r.choice([0,1,2],20,p=[.1,.2,.7]))'),
'20':('forward→loss→backprop→update','build a complete network loop','import numpy as np\nx=np.arange(4.).reshape(-1,1); y=2*x+1; W=np.array([[.1]]); b=np.array([0.])\nfor _ in range(1000):\n e=x@W+b-y; W-=.02*(2*x*e).mean(0).reshape(1,1); b-=.02*(2*e).mean(0)\nprint(W,b)'),
'21':('chain rule on a computation graph','verify automatic differentiation','import torch\nx=torch.tensor(2.,requires_grad=True); y=(x*x+1)**3; y.backward(); print(y.item(),x.grad.item())'),
'22':('f(x)≈neural_net(x)','approximate a curved function','import torch\nx=torch.linspace(-3,3,200).unsqueeze(1); y=torch.sin(x); net=torch.nn.Sequential(torch.nn.Linear(1,16),torch.nn.Tanh(),torch.nn.Linear(16,1)); opt=torch.optim.Adam(net.parameters(),lr=.03)\nfor _ in range(300): opt.zero_grad(); L=((net(x)-y)**2).mean(); L.backward(); opt.step()\nprint(L.item())'),
'23':('f(gx)=f(x) or g f(x)','test invariance/equivariance','import torch\nx=torch.tensor([1.,2.]); g=-x; f=lambda z:(z*z).sum(); print(f(x),f(g))'),
'24':('aggregate(neighbors)→update','exchange messages on a graph','import torch\nA=torch.tensor([[0.,1.,1.],[1.,0.,1.],[1.,1.,0.]]); h=torch.tensor([[1.],[2.],[4.]]); print(A@h); print(torch.relu(h+A@h))'),
'25':('training fit→unseen data','compare train and test error','import numpy as np\nr=np.random.default_rng(1); x=np.linspace(-1,1,30); y=x*x+.05*r.normal(size=30)\nfor d in [1,3,15]:\n c=np.polyfit(x,y,d); print(d,np.mean((np.polyval(c,x)-y)**2))'),
'26':('model/data/compute trade-off','change model width and observe fit','import torch\nx=torch.linspace(-2,2,100).unsqueeze(1); y=torch.sin(x)\nfor width in [2,8,32]:\n n=torch.nn.Sequential(torch.nn.Linear(1,width),torch.nn.Tanh(),torch.nn.Linear(width,1)); o=torch.optim.SGD(n.parameters(),lr=.05)\n for _ in range(100): o.zero_grad(); L=((n(x)-y)**2).mean(); L.backward(); o.step()\n print(width,L.item())'),
'27':('raw input→learned representation→task','learn a hidden representation','import torch\nx=torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]]); y=torch.tensor([[0.],[1.],[1.],[0.]]); n=torch.nn.Sequential(torch.nn.Linear(2,4),torch.nn.Tanh(),torch.nn.Linear(4,1)); o=torch.optim.Adam(n.parameters(),lr=.05)\nfor _ in range(300): o.zero_grad(); L=((n(x)-y)**2).mean(); L.backward(); o.step()\nprint(L.item())'),
'28':('positive close; negative far','inspect contrastive similarities','import torch\nz=torch.tensor([[1.,0.],[.9,.1],[-1.,0.],[0.,1.]]); print(z@z.T)'),
'29':('p(x|condition)','generate different outputs by condition','import numpy as np\nr=np.random.default_rng(7)\nfor c,mu in [(0,-2),(1,2)]: print(c,r.normal(mu,.3,5))'),
'30':('train distribution→shifted input','measure behavior under distribution shift','import numpy as np\nr=np.random.default_rng(7); a=r.normal(0,1,1000); b=r.normal(2,1,1000); print((a>0).mean(),(b>0).mean())'),
'31':('pretrained parameters→new task','reuse features and adapt a head','import torch\nbase=torch.nn.Linear(2,3); head=torch.nn.Linear(3,1); x=torch.tensor([[1.,0.],[0.,1.]]); f=base(x).detach(); print(f); print(head(f))'),
'32':('performance vs resource','fit a simple scaling trend','import numpy as np\nc=np.array([1,2,4,8,16.]); loss=np.array([1,.72,.55,.43,.35]); q=np.polyfit(np.log(c),loss,1); print(q,np.polyval(q,np.log(32)))'),
'33':('scores→decoding rule→output','compare temperature choices','import torch\nl=torch.tensor([2.,1.,.2,-1.])\nfor T in [.5,1.,2.]: print(T,torch.softmax(l/T,0))'),
'34':('observe→isolate→test→fix','debug a learning calculation','import torch\nw=torch.tensor(0.,requires_grad=True); x=torch.tensor([1.,2.,3.]); y=2*x+1; L=((w*x-y)**2).mean(); L.backward(); print(L.item(),w.grad.item())'),
'35':('architecture→inductive bias→representation','see a local operator encode a bias','import torch\nx=torch.arange(9.).reshape(1,1,3,3); k=torch.tensor([[[[1.,0.,-1.],[1.,0.,-1.],[1.,0.,-1.]]]]); print(torch.nn.functional.conv2d(x,k))'),
'36':('representation→metric→geometry','measure distance and cosine similarity','import torch\nz=torch.tensor([[1.,0.],[.8,.2],[-1.,0.]]); print(torch.cdist(z,z)); print(torch.nn.functional.normalize(z,1)@torch.nn.functional.normalize(z,1).T)'),
'37':('preference→objective→policy update','increase probability of a preferred action','import torch\nl=torch.tensor([0.,0.],requires_grad=True); loss=-torch.log_softmax(l,0)[1]; loss.backward(); print(torch.softmax(l,0),l.grad)')}

def cell_md(s): return {'cell_type':'markdown','metadata':{},'source':s.splitlines(True)}
def cell_code(s): return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':s.splitlines(True)}
def parse_sections(text):
 out=[]; start=0; heads=list(re.finditer(r'^#{1,3} .+$',text,re.M))
 for i,m in enumerate(heads):
  if i and text[start:m.start()].strip(): out.append((heads[i-1].group().lstrip('#').strip(),text[start:m.start()].strip()))
  start=m.start()
 if heads: out.append((heads[-1].group().lstrip('#').strip(),text[start:].strip()))
 return out
def headings(text,words): return [m.group().strip() for m in re.finditer(r'^#{1,4}.*(?:'+words+').*$',text,re.I|re.M)]
def build(blog):
 num=blog.name[:2]; text=blog.read_text(encoding='utf-8'); title=next((x[2:].strip() for x in text.splitlines() if x.startswith('# ')),blog.stem)
 cells=[cell_md(f'# Lesson {num} — {title}\n\n**Source of truth:** `blogs/{blog.name}`\n\nThis notebook is a computational reconstruction of that exact blog. Every major idea below is kept in the blog’s terminology and context.\n\n**Learning loop:** story → intuition → hand calculation → NumPy → PyTorch → visualization → one-variable experiment → intentional failure → blog exercises → mini-project → research.\n')]
 for h,b in parse_sections(text):
  if h!=title: cells.append(cell_md(f'## 📖 Blog context — {h}\n\n{b}\n'))
 eq,desc,code=RECIPES.get(num,('chapter equation','chapter-specific numerical laboratory','print("Add the chapter calculation here")'))
 cells += [cell_md(f'## ✍️ Hand calculation\n\n**Mathematical anchor:** `{eq}`\n\n**Task:** use the smallest numerical values from the blog and calculate the result by hand. Write every intermediate step. Then verify it below.\n\n**Why:** the notebook must reproduce the blog’s mathematics, not replace it with a generic example.'),cell_code(code),cell_md('## 🔥 PyTorch verification\n\nRepeat the same mathematical operation with tensors. Compare the numerical result with the NumPy/reference calculation above. For derivative chapters, also compare analytical, finite-difference, and autograd gradients.'),cell_code('import torch\n# Translate the smallest calculation above into tensors.\n# Keep the numbers identical to the blog example when possible.\nprint(torch.tensor([1.,2.,3.]))')]
 cells += [cell_md('## 📈 Visualization\n\nBefore running the plot, predict what the mathematics says should happen. Then visualize the chapter quantity. A plot is evidence: explain its shape, direction, slope, distance, probability, or trajectory.'),cell_code("import matplotlib.pyplot as plt\n# Build a visualization from the chapter-specific values above.\n# Keep it tied to the blog's mathematical question.\nplt.figure(figsize=(7,4)); plt.grid(); plt.title('Mathematical prediction from the blog'); plt.show()"),cell_md('## 🔬 Change exactly one variable\n\nChange ONE variable that matters to this chapter. Predict the result first, run it, and explain why it changed. Do not change several variables simultaneously.'),cell_code("experiment_value=1.0\nprint('Change only experiment_value:',experiment_value)"),cell_md('## 💥 Intentional failure\n\nBreak the actual assumption taught in this blog. Record: **changed assumption → symptom → mathematical reason → fix**. Examples include excessive learning rate, incompatible shapes, extreme logits, data leakage, large distribution shift, wrong target, or unstable optimization—choose the one that belongs to this chapter.'),cell_code("broken_value=None\nprint('Set broken_value to a deliberately bad chapter-specific value, then explain the failure.')")]
 ex=headings(text,'exercise|challenge|practice|try it|question'); exlines='\n'.join(f'- **BLOG-{num}-EX-{i:02d}:** {h}' for i,h in enumerate(ex,1)) or '- No explicit exercise heading detected; use the questions embedded in the blog context above.'
 cells.append(cell_md(f'## 📝 Blog-synchronized exercises\n\nThese are extracted from this exact blog. They must not be replaced by unrelated exercises.\n\n{exlines}'))
 res=headings(text,'research|paper|reproduction|future work'); cells.append(cell_md('## 🛠️ Mini-project\n\nBuild an extension of the blog’s central example. Include a hypothesis, implementation, visualization, one controlled variable change, one deliberate failure, and a written conclusion. If the blog specifies a project, follow that project rather than switching topics.\n\n## 🎓 Research bridge\n\n'+(('Research-related sections in the blog: '+', '.join(res)) if res else 'Formulate a falsifiable question from the blog’s main assumption: **If I change X while holding Y and Z fixed, does metric M change? Why?** Do not invent a paper citation unless the blog provides one.')))
 cells.append(cell_md('## ✅ Mastery check\n\n- Can I explain the blog’s story and intuition?\n- Can I reproduce its smallest numerical example by hand?\n- Can I map every important equation to code?\n- Do NumPy/PyTorch implement the same idea?\n- Can I predict the visualization?\n- Did I change exactly one variable?\n- Did I deliberately break the relevant assumption?\n- Did I complete every blog exercise?\n- Can I build the mini-project and formulate a research question?'))
 return {'cells':cells,'metadata':{'language_info':{'name':'python'},'blog_source':blog.name,'generated_by':'blog-driven-rebuild'},'nbformat':4,'nbformat_minor':5}

def main():
 NOTEBOOKS.mkdir(exist_ok=True); report=[]
 for blog in sorted(BLOGS.glob('[0-9][0-9]-*.md')):
  out=NOTEBOOKS/(blog.stem+'.ipynb'); out.write_text(json.dumps(build(blog),indent=2,ensure_ascii=False)+'\n',encoding='utf-8'); report.append((blog.name,len(json.loads(out.read_text())['cells'])))
 print(f'Rebuilt {len(report)} notebooks from matching blogs.')
 for x,c in report: print(x,c)
if __name__=='__main__': main()
