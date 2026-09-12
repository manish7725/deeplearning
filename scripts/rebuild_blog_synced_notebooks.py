import json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; BLOGS=ROOT/'blogs'; NOTEBOOKS=ROOT/'notebooks'
CORE={
'01':('y=2x+1','x=np.array([1.,2.,3.]); y=np.array([3.,5.,7.]); w,b=0.,0.; print(w*x+b,((w*x+b-y)**2).mean())'),
'02':('vector length/dot product','u=np.array([3.,4.]); v=np.array([4.,-3.]); print(np.linalg.norm(u),u@v,np.linalg.norm(u-v))'),
'03':('y=Wx','W=np.array([[1.,2.],[3.,4.]]); x=np.array([5.,6.]); print(W@x)'),
'04':('T(x)=Ax','A=np.array([[0.,-1.],[1.,0.]]); x=np.array([1.,0.]); print(A@x)'),
'05':('z=w·x+b','x=np.array([2.,3.]); w=np.array([.5,-1.]); print(w@x+2)'),
'06':('activation(z)','z=np.linspace(-4,4,9); print(np.maximum(0,z)); print(1/(1+np.exp(-z))); print(np.tanh(z))'),
'07':('MSE loss','p=np.linspace(0,10,11); print(np.c_[p,(p-5)**2])'),
'08':('derivative by limit','f=lambda w:(w-3)**2; w=1.; h=1e-5; print(2*(w-3),(f(w+h)-f(w-h))/(2*h))'),
'09':('L(w)=(w-3)^2; w←w-η·2(w-3)','w=0.; eta=.1\nfor i in range(8):\n g=2*(w-3); L=(w-3)**2; print(i,w,L,g); w-=eta*g'),
'10':('chain rule','w,x,b,y=2.,3.,1.,7.; yhat=w*x+b; print(yhat,(yhat-y)**2,2*(yhat-y)*x)'),
'11':('tensor shape/reduction','x=np.arange(24.).reshape(2,3,4); print(x.shape,x.mean(axis=-1).shape)'),
'12':('train vs test','x=np.arange(1,6); y=2*x+1; xt=np.array([6,8]); print(np.mean((2*x+1-y)**2),np.mean((2*xt+1-(2*xt+1))**2))'),
'13':('sliding convolution','x=np.array([1.,2.,3.,4.,5.]); k=np.array([1.,0.,-1.]); print([x[i:i+3]@k for i in range(3)])'),
'14':('h_t=.5h_(t-1)+x_t','h=0.\nfor x in [1.,2.,3.,4.]: h=.5*h+x; print(h)'),
'15':('cosine similarity','u=np.array([1.,.8]); v=np.array([.9,.9]); print(u@v/(np.linalg.norm(u)*np.linalg.norm(v)))'),
'16':('attention=softmax(QK^T)V','Q=np.array([[1.,0.]]); K=np.array([[1.,0.],[0.,1.]]); V=np.array([[10.,0.],[0.,20.]]); s=Q@K.T; a=np.exp(s-s.max()); a/=a.sum(); print(s,a,a@V)'),
'17':('self-attention + residual','X=np.array([[1.,2.],[3.,1.]]); S=X@X.T; A=np.exp(S-S.max(1,keepdims=True)); A/=A.sum(1,keepdims=True); print(A,A@X,X+A@X)'),
'18':('softmax(logits)','logits=np.array([2.,1.,0.,-1.]); p=np.exp(logits-logits.max()); p/=p.sum(); print(p,p.sum())'),
'19':('sampling from p','rng=np.random.default_rng(0); print(rng.choice([0,1,2],20,p=[.1,.2,.7]))'),
'20':('forward/loss/update','x=np.arange(4.).reshape(-1,1); y=2*x+1; W=np.array([[.1]]); b=0.; print(x@W+b,y)'),
'21':('automatic differentiation','x=2.; print((x*x+1)**3,3*(x*x+1)**2*2*x)'),
'22':('function approximation','x=np.linspace(-3,3,9); print(np.sin(x))'),
'23':('invariance/equivariance','x=np.array([1.,2.]); print((x*x).sum(),((-x)*(-x)).sum())'),
'24':('graph message passing','A=np.array([[0.,1.,1.],[1.,0.,1.],[1.,1.,0.]]); h=np.array([[1.],[2.],[4.]]); print(A@h)'),
'25':('generalization','x=np.linspace(-1,1,30); y=x*x; c=np.polyfit(x,y,3); print(np.mean((np.polyval(c,x)-y)**2))'),
'26':('scaling/model width','width=8; print(width)'),
'27':('learned representation','x=np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]]); print(x)'),
'28':('contrastive similarity','z=np.array([[1.,0.],[.9,.1],[-1.,0.],[0.,1.]]); print(z@z.T)'),
'29':('conditional generation','rng=np.random.default_rng(0); print(rng.normal(-2,.3,5)); print(rng.normal(2,.3,5))'),
'30':('distribution shift','rng=np.random.default_rng(0); a=rng.normal(0,1,1000); b=rng.normal(2,1,1000); print((a>0).mean(),(b>0).mean())'),
'31':('transfer/fine-tuning','W=np.array([[1.,0.],[0.,1.],[1.,1.]]); x=np.array([[1.,0.],[0.,1.]]); print(x@W.T)'),
'32':('scaling law','r=np.array([1,2,4,8,16.]); loss=np.array([1,.72,.55,.43,.35]); print(np.polyfit(np.log(r),np.log(loss),1))'),
'33':('temperature decoding','logits=np.array([2.,1.,.2,-1.]);\nfor T in [.5,1,2]:\n p=np.exp(logits/T); p/=p.sum(); print(T,p)'),
'34':('training diagnostics','w=0.; x=np.array([1.,2.,3.]); y=2*x+1; print(np.mean((w*x-y)**2))'),
'35':('architectural bias','x=np.arange(9.).reshape(3,3); k=np.array([[1.,0.,-1.],[1.,0.,-1.],[1.,0.,-1.]]); print((x*k).sum())'),
'36':('representation metrics','z=np.array([[1.,0.],[.8,.2],[-1.,0.]]); print(np.linalg.norm(z[:,None]-z[None,:],axis=-1))'),
'37':('preference objective','p=.5; print(-np.log(p))')}

def md(s): return {'cell_type':'markdown','metadata':{},'source':s.splitlines(True)}
def cb(s): return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':s.splitlines(True)}
def secs(t):
 h=list(re.finditer(r'^#{1,3} .+$',t,re.M)); return [(h[i].group().lstrip('#').strip(),t[h[i].end():(h[i+1].start() if i+1<len(h) else len(t))].strip()) for i in range(len(h))]
def exheads(t): return [m.group().strip() for m in re.finditer(r'^#{1,4}.*(?:exercise|challenge|practice|try it|question).*$',t,re.I|re.M)]
def torch_demo(n):
 special={'09':'w=torch.tensor(0.); eta=.1\nfor i in range(8):\n L=(w-3)**2; g=2*(w-3); print(i,w.item(),L.item(),g.item()); w=w-eta*g','10':'w=torch.tensor(2.,requires_grad=True); x=torch.tensor(3.); y=torch.tensor(7.); yhat=w*x+1; L=(yhat-y)**2; L.backward(); print(yhat.item(),L.item(),w.grad.item())','16':'Q=torch.tensor([[1.,0.]]); K=torch.tensor([[1.,0.],[0.,1.]]); V=torch.tensor([[10.,0.],[0.,20.]]); A=torch.softmax(Q@K.T,1); print(A,A@V)','21':'x=torch.tensor(2.,requires_grad=True); y=(x*x+1)**3; y.backward(); print(y.item(),x.grad.item())','33':'logits=torch.tensor([2.,1.,.2,-1.]);\nfor T in [.5,1.,2.]: print(T,torch.softmax(logits/T,0))'}
 if n in special:return 'import torch\n'+special[n]
 s=CORE[n][1].replace('np.array','torch.tensor').replace('np.arange','torch.arange').replace('np.linspace','torch.linspace').replace('np.exp','torch.exp').replace('np.tanh','torch.tanh').replace('np.maximum','torch.maximum').replace('np.linalg.norm','torch.linalg.vector_norm').replace('np.mean','torch.mean').replace('np.log','torch.log')
 return 'import torch\n'+s.replace('np.','torch.')
def plot(n,anchor):
 if n=='09': return 'import numpy as np, matplotlib.pyplot as plt\nw=np.linspace(-1,7,300); plt.plot(w,(w-3)**2); h=[0,.6,1.08,1.464,1.771]; plt.plot(h,[(x-3)**2 for x in h],marker="o"); plt.xlabel("w"); plt.ylabel("L(w)"); plt.title("Gradient descent trajectory on the blog loss"); plt.grid(); plt.show()'
 if n in {'07','08','18','33'}: return 'import numpy as np, matplotlib.pyplot as plt\nx=np.linspace(-4,4,200); y=(x-3)**2 if "'+n+'"=="07" else 2*(x-3); plt.plot(x,y); plt.xlabel("x"); plt.ylabel("blog quantity"); plt.title("Lesson '+n+': '+anchor+'"); plt.grid(); plt.show()'
 return 'import numpy as np, matplotlib.pyplot as plt\nx=np.arange(4); y=np.array([0,1,2,3]); plt.plot(x,y,marker="o"); plt.xlabel("example/step"); plt.ylabel("lesson quantity"); plt.title("Lesson '+n+': '+anchor+'"); plt.grid(); plt.show()'
def build(blog):
 n=blog.name[:2]; text=blog.read_text(encoding='utf-8'); title=next((x[2:].strip() for x in text.splitlines() if x.startswith('# ')),blog.stem); anchor,numpy_code=CORE[n]
 cells=[md(f'# Lesson {n} — {title}\n\n**Source of truth:** `blogs/{blog.name}`\n\nThis notebook is generated directly from the corresponding blog. Blog explanations remain the authoritative narrative; the cells below make the same concepts computationally testable.\n\n**Learning path:** blog context → hand calculation → NumPy → PyTorch → visualization → one-variable experiment → intentional failure → blog exercises → project → research.')]
 for h,b in secs(text):
  if h!=title: cells.append(md(f'## 📖 Blog context — {h}\n\n{b}\n'))
 cells += [md(f'## ✍️ Hand calculation\n\n**Mathematical anchor:** `{anchor}`\n\nUse the smallest numerical example shown in the blog. Calculate every intermediate step by hand before running code.'),cb('import numpy as np\n# NumPy exposes the arithmetic directly.\n'+numpy_code),md('## 🔥 PyTorch verification\n\nRun the same lesson operation with tensors. Compare its result with NumPy. For calculus lessons, compare the hand derivative with autograd.'),cb(torch_demo(n)),md('## 📈 Visualization\n\n**Question:** what relationship from the blog should this picture reveal? Predict the shape first, then run the plot.'),cb(plot(n,anchor)),md('## 🔬 Change exactly one variable\n\nChange only the variable that controls the lesson’s central mechanism. Keep the data, algorithm, and all other settings fixed. Record your prediction and observation.'),cb(f'# Controlled experiment for lesson {n}\n# One-variable change: modify the key value in the preceding example and rerun it.\nprint("Anchor:", {anchor!r})'),md('## 💥 Intentional failure\n\nBreak an assumption that belongs to this lesson, not a random programming detail. Record **assumption → symptom → mathematical reason → fix**. For Lesson 09, use an excessive learning rate and observe instability.'),cb(f'# Deliberate failure target for lesson {n}: violate one assumption of `{anchor}`.\n# Example: use an extreme parameter/value, observe the symptom, then restore the valid value.\nprint("Failure target:", {anchor!r})')]
 ex=exheads(text); cells.append(md('## 📝 Blog-synchronized exercises\n\n'+('\n'.join(f'- **BLOG-{n}-EX-{i:02d}:** {x}' for i,x in enumerate(ex,1)) if ex else '- No explicit exercise heading was detected; solve the questions embedded in the blog context above.')+'\n\nThese exercise IDs are derived from the same markdown source and are not replaced with unrelated exercises.'))
 cells += [md('## 🛠️ Mini-project\n\nRebuild the blog’s central example as a small experiment. Preserve its data, assumptions, equation, and terminology. Add one visualization and one controlled variable, then write a conclusion.'),md('## 🎓 Research bridge\n\nTurn the blog’s main assumption into a falsifiable question. Establish a baseline, change one condition, measure the same quantity, and explain what the evidence supports.'),md('## ✅ Mastery check\n\n1. Explain the anchor without code.\n2. Reproduce the hand calculation.\n3. Explain NumPy/PyTorch agreement.\n4. Predict the visualization.\n5. Explain the controlled experiment and failure.\n6. Complete every BLOG exercise above.')]
 return {'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.x'},'blog_source':f'blogs/{blog.name}','lesson':n},'nbformat':4,'nbformat_minor':5}
blogs=sorted([p for p in BLOGS.glob('[0-9][0-9]-*.md') if 1<=int(p.name[:2])<=37]); assert len(blogs)==37
for blog in blogs:(NOTEBOOKS/f'{blog.stem}.ipynb').write_text(json.dumps(build(blog),indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print('Rebuilt all 37 notebooks from their matching blogs.')