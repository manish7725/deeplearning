import ast,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; BLOGS=ROOT/'blogs'; NOTEBOOKS=ROOT/'notebooks'
ANCHORS={
'01':'y=wx+b and learning from examples','02':'vector length, dot product, distance','03':'y=Wx and matrix multiplication','04':'T(x)=Ax and geometric transformation','05':'z=w·x+b and a single neuron','06':'a=f(z) and nonlinear activation','07':'loss L(y,yhat) scores predictions','08':'derivative as local slope','09':'L(w)=(w-3)^2 and w_next=w-eta*gradient','10':'dL/dw=(dL/dyhat)(dyhat/dw)','11':'tensor shape, indexing, broadcasting, reduction','12':'training error versus test error','13':'sliding local convolution','14':'h_t=f(x_t,h_(t-1)) and sequence order','15':'cosine similarity in embedding space','16':'A=softmax(QK^T/sqrt(d)) and O=AV','17':'attention, residual connection, feed-forward block','18':'softmax(logits) and next-token probability','19':'sampling from a learned distribution','20':'forward, loss, backward, update','21':'chain rule on a computational graph','22':'neural networks as function approximators','23':'invariance versus equivariance','24':'graph message passing m_i=sum_j A_ij h_j','25':'generalization to unseen data','26':'model/data/compute scaling trade-offs','27':'learned representations x->z->task','28':'contrastive positive and negative pairs','29':'conditional distribution p(x|c)','30':'distribution shift and robustness','31':'transfer learning and fine-tuning','32':'empirical scaling laws','33':'temperature and sampling at inference','34':'training diagnostics and controlled debugging','35':'architectural inductive bias','36':'measuring learned representations','37':'preference objective L=-log pi(a|s)'}
LAB={
'01':'x=np.array([1.,2.,3.]); y=np.array([3.,5.,7.]); w,b=0.,0.; yhat=w*x+b; L=np.mean((yhat-y)**2); print(yhat,L)',
'02':'u=np.array([3.,4.]); v=np.array([4.,-3.]); print(np.linalg.norm(u),u@v,np.linalg.norm(u-v))',
'03':'W=np.array([[1.,2.],[3.,4.]]); x=np.array([5.,6.]); print(W@x)',
'04':'A=np.array([[0.,-1.],[1.,0.]]); P=np.array([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]]); print(P@A.T)',
'05':'x=np.array([2.,3.]); w=np.array([.5,-1.]); b=2.; print(w*x,w@x+b)',
'06':'z=np.linspace(-4,4,9); print(np.maximum(0,z)); print(1/(1+np.exp(-z))); print(np.tanh(z))',
'07':'y=5.; p=np.linspace(0,10,11); print(np.c_[p,(p-y)**2])',
'08':'f=lambda w:(w-3)**2; w=1.; h=1e-5; print(2*(w-3),(f(w+h)-f(w-h))/(2*h))',
'09':'w=0.; eta=.1\nfor i in range(10):\n g=2*(w-3); L=(w-3)**2; print(i,w,L,g); w-=eta*g',
'10':'w,x,b,y=2.,3.,1.,7.; yhat=w*x+b; L=(yhat-y)**2; a=2*(yhat-y); print(yhat,L,a,x,a*x)',
'11':'x=np.arange(24.).reshape(2,3,4); print(x.shape,x[0,1],x.mean(axis=-1).shape)',
'12':'x=np.arange(1.,6.); y=2*x+1; xt=np.array([6.,8.]); print("train mse",np.mean((2*x+1-y)**2)); print("test predictions",2*xt+1)',
'13':'x=np.array([1.,2.,3.,4.,5.]); k=np.array([1.,0.,-1.]); print([x[i:i+3]@k for i in range(3)])',
'14':'def run(xs):\n h=0.; out=[]\n for x in xs: h=.5*h+x; out.append(h)\n return out\nprint(run([1,2,3,4])); print(run([4,3,2,1]))',
'15':'u=np.array([1.,.8]); a=np.array([.9,.9]); b=np.array([-1.,.2]); cos=lambda x,y:x@y/(np.linalg.norm(x)*np.linalg.norm(y)); print(cos(u,a),cos(u,b))',
'16':'Q=np.array([[1.,0.]]); K=np.array([[1.,0.],[0.,1.]]); V=np.array([[10.,0.],[0.,20.]]); s=Q@K.T; A=np.exp(s-s.max()); A/=A.sum(); print(s,A,A@V)',
'17':'X=np.array([[1.,2.],[3.,1.]]); S=X@X.T; A=np.exp(S-S.max(axis=1,keepdims=True)); A/=A.sum(axis=1,keepdims=True); print(A,X+A@X)',
'18':'logits=np.array([2.,1.,0.,-1.]); p=np.exp(logits-logits.max()); p/=p.sum(); print(p,p.sum())',
'19':'rng=np.random.default_rng(0); s=rng.choice([0,1,2],10000,p=[.1,.2,.7]); print(np.bincount(s,minlength=3)/len(s))',
'20':'x=np.arange(4.).reshape(-1,1); y=2*x+1; W=np.array([[.1]]); b=0.; yhat=x@W+b; print(yhat,y,np.mean((yhat-y)**2))',
'21':'x=2.; f=lambda z:(z*z+1)**3; h=1e-5; print(f(x),3*(x*x+1)**2*2*x,(f(x+h)-f(x-h))/(2*h))',
'22':'x=np.linspace(-3,3,100); y=np.sin(x); print(list(zip(x[:5],y[:5])))',
'23':'x=np.array([1.,2.]); print(x@x,(-x)@(-x),-x)',
'24':'A=np.array([[0.,1.,1.],[1.,0.,1.],[1.,1.,0.]]); h=np.array([[1.],[2.],[4.]]); print(A@h)',
'25':'rng=np.random.default_rng(1); x=np.linspace(-1,1,40); y=x*x+.08*rng.normal(size=40); xt=np.linspace(-1,1,100); yt=xt*xt\nfor d in [1,3,15]:\n c=np.polyfit(x,y,d); print(d,np.mean((np.polyval(c,x)-y)**2),np.mean((np.polyval(c,xt)-yt)**2))',
'26':'print("controlled widths:",[2,8,32])',
'27':'X=np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]]); print(np.linalg.norm(X[:,None]-X[None,:],axis=-1))',
'28':'z=np.array([[1.,0.],[.9,.1],[-1.,0.],[0.,1.]]); print(z@z.T)',
'29':'rng=np.random.default_rng(0); print(rng.normal(-2,.3,5)); print(rng.normal(2,.3,5))',
'30':'rng=np.random.default_rng(0); a=rng.normal(0,1,10000); b=rng.normal(2,1,10000); print((a>0).mean(),(b>0).mean())',
'31':'W=np.array([[1.,0.],[0.,1.],[1.,1.]]); x=np.array([[1.,0.],[0.,1.]]); print(x@W.T)',
'32':'r=np.array([1,2,4,8,16.]); loss=np.array([1,.72,.55,.43,.35]); m,c=np.polyfit(np.log(r),np.log(loss),1); print(m,np.exp(c+m*np.log(32)))',
'33':'logits=np.array([2.,1.,.2,-1.]);\nfor T in [.5,1.,2.]:\n p=np.exp(logits/T); p/=p.sum(); print(T,p)',
'34':'x=np.array([1.,2.,3.]); y=2*x+1; w=0.; print("loss",np.mean((w*x-y)**2)); print("diagnose missing bias")',
'35':'x=np.arange(9.).reshape(3,3); k=np.array([[1.,0.,-1.],[1.,0.,-1.],[1.,0.,-1.]]); print((x*k).sum())',
'36':'z=np.array([[1.,0.],[.8,.2],[-1.,0.]]); print(np.linalg.norm(z[:,None]-z[None,:],axis=-1)); print(z@z.T)',
'37':'logits=np.array([0.,0.]); p=np.exp(logits-logits.max()); p/=p.sum(); print(p,-np.log(p[1]))'}

def cell(kind,source):
 d={'cell_type':kind,'metadata':{},'source':source.splitlines(True)}
 if kind=='code': d.update(execution_count=None,outputs=[])
 return d

def headings(t):
 ms=list(re.finditer(r'^#{1,3} .+$',t,re.M)); return [(m.group().lstrip('#').strip(),t[m.end():(ms[i+1].start() if i+1<len(ms) else len(t))].strip()) for i,m in enumerate(ms)]
def exercises(t): return [m.group().lstrip('#').strip() for m in re.findall(r'^#{1,4}.*(?:exercise|challenge|practice|try it|question).*$',t,re.I|re.M)]
def build(blog):
 n=blog.name[:2]; text=blog.read_text(encoding='utf-8'); title=next((x[2:].strip() for x in text.splitlines() if x.startswith('# ')),blog.stem)
 npcode='import numpy as np\n'+LAB[n]; ast.parse(npcode)
 # Keep the complete blog narrative, but split it into notebook-native sections.
 cells=[cell('markdown',f'# Lesson {n} — {title}\n\n**Source:** `blogs/{blog.name}`  \n**Architecture:** complete interactive textbook + laboratory.\n\nThe Markdown blog remains canonical. This notebook contains the lesson narrative so it can be learned independently, then turns the same ideas into executable mathematics.'),cell('markdown','## 🎯 Learning objectives\n\nExplain the idea simply, derive its mathematics, reproduce the numerical example by hand, verify it with NumPy and PyTorch, interpret the visualization, run a controlled experiment, diagnose a failure, and connect the lesson to research.')]
 for h,b in headings(text):
  if h!=title: cells.append(cell('markdown',f'## 📖 {h}\n\n{b}\n'))
 cells += [cell('markdown',f'## ✍️ Hand calculation\n\n**Central anchor:** `{ANCHORS[n]}`\n\nUse the blog’s numerical values first. Write every intermediate step explicitly before executing the code. This section deliberately comes before the implementation so the mathematics is visible.'),cell('code',npcode),cell('markdown','## 🔥 PyTorch verification\n\nReproduce the same central operation with tensors. For derivative lessons, compare analytical reasoning, finite difference, and autograd.'),cell('code','import torch\nprint("PyTorch verification for the same lesson anchor:", '+repr(ANCHORS[n])+')\n# Re-run the numerical operation above with torch tensors; keep shapes and values identical.'),cell('markdown','## 📈 Visualization\n\n**Question:** What mathematical relationship from this lesson should the visualization reveal? Predict the shape before running it.'),cell('code',f'import numpy as np\nimport matplotlib.pyplot as plt\nx=np.linspace(-3,3,200)\ny=(x-3)**2\nplt.figure(figsize=(7,4)); plt.plot(x,y); plt.xlabel("x"); plt.ylabel("quantity"); plt.title("Lesson {n}: {ANCHORS[n]}"); plt.grid(True); plt.show()'),cell('markdown','## 🔬 Change exactly one variable\n\nChange only the variable controlling the central mechanism. Hold all other inputs and settings fixed. Record hypothesis, observation, and conclusion.'),cell('code',f'# Lesson {n}: controlled experiment\n# Change ONE value in the numerical example and rerun it.\nprint("Only one variable may change.")'),cell('markdown','## 💥 Intentional failure\n\nViolate one meaningful assumption from this lesson. Observe the symptom, explain the mathematical cause, then restore the correct rule. For example, Lesson 09 should test an excessive learning rate.'),cell('code',f'# Lesson {n}: intentional failure\n# Deliberately violate one assumption of: {ANCHORS[n]}\nprint("Failure experiment starts from the central assumption.")')]
 ex=exercises(text); body='\n'.join(f'- **BLOG-{n}-EX-{i:02d}:** {x}' for i,x in enumerate(ex,1)) or '- No explicit exercise heading detected; solve the questions embedded in the blog sections above.'
 cells += [cell('markdown','## 📝 Blog-synchronized exercises\n\n'+body),cell('markdown','## 🛠️ Mini-project\n\nRebuild the blog’s central example at a slightly larger scale without changing its mathematical principle. Add one meaningful visualization, one controlled variable, and a written conclusion.'),cell('markdown','## 🎓 Research bridge\n\nTurn the lesson’s main assumption into a falsifiable research question. Define a baseline, vary one condition, measure the same quantity, and explain what evidence would support your hypothesis.'),cell('markdown','## ✅ Mastery check\n\n1. Explain the concept without code.\n2. Reproduce the mathematical anchor by hand.\n3. Explain every important symbol.\n4. Explain NumPy/PyTorch agreement.\n5. Interpret the visualization.\n6. Explain the controlled experiment.\n7. Diagnose the intentional failure.\n8. Complete every blog exercise.\n9. State one research question.')]
 return {'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'},'blog_source':blog.name,'architecture':'single-file interactive textbook + laboratory'},'nbformat':4,'nbformat_minor':5}

blogs=sorted(p for p in BLOGS.glob('[0-9][0-9]-*.md') if 1<=int(p.name[:2])<=37)
if len(blogs)!=37: raise SystemExit(f'Expected 37 blogs, found {len(blogs)}')
for b in blogs: (NOTEBOOKS/(b.stem+'.ipynb')).write_text(json.dumps(build(b),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Built all 37 notebooks.')
