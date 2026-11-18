import sys, copy, torch, torch_tpu
torch.manual_seed(0)
if len(sys.argv)>1: torch.set_float32_matmul_precision(sys.argv[1])
def cmp(label, mod, x):
    mt=copy.deepcopy(mod).to("tpu"); xc=x.clone().requires_grad_(); xt=x.clone().to("tpu").requires_grad_()
    g=torch.randn_like(mod(x))
    yc=mod(xc); yc.backward(g); yt=mt(xt); yt.backward(g.to("tpu"))
    def rel(a,b): return ((a-b.cpu()).norm()/a.norm()).item()
    out=f"{label}: y={rel(yc,yt):.2e} dx={rel(xc.grad,xt.grad):.2e}"
    for (n,p),pt in zip(mod.named_parameters(),mt.parameters()): out+=f" d{n}={rel(p.grad,pt.grad):.2e}"
    print(out, flush=True)
x=torch.randn(2,16,8,8)
cmp("bn2d train", torch.nn.BatchNorm2d(16).train(), x)
cmp("bn2d eval", torch.nn.BatchNorm2d(16).eval(), x)
cmp("conv3x3", torch.nn.Conv2d(16,16,3,padding=1,bias=False), x)
cmp("conv+bn", torch.nn.Sequential(torch.nn.Conv2d(16,16,3,padding=1,bias=False),torch.nn.BatchNorm2d(16)), x)
cmp("conv+bn+relu", torch.nn.Sequential(torch.nn.Conv2d(16,16,3,padding=1,bias=False),torch.nn.BatchNorm2d(16),torch.nn.ReLU()), x)
cmp("linear", torch.nn.Linear(64,64), torch.randn(32,64))
