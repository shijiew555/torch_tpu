import sys, copy, time, torch, torch_tpu, traceback
import transformers as T
torch.manual_seed(0)
dec=dict(vocab_size=256,hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128)
def mk(name):
    if name=="mistral": return T.MistralForCausalLM(T.MistralConfig(**dec, sliding_window=8)), "lm"
    if name=="gemma2": return T.Gemma2ForCausalLM(T.Gemma2Config(**dec, head_dim=16, sliding_window=8)), "lm"
    if name=="gemma3": return T.Gemma3ForCausalLM(T.Gemma3TextConfig(**dec, head_dim=16, sliding_window=8)), "lm"
    if name=="phi3": return T.Phi3ForCausalLM(T.Phi3Config(**dec, pad_token_id=0)), "lm"
    if name=="mixtral": return T.MixtralForCausalLM(T.MixtralConfig(**dec, num_local_experts=4, num_experts_per_tok=2)), "lm"
    if name=="qwen3moe": return T.Qwen3MoeForCausalLM(T.Qwen3MoeConfig(**dec, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32, head_dim=16)), "lm"
    if name=="qwen3": return T.Qwen3ForCausalLM(T.Qwen3Config(**dec, head_dim=16)), "lm"
    if name=="roberta": return T.RobertaForSequenceClassification(T.RobertaConfig(vocab_size=256,hidden_size=64,num_hidden_layers=2,num_attention_heads=4,intermediate_size=128,max_position_embeddings=130,num_labels=3)), "cls"
    if name=="deberta": return T.DebertaV2ForSequenceClassification(T.DebertaV2Config(vocab_size=256,hidden_size=64,num_hidden_layers=2,num_attention_heads=4,intermediate_size=128,max_position_embeddings=128,num_labels=3)), "cls"
    if name=="vit": return T.ViTForImageClassification(T.ViTConfig(hidden_size=64,num_hidden_layers=2,num_attention_heads=4,intermediate_size=128,image_size=32,patch_size=8,num_labels=10)), "img"
    if name=="clip": return T.CLIPModel(T.CLIPConfig(text_config=dict(vocab_size=256,hidden_size=64,num_hidden_layers=2,num_attention_heads=4,intermediate_size=128),vision_config=dict(hidden_size=64,num_hidden_layers=2,num_attention_heads=4,intermediate_size=128,image_size=32,patch_size=8))), "clip"
    if name=="whisper": return T.WhisperForConditionalGeneration(T.WhisperConfig(vocab_size=256,d_model=64,encoder_layers=2,decoder_layers=2,encoder_attention_heads=4,decoder_attention_heads=4,encoder_ffn_dim=128,decoder_ffn_dim=128,num_mel_bins=16,max_source_positions=32,max_target_positions=64,decoder_start_token_id=1,pad_token_id=0)), "whisper"
    if name=="bart": return T.BartForConditionalGeneration(T.BartConfig(vocab_size=256,d_model=64,encoder_layers=2,decoder_layers=2,encoder_attention_heads=4,decoder_attention_heads=4,encoder_ffn_dim=128,decoder_ffn_dim=128,max_position_embeddings=128)), "s2s"
    if name=="convnext": return T.ConvNextForImageClassification(T.ConvNextConfig(hidden_sizes=[16,32,64,128],depths=[1,1,1,1],num_labels=10,image_size=32)), "img"
    if name=="resnet": return T.ResNetForImageClassification(T.ResNetConfig(hidden_sizes=[16,32,64,128],depths=[1,1,1,1],embedding_size=16,num_labels=10)), "img"
def inputs(kind):
    g=torch.Generator().manual_seed(1)
    ids=torch.randint(3,256,(2,16),generator=g); mask=torch.ones_like(ids); mask[1,:4]=0
    if kind=="lm": return dict(input_ids=ids,attention_mask=mask,labels=ids)
    if kind=="cls": return dict(input_ids=ids,attention_mask=mask,labels=torch.tensor([0,2]))
    if kind=="img": return dict(pixel_values=torch.randn(2,3,32,32,generator=g),labels=torch.tensor([1,7]))
    if kind=="clip": return dict(input_ids=ids,attention_mask=mask,pixel_values=torch.randn(2,3,32,32,generator=g),return_loss=True)
    if kind=="whisper": return dict(input_features=torch.randn(2,16,64,generator=g),labels=torch.randint(3,256,(2,8),generator=g))
    if kind=="s2s": return dict(input_ids=ids,attention_mask=mask,labels=torch.randint(3,256,(2,8),generator=g))
def nodrop(m):
    for mod in m.modules():
        if isinstance(mod, torch.nn.Dropout): mod.p=0.0
    for k in ["attention_dropout","dropout","hidden_dropout_prob","attention_probs_dropout_prob","activation_dropout"]:
        for c in [m.config]+[getattr(m.config,s,None) for s in ["text_config","vision_config"]]:
            if c is not None and hasattr(c,k): setattr(c,k,0.0)
    for mod in m.modules():
        for a in ["dropout","attention_dropout"]:
            if isinstance(getattr(mod,a,None),float): setattr(mod,a,0.0)
if __name__=='__main__':
  mode=sys.argv[2] if len(sys.argv)>2 else "eager"
  for name in [n for n in sys.argv[1].split(",") if n]:
      try:
          torch.manual_seed(0); m,kind=mk(name); nodrop(m); m.train(); mt=copy.deepcopy(m).to("tpu")
          b=inputs(kind)
          oc=m(**b); oc.loss.backward()
          f=torch.compile(mt) if mode=="compile" else mt
          t=time.time(); ot=f(**{k:(v.to("tpu") if torch.is_tensor(v) else v) for k,v in b.items()}); ot.loss.backward()
          pairs=[(n,p1.grad,p2.grad.cpu()) for (n,p1),p2 in zip(m.named_parameters(),mt.parameters()) if p1.grad is not None and p2.grad is not None]
          rel=[(((a-b).norm()/(a.norm()+1e-3*max(x[1].norm() for x in pairs))).item(),n) for n,a,b in pairs]
          gd,worst=max(rel)
          nmiss=sum(1 for p1,p2 in zip(m.parameters(),mt.parameters()) if (p1.grad is None)!=(p2.grad is None))
          print(f"{name} {mode} loss cpu={oc.loss.item():.5f} tpu={ot.loss.item():.5f} worst_rel_grad_norm_diff={gd:.2e}({worst}) grad_none_mismatch={nmiss} t={time.time()-t:.1f}s", flush=True)
      except Exception as e:
          traceback.print_exc(); print(f"{name} {mode} FAIL {type(e).__name__}: {str(e)[:300]}", flush=True)
