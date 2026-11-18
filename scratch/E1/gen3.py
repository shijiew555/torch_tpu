import copy, torch, torch_tpu
from train_eager import build
torch.manual_seed(0)
base=build("llama").eval(); mt=copy.deepcopy(base).to("tpu")
g=torch.Generator().manual_seed(1)
ids=torch.randint(3,256,(2,10),generator=g); mask=torch.ones_like(ids); mask[1,:3]=0; ids[1,:3]=0
cases={
 "rep_penalty":dict(repetition_penalty=1.3),
 "no_repeat_ngram":dict(no_repeat_ngram_size=2),
 "bad_words":dict(bad_words_ids=[[83],[159]]),
 "min_new+eos":dict(min_new_tokens=5,eos_token_id=83),
 "eos_stop":dict(eos_token_id=[83,159]),
 "num_return_beam":dict(num_beams=4,num_return_sequences=2,length_penalty=0.5),
 "suppress":dict(suppress_tokens=[83,159,1]),
 "hidden_states":dict(output_hidden_states=True,return_dict_in_generate=True),
 "logprobs":dict(output_scores=True,return_dict_in_generate=True),
}
for k,kw in cases.items():
    try:
        kk={**dict(max_new_tokens=15,do_sample=False,pad_token_id=0,eos_token_id=None),**kw}
        oc=base.generate(input_ids=ids,attention_mask=mask,**kk)
        ot=mt.generate(input_ids=ids.to("tpu"),attention_mask=mask.to("tpu"),**kk)
        sc=oc.sequences if hasattr(oc,"sequences") else oc; st=(ot.sequences if hasattr(ot,"sequences") else ot).cpu()
        print(f"{k}: match={torch.equal(sc,st)} shape={tuple(st.shape)}", flush=True)
        if not torch.equal(sc,st): print("  cpu",sc.tolist(),"\n  tpu",st.tolist())
    except Exception as e:
        print(f"{k}: FAIL {type(e).__name__}: {str(e)[:250]}", flush=True)
