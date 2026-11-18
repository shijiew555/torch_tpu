cd /root/torch_tpu/.claude/tmp/E3

G1=gather,scatter,scatter_add,scatter_reduce,index_select,take_along_dim,index_put,masked_fill,masked_scatter,where,cumsum,cumprod,cummax,logcumsumexp,sort,topk,nn.functional.embedding,nn.functional.cross_entropy,nn.functional.nll_loss,nn.functional.mse_loss,nn.functional.l1_loss,nn.functional.binary_cross_entropy,nn.functional.binary_cross_entropy_with_logits,nn.functional.kl_div
G2=nn.functional.layer_norm,nn.functional.batch_norm,nn.functional.instance_norm,nn.functional.rms_norm,nn.functional.normalize,nn.functional.conv1d,nn.functional.conv2d,nn.functional.conv_transpose2d,nn.functional.max_pool2d,nn.functional.avg_pool2d,nn.functional.adaptive_avg_pool2d,nn.functional.adaptive_max_pool2d,nn.functional.interpolate,nn.functional.pad,softmax,log_softmax,nn.functional.gelu,nn.functional.silu
G3=matmul,bmm,baddbmm,addmm,einsum,linalg.vector_norm,norm,sum,mean,amax,amin,prod,var,std,logsumexp,clamp,repeat_interleave,roll,flip,cat,stack,unfold,cdist,nn.functional.cosine_similarity,nn.functional.grid_sample,nn.functional.unfold,nn.functional.pixel_shuffle,diff,cross,outer,nn.functional.triplet_margin_loss,nn.functional.gaussian_nll_loss,nn.functional.poisson_nll_loss,nn.functional.margin_ranking_loss,nn.functional.soft_margin_loss,nn.functional.multilabel_soft_margin_loss,nn.functional.cosine_embedding_loss,nn.functional.hinge_embedding_loss
for g in $G1 $G2 $G3; do
  flock /root/tpu.lock timeout 280 python gradworker.py $g grad.jsonl >> grad.log 2>&1
  echo "grad rc=$? $(date -u +%T)" >> driver.log
  sleep 5
done
