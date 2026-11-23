# torch_tpu PR deliverables

Prepared 2026-11-23.
All branches are on the fork `shijiew555/torch_tpu` and based on upstream `main` at `82d0a494`.
The issue and PR text for each PR is in `pr_desc/` on the fork's `agent-coord` branch.
Each draft file has the issue title and body first (if an issue is needed), then the PR title and body.
No issue or PR is opened yet. Fill in the GitHub numbers after you open them.

How to submit each one:

1. If the PR needs an issue, open the issue first: https://github.com/google-pytorch/torch_tpu/issues/new
2. Open the PR with the "Open PR" link. Paste the PR title and body from the draft.
3. Replace `Fixes #<issue>` in the PR body with the issue number.

## PR 1: Make torch.autocast("tpu") default to bfloat16

* Issue first: Yes
* Issue and PR text: https://github.com/shijiew555/torch_tpu/blob/agent-coord/pr_desc/autocast-bf16-default.md
* Branch: https://github.com/shijiew555/torch_tpu/tree/autocast-bf16-default-pr2 (commit `3ef9ff18`)
* Open PR: https://github.com/google-pytorch/torch_tpu/compare/main...shijiew555:torch_tpu:autocast-bf16-default-pr2?expand=1
* Issue number:
* PR number:

## PR 2: Keep aliasing between outputs of compiled TPU graphs

* Issue first: Yes
* Issue and PR text: https://github.com/shijiew555/torch_tpu/blob/agent-coord/pr_desc/compiled-output-alias.md
* Branch: https://github.com/shijiew555/torch_tpu/tree/compiled-output-alias-pr2 (commit `f7ee0c2f`)
* Open PR: https://github.com/google-pytorch/torch_tpu/compare/main...shijiew555:torch_tpu:compiled-output-alias-pr2?expand=1
* Issue number:
* PR number:

## PR 3: Restore the TPU RNG state in eager activation checkpointing

* Issue first: Yes
* Issue and PR text: https://github.com/shijiew555/torch_tpu/blob/agent-coord/pr_desc/ckpt-tpu-rng-state.md
* Branch: https://github.com/shijiew555/torch_tpu/tree/ckpt-tpu-rng-state-pr (commit `aae1baaf`)
* Open PR: https://github.com/google-pytorch/torch_tpu/compare/main...shijiew555:torch_tpu:ckpt-tpu-rng-state-pr?expand=1
* Issue number:
* PR number:

## PR 4: Use HIGHEST precision for float32 unfold backward and col2im

* Issue first: Yes
* Issue and PR text: https://github.com/shijiew555/torch_tpu/blob/agent-coord/pr_desc/unfold-col2im-precision.md
* Branch: https://github.com/shijiew555/torch_tpu/tree/unfold-col2im-precision-pr2 (commit `be7d0e6d`)
* Open PR: https://github.com/google-pytorch/torch_tpu/compare/main...shijiew555:torch_tpu:unfold-col2im-precision-pr2?expand=1
* Issue number:
* PR number:

## PR 5: Accept sorted=False in torch.unique

* Issue first: No
* Issue and PR text: https://github.com/shijiew555/torch_tpu/blob/agent-coord/pr_desc/unique-sorted-false.md
* Branch: https://github.com/shijiew555/torch_tpu/tree/unique-sorted-false-pr2 (commit `947d7f21`)
* Open PR: https://github.com/google-pytorch/torch_tpu/compare/main...shijiew555:torch_tpu:unique-sorted-false-pr2?expand=1
* Issue number:
* PR number:

## Notes

* PR 4 uses HIGHEST precision for float32 (v1). It makes float32 fold and unfold 1.7x to 3.4x slower. If review objects, the fallback (v2) is to follow `torch.set_float32_matmul_precision`, which keeps the default speed.
* PR 5 conflicts with open upstream PR #6879 in the same function. The PR text explains the simple resolution.
* The older branches `autocast-bf16-default-pr` and `compiled-output-alias-pr` are superseded by the `-pr2` branches.
