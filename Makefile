SHELL := /bin/bash
VENV_NAME := adp
CONDA_ACTIVATE := source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $(VENV_NAME)
PYTHON := $(CONDA_ACTIVATE) && python
PIP := $(CONDA_ACTIVATE) && pip3
# Stacked single-node multi-worker: https://pytorch.org/docs/stable/elastic/run.html#stacked-single-node-multi-worker 
TORCHRUN = $(CONDA_ACTIVATE) && torchrun --rdzv-backend=c10d --rdzv_endpoint localhost:0 --nnodes=1 --nproc_per_node=$(NGPU)

# Taken from https://tech.davis-hansson.com/p/make/
ifeq ($(origin .RECIPEPREFIX), undefined)
  $(error This Make does not support .RECIPEPREFIX. Please use GNU Make 4.0 or later)
endif
.RECIPEPREFIX = >

#
# Virtual Environment Targets
#
clean:
> rm -f .venv_done

.venv_done: clean
> conda create -n $(VENV_NAME) python=3.9
> $(PIP) install -r requirements.txt
> $(PIP) install -e .
> touch $@

#
# Python Targets
#
include CONFIG.mk
SRC := src
BEST_MODEL = "../$(word 1,$(subst -, ,$*))/model/best.pt"
RESULT_JSON := html.json
TEDS_STRUCTURE = -f "../experiments/$*/$(RESULT_JSON)" -s

######################
NGPU := 1

.SECONDARY:

# training
experiments/%/.done_train:
> @echo "Using experiment configurations from variable EXP_$*"
> cd $(SRC) && $(TORCHRUN) -m main ++name=$* $(EXP_$*) ++trainer.mode="train"
> touch $@

experiments/%/.done_finetune:
> @echo "Finetuning phase 1 - using experiment configurations from variable EXP_$*"
> cd $(SRC) && $(TORCHRUN) -m main ++name=$* $(EXP_$*) ++trainer.mode="train"
> @echo "Finetuning phase 2 - starting from epoch 4"
> cd $(SRC) && $(TORCHRUN) -m main ++name=$* $(EXP_$*) ++trainer.mode="train" ++trainer.trainer.snapshot="epoch3_snapshot.pt" ++trainer.trainer.beit_pretrained_weights=null
> touch $@

# =============
# experiments/mini_pubtabnet/.done_%:
# > @echo "Testing model $* on mini_pubtabnet"
# > @rm -f $(@D)/$*/*
# > @mkdir -p $(@D)
# > cd $(SRC) && $(TORCHRUN) -m main ++name=$* $(EXP_$*) ++trainer.mode="test" \
#   ++hydra.run.dir="../$(@D)" $(MINIPUBTABNET) ++trainer.trainer.model_weights=$(BEST_MODEL)
# > cd $(SRC) && $(PYTHON) -m utils.engine -f ../$(@D)/$* -t $(word 2,$(subst -, ,$*))
# > touch $@

# experiments/mini_pubtabnet/.teds_%:
# > @echo "Testing model $* on mini_pubtabnet for teds"
# > cd $(SRC) && $(PYTHON) -m utils.teds \
#   -f ../$(@D)/$*/final.json -t $(word 2,$(subst -, ,$*))
# # # > touch $@

# experiments/mini_pubtabnet/.map_%:
# > @echo "Testing model $* on mini_pubtabnet for teds"
# > cd $(SRC) && $(PYTHON) -m utils.coco_map -f ../$(@D)/$*/final.json
# # # > touch $@

experiments/syn_fintabnet/.done_%:
> @echo "Testing model $* on synthtabnet fintabnet"
> @rm -f $(@D)/$*/*
> @mkdir -p $(@D)
> cd $(SRC) && $(TORCHRUN) -m main ++name=$* $(EXP_$*) ++trainer.mode="test" \
  ++hydra.run.dir="../$(@D)" $(SYN_fin) ++trainer.trainer.model_weights=$(BEST_MODEL)
> cd $(SRC) && $(PYTHON) -m utils.engine -f ../$(@D)/$* -t $(word 2,$(subst -, ,$*))
> touch $@

experiments/syn_fintabnet/.teds_%: experiments/syn_fintabnet/.done_%
> @echo "Testing model $* on syn_fintabnet for teds"
> cd $(SRC) && $(PYTHON) -m utils.teds \
  -f ../$(@D)/$*/final.json -t $(word 2,$(subst -, ,$*))
> touch $@

experiments/syn_fintabnet/.map_%: experiments/syn_fintabnet/.done_%
> @echo "Testing model $* on syn_fintabnet for mAP"
> cd $(SRC) && $(PYTHON) -m utils.coco_map -f ../$(@D)/$*/final.json
> touch $@

experiments/syn_marketing/.done_%:
> @echo "Testing model $* on synthtabnet marketing"
> @rm -f $(@D)/$*/*
> @mkdir -p $(@D)
> cd $(SRC) && $(TORCHRUN) -m main ++name=$* $(EXP_$*) ++trainer.mode="test" \
  ++hydra.run.dir="../$(@D)" $(SYN_market) ++trainer.trainer.model_weights=$(BEST_MODEL)
> cd $(SRC) && $(PYTHON) -m utils.engine -f ../$(@D)/$* -t $(word 2,$(subst -, ,$*))
> touch $@

experiments/syn_marketing/.teds_%: experiments/syn_marketing/.done_%
> @echo "Testing model $* on syn_marketing for teds"
> cd $(SRC) && $(PYTHON) -m utils.teds \
  -f ../$(@D)/$*/final.json -t $(word 2,$(subst -, ,$*))
> touch $@

experiments/syn_marketing/.map_%: experiments/syn_marketing/.done_%
> @echo "Testing model $* on syn_marketing for mAP"
> cd $(SRC) && $(PYTHON) -m utils.coco_map -f ../$(@D)/$*/final.json
> touch $@

experiments/syn_pubtabnet/.done_%:
> @echo "Testing model $* on synthtabnet pubtabnet"
> @rm -f $(@D)/$*/*
> @mkdir -p $(@D)
> cd $(SRC) && $(TORCHRUN) -m main ++name=$* $(EXP_$*) ++trainer.mode="test" \
  ++hydra.run.dir="../$(@D)" $(SYN_pub) ++trainer.trainer.model_weights=$(BEST_MODEL)
> cd $(SRC) && $(PYTHON) -m utils.engine -f ../$(@D)/$* -t $(word 2,$(subst -, ,$*))
> touch $@

experiments/syn_pubtabnet/.teds_%: experiments/syn_pubtabnet/.done_%
> @echo "Testing model $* on syn_pubtabnet for teds"
> cd $(SRC) && $(PYTHON) -m utils.teds \
  -f ../$(@D)/$*/final.json -t $(word 2,$(subst -, ,$*))
> touch $@

experiments/syn_pubtabnet/.map_%: experiments/syn_pubtabnet/.done_%
> @echo "Testing model $* on syn_pubtabnet for mAP"
> cd $(SRC) && $(PYTHON) -m utils.coco_map -f ../$(@D)/$*/final.json
> touch $@

experiments/syn_sparse/.done_%:
> @echo "Testing model $* on synthtabnet sparse"
> @rm -f $(@D)/$*/*
> @mkdir -p $(@D)
> cd $(SRC) && $(TORCHRUN) -m main ++name=$* $(EXP_$*) ++trainer.mode="test" \
  ++hydra.run.dir="../$(@D)" $(SYN_sparse) ++trainer.trainer.model_weights=$(BEST_MODEL)
> cd $(SRC) && $(PYTHON) -m utils.engine -f ../$(@D)/$* -t $(word 2,$(subst -, ,$*))
> touch $@

experiments/syn_sparse/.teds_%: experiments/syn_sparse/.done_%
> @echo "Testing model $* on syn_sparse for teds"
> cd $(SRC) && $(PYTHON) -m utils.teds \
  -f ../$(@D)/$*/final.json -t $(word 2,$(subst -, ,$*))
> touch $@

experiments/syn_sparse/.map_%: experiments/syn_sparse/.done_%
> @echo "Testing model $* on syn_sparse for mAP"
> cd $(SRC) && $(PYTHON) -m utils.coco_map -f ../$(@D)/$*/final.json
> touch $@

experiments/pubtabnet/.done_%:
> @echo "Testing model $* on pubtabnet"
> @rm -f $(@D)/$*/*
> @mkdir -p $(@D)
> cd $(SRC) && $(TORCHRUN) -m main ++name=$* $(EXP_$*) ++trainer.mode="test" \
  ++hydra.run.dir="../$(@D)" $(PUBTABNET) ++trainer.trainer.model_weights=$(BEST_MODEL)
> cd $(SRC) && $(PYTHON) -m utils.engine -f ../$(@D)/$* -t $(word 2,$(subst -, ,$*))
> touch $@

experiments/pubtabnet/.teds_%: experiments/pubtabnet/.done_%
> @echo "Testing model $* on pubtabnet for teds"
> cd $(SRC) && $(PYTHON) -m utils.teds \
  -f ../$(@D)/$*/final.json -t $(word 2,$(subst -, ,$*))
> touch $@

experiments/pubtabnet/.map_%: experiments/pubtabnet/.done_%
> @echo "Testing model $* on pubtabnet for mAP"
> cd $(SRC) && $(PYTHON) -m utils.coco_map -f ../$(@D)/$*/final.json
> touch $@

experiments/pubtables1m/.done_%:
> @echo "Testing model $* on pubtables1m"
> @rm -f $(@D)/$*/*
> @mkdir -p $(@D)
> cd $(SRC) && $(TORCHRUN) -m main ++name=$* $(EXP_$*) ++trainer.mode="test" \
  ++hydra.run.dir="../$(@D)" $(PUBTABLES1M) ++trainer.trainer.model_weights=$(BEST_MODEL)
> cd $(SRC) && $(PYTHON) -m utils.engine -f ../$(@D)/$* -t $(word 2,$(subst -, ,$*))
> touch $@

experiments/pubtables1m/.map_%: experiments/pubtables1m/.done_%
> @echo "Testing model $* on pubtables1m for mAP"
> cd $(SRC) && $(PYTHON) -m utils.coco_map -f ../$(@D)/$*/final.json
> touch $@

experiments/fintabnet/.done_%:
> @echo "Testing model $* on fintabnet"
> @rm -f $(@D)/$*/*
> @mkdir -p $(@D)
> cd $(SRC) && $(TORCHRUN) -m main ++name=$* $(EXP_$*) ++trainer.mode="test" \
  ++hydra.run.dir="../$(@D)" $(FINTABNET) ++trainer.trainer.model_weights=$(BEST_MODEL)
> cd $(SRC) && $(PYTHON) -m utils.engine -f ../$(@D)/$* -t $(word 2,$(subst -, ,$*))
> touch $@

experiments/fintabnet/.teds_%: experiments/fintabnet/.done_%
> @echo "Testing model $* on fintabnet for teds"
> cd $(SRC) && $(PYTHON) -m utils.teds \
  -f ../$(@D)/$*/final.json -t $(word 2,$(subst -, ,$*))
> touch $@
