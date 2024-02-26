#
# Datasets
#

# label type
LABEL_IMAGE = ++trainer.label_type="image"
LABEL_HTML = ++trainer.label_type="html" "++trainer.train.loss_weights.html=1"
LABEL_CELL = ++trainer.label_type="cell" "++trainer.train.loss_weights.cell=1"
LABEL_BBOX = ++trainer.label_type="bbox" "++trainer.train.loss_weights.bbox=1"
LABEL_HTML_CELL_BBOX = ++trainer.label_type="html+cell+bbox"
MEAN = [0.86597056,0.88463002,0.87491087]
STD = [0.20686628,0.18201602,0.18485524]

# augmentation
AUG_VQVAE = dataset/augmentation=vqvae
AUG_BEIT = dataset/augmentation=beit \
	++dataset.augmentation.mean=$(MEAN) ++dataset.augmentation.std=$(STD)
AUG_RESIZE_NORM = dataset/augmentation=resize_normalize \
	++dataset.augmentation.transforms.2.mean=$(MEAN) ++dataset.augmentation.transforms.2.std=$(STD)

# single dataset
DATA_SINGLE = dataset=single_dataset
PUBTABNET = $(DATA_SINGLE) \
	+dataset/pubtabnet@dataset.train_dataset=train_dataset \
	+dataset/pubtabnet@dataset.valid_dataset=valid_dataset \
	+dataset/pubtabnet@dataset.test_dataset=test_dataset
MINIPUBTABNET = $(DATA_SINGLE) \
	+dataset/mini_pubtabnet@dataset.train_dataset=train_dataset \
	+dataset/mini_pubtabnet@dataset.valid_dataset=valid_dataset \
	+dataset/mini_pubtabnet@dataset.test_dataset=test_dataset

# multiple datasets
DATA_MULTI = dataset=concat_dataset
PUBTABNET_M = +dataset/pubtabnet@dataset.train.d1=train_dataset \
	+dataset/pubtabnet@dataset.valid.d1=valid_dataset \
	+dataset/pubtabnet@dataset.test.d1=test_dataset
SYN_MARKET_M = +dataset/synthtabnet_marketing@dataset.train.d2=train_dataset \
	+dataset/synthtabnet_marketing@dataset.valid.d2=valid_dataset \
	+dataset/synthtabnet_marketing@dataset.test.d2=test_dataset
SYN_FIN_M = +dataset/synthtabnet_fintabnet@dataset.train.d3=train_dataset \
	+dataset/synthtabnet_fintabnet@dataset.valid.d3=valid_dataset \
	+dataset/synthtabnet_fintabnet@dataset.test.d3=test_dataset
SYN_SPARSE_M = +dataset/synthtabnet_sparse@dataset.train.d4=train_dataset \
	+dataset/synthtabnet_sparse@dataset.valid.d4=valid_dataset \
	+dataset/synthtabnet_sparse@dataset.test.d4=test_dataset
SYN_PUB_M = +dataset/synthtabnet_pubtabnet@dataset.train.d5=train_dataset \
	+dataset/synthtabnet_pubtabnet@dataset.valid.d5=valid_dataset \
	+dataset/synthtabnet_pubtabnet@dataset.test.d5=test_dataset
ICDAR_M = +dataset/icdar@dataset.train.d6=train_dataset \
	+dataset/icdar@dataset.valid.d6=valid_dataset \
	+dataset/icdar@dataset.test.d6=test_dataset
PUBTABLES_M = +dataset/pubtables1m@dataset.train.d7=train_dataset \
	+dataset/pubtables1m@dataset.valid.d7=valid_dataset \
	+dataset/pubtables1m@dataset.test.d7=test_dataset
TABLEBANK_M = +dataset/tablebank@dataset.train.d8=train_dataset \
	+dataset/tablebank@dataset.valid.d8=valid_dataset \
	+dataset/tablebank@dataset.test.d8=test_dataset
FINTABNET_M = +dataset/fintabnet@dataset.train.d9=train_dataset \
	+dataset/fintabnet@dataset.valid.d9=valid_dataset \
	+dataset/fintabnet@dataset.test.d9=test_dataset

DATA_VQVAE_1M = $(DATA_MULTI) \
	$(PUBTABNET_M) $(SYN_MARKET_M) $(SYN_FIN_M) $(SYN_SPARSE_M)
DATA_VQVAE_2M = $(DATA_MULTI) \
	$(PUBTABNET_M) $(SYN_MARKET_M) $(SYN_FIN_M) $(SYN_SPARSE_M) $(SYN_PUB_M) \
	$(PUBTABLES_M) $(TABLEBANK_M)

ICDAR = $(DATA_MULTI) $(ICDAR_M)
PUBTABLES1M = $(DATA_MULTI) $(PUBTABLES_M)
FINTABNET = $(DATA_MULTI) $(FINTABNET_M)

PUB_SYN = $(DATA_MULTI) \
	$(PUBTABNET_M) $(SYN_MARKET_M) $(SYN_FIN_M) $(SYN_SPARSE_M) $(SYN_PUB_M)

PUB_SYN_FIN = $(DATA_MULTI) $(PUBTABNET_M) $(FINTABNET_M) \
	$(SYN_MARKET_M) $(SYN_FIN_M) $(SYN_SPARSE_M) $(SYN_PUB_M)

PUB_SYN_PUB1M = $(DATA_MULTI) $(PUBTABNET_M) $(PUBTABLES_M) \
	$(SYN_MARKET_M) $(SYN_FIN_M) $(SYN_SPARSE_M) $(SYN_PUB_M)

SYN = $(DATA_MULTI) $(SYN_MARKET_M) $(SYN_FIN_M) $(SYN_SPARSE_M) $(SYN_PUB_M)

SYN_fin = $(DATA_MULTI) $(SYN_FIN_M)
SYN_market = $(DATA_MULTI) $(SYN_MARKET_M)
SYN_pub = $(DATA_MULTI) $(SYN_PUB_M)
SYN_sparse = $(DATA_MULTI) $(SYN_SPARSE_M)

#
# Vocab
#
VOCAB_NONE = vocab=empty
VOCAB_HTML = vocab=html
VOCAB_BBOX = vocab=bbox
VOCAB_CELL = vocab=cell


#
# Trainer
#

# trainer type
TRAINER_VQVAE = trainer=vqvae
TRAINER_BEIT = trainer=beit
TRAINER_TABLE = trainer=table

# input image size
I224 = ++trainer.img_size=[224,224]
I448 = ++trainer.img_size=[448,448]
I112_336 = ++trainer.img_size=[112,336]
I112_224 = ++trainer.img_size=[112,224]
I112_448 = ++trainer.img_size=[112,448]

# max sequence length
SEQ200 = trainer.max_seq_len=200
SEQ512 = trainer.max_seq_len=512
SEQ896 = trainer.max_seq_len=896
SEQ1024 = trainer.max_seq_len=1024
SEQ1280 = trainer.max_seq_len=1280

# batch size + epoch
BATCH12 = ++trainer.train.dataloader.batch_size=12 ++trainer.valid.dataloader.batch_size=12
BATCH24 = ++trainer.train.dataloader.batch_size=24 ++trainer.valid.dataloader.batch_size=24
BATCH30 = ++trainer.train.dataloader.batch_size=30 ++trainer.valid.dataloader.batch_size=30
BATCH36 = ++trainer.train.dataloader.batch_size=36 ++trainer.valid.dataloader.batch_size=36
BATCH40 = ++trainer.train.dataloader.batch_size=40 ++trainer.valid.dataloader.batch_size=40
BATCH48 = ++trainer.train.dataloader.batch_size=48 ++trainer.valid.dataloader.batch_size=48
BATCH64 = ++trainer.train.dataloader.batch_size=64 ++trainer.valid.dataloader.batch_size=64
BATCH72 = ++trainer.train.dataloader.batch_size=72 ++trainer.valid.dataloader.batch_size=72
BATCH80 = ++trainer.train.dataloader.batch_size=80 ++trainer.valid.dataloader.batch_size=80
BATCH90 = ++trainer.train.dataloader.batch_size=90 ++trainer.valid.dataloader.batch_size=90
BATCH96 = ++trainer.train.dataloader.batch_size=96 ++trainer.valid.dataloader.batch_size=96
BATCH108 = ++trainer.train.dataloader.batch_size=108 ++trainer.valid.dataloader.batch_size=108
BATCH256 = ++trainer.train.dataloader.batch_size=256 ++trainer.valid.dataloader.batch_size=256
BATCH384 = ++trainer.train.dataloader.batch_size=384 ++trainer.valid.dataloader.batch_size=384

EPOCH24 = ++trainer.train.epochs=24
EPOCH30 = ++trainer.train.epochs=30
EPOCH48 = ++trainer.train.epochs=48
EPOCH300 = ++trainer.train.epochs=300

# optimizer
OPT_ADAMW = trainer/train/optimizer=adamw
OPT_WD5e2 = ++trainer.train.optimizer.weight_decay=5e-2

# lr + scheduler
LR_1e3 = ++trainer.train.optimizer.lr=1e-3
LR_5e4 = ++trainer.train.optimizer.lr=5e-4
LR_4e4 = ++trainer.train.optimizer.lr=4e-4
LR_3e4 = ++trainer.train.optimizer.lr=3e-4
LR_1e4 = ++trainer.train.optimizer.lr=1e-4
LR_8e5 = ++trainer.train.optimizer.lr=8e-5
LR_1e5 = ++trainer.train.optimizer.lr=1e-5

LR_cosine = trainer/train/lr_scheduler=cosine ++trainer.train.lr_scheduler.lr_lambda.min_ratio=5e-3
LR_cosine93k_warm6k = $(LR_cosine) ++trainer.train.lr_scheduler.lr_lambda.total_step=93400 ++trainer.train.lr_scheduler.lr_lambda.warmup=5800
LR_cosine89k_warm6k = $(LR_cosine) ++trainer.train.lr_scheduler.lr_lambda.total_step=89300 ++trainer.train.lr_scheduler.lr_lambda.warmup=5500
LR_cosine63k_warm8k = $(LR_cosine) ++trainer.train.lr_scheduler.lr_lambda.total_step=62600 ++trainer.train.lr_scheduler.lr_lambda.warmup=7800
LR_cosine77k_warm8k = $(LR_cosine) ++trainer.train.lr_scheduler.lr_lambda.total_step=76600 ++trainer.train.lr_scheduler.lr_lambda.warmup=7660
LR_cosine50k_warm10k = $(LR_cosine) ++trainer.train.lr_scheduler.lr_lambda.total_step=50000 ++trainer.train.lr_scheduler.lr_lambda.warmup=10000
LR_cosine40k_warm8k = $(LR_cosine) ++trainer.train.lr_scheduler.lr_lambda.total_step=40000 ++trainer.train.lr_scheduler.lr_lambda.warmup=8000
LR_cosine40k_warm5k = $(LR_cosine) ++trainer.train.lr_scheduler.lr_lambda.total_step=40000 ++trainer.train.lr_scheduler.lr_lambda.warmup=5000
LR_cosine31k_warm4k = $(LR_cosine) ++trainer.train.lr_scheduler.lr_lambda.total_step=31300 ++trainer.train.lr_scheduler.lr_lambda.warmup=3900
LR_cosine30k_warm4k = $(LR_cosine) ++trainer.train.lr_scheduler.lr_lambda.total_step=30500 ++trainer.train.lr_scheduler.lr_lambda.warmup=4000
LR_cosine30k_warm3k = $(LR_cosine) ++trainer.train.lr_scheduler.lr_lambda.total_step=30000 ++trainer.train.lr_scheduler.lr_lambda.warmup=3750
LR_cosine23k_warm3k = $(LR_cosine) ++trainer.train.lr_scheduler.lr_lambda.total_step=22500 ++trainer.train.lr_scheduler.lr_lambda.warmup=2800
LR_cosine21k_warm3k = $(LR_cosine) ++trainer.train.lr_scheduler.lr_lambda.total_step=20800 ++trainer.train.lr_scheduler.lr_lambda.warmup=2600
LR_cosine20k_warm3k = $(LR_cosine) ++trainer.train.lr_scheduler.lr_lambda.total_step=20000 ++trainer.train.lr_scheduler.lr_lambda.warmup=2500
LR_cosine16k_warm2k = $(LR_cosine) ++trainer.train.lr_scheduler.lr_lambda.total_step=16000 ++trainer.train.lr_scheduler.lr_lambda.warmup=2000
LR_cosine8k_warm1k = $(LR_cosine) ++trainer.train.lr_scheduler.lr_lambda.total_step=7600 ++trainer.train.lr_scheduler.lr_lambda.warmup=800
LR_cosine47k_warm6k = $(LR_cosine) ++trainer.train.lr_scheduler.lr_lambda.total_step=47400 ++trainer.train.lr_scheduler.lr_lambda.warmup=5920
LR_cosine44k_warm6k = $(LR_cosine) ++trainer.train.lr_scheduler.lr_lambda.total_step=44100 ++trainer.train.lr_scheduler.lr_lambda.warmup=5500
LR_cosine57k_warm7k = $(LR_cosine) ++trainer.train.lr_scheduler.lr_lambda.total_step=56900 ++trainer.train.lr_scheduler.lr_lambda.warmup=7100
LR_cosine118k_warm15k = $(LR_cosine) ++trainer.train.lr_scheduler.lr_lambda.total_step=117800 ++trainer.train.lr_scheduler.lr_lambda.warmup=14700
LR_cosine216k_warm27k = $(LR_cosine) ++trainer.train.lr_scheduler.lr_lambda.total_step=216000 ++trainer.train.lr_scheduler.lr_lambda.warmup=27000
LR_cosine3k_warm04k = $(LR_cosine) ++trainer.train.lr_scheduler.lr_lambda.total_step=3050 ++trainer.train.lr_scheduler.lr_lambda.warmup=380
LR_cosine32k = $(LR_cosine) ++trainer.train.lr_scheduler.lr_lambda.total_step=32000 ++trainer.train.lr_scheduler.lr_lambda.warmup=0
LR_cosine118k = $(LR_cosine) ++trainer.train.lr_scheduler.lr_lambda.total_step=118000 ++trainer.train.lr_scheduler.lr_lambda.warmup=0

LR_step = trainer/train/lr_scheduler=step
LR_step5k = $(LR_step) ++trainer.train.lr_scheduler.step_size=5000

GRAD_CLIP12 = ++trainer.train.grad_clip=12

# bbox sampling
NBBOX_TRAIN_10 = ++trainer.bbox_sampling=10
NBBOX_TEST_10 = ++trainer.test.dataloader.bbox_sampling=10

# vqvae
VQVAE_TEMP_1M = ++trainer.train.starting_temp=1. \
	++trainer.train.temp_min=5e-3 ++trainer.train.temp_anneal_rate=1e-3
VQVAE_TEMP_2M = ++trainer.train.starting_temp=1. \
	++trainer.train.temp_min=1e-3 ++trainer.train.temp_anneal_rate=2e-4

# beit specific
TRANS448_VQVAE224_GRID28_MASK300 = ++trainer.trans_size=448 ++trainer.vqvae_size=224 ++trainer.grid_size=28 ++trainer.num_mask_patches=300
VQVAE1M_WEIGHTS = $(MODEL_VQVAE) ++trainer.vqvae_weights="../vqvae_1M/model/best.pt"
VQVAE2M_WEIGHTS = $(MODEL_VQVAE_L) ++trainer.vqvae_weights="../vqvae_2M/model/best.pt"
SNAPSHOT_BEIT = ++trainer.trainer.snapshot="/raid/speng65/MISC/adp_tables/experiments/beit/snapshot/epoch6_snapshot.pt"

# finetuning specific
WEIGHTS_beit_none = ++trainer.trainer.beit_pretrained_weights=null
WEIGHTS_beit_1m_small = ++trainer.trainer.beit_pretrained_weights="../beit_1M_small/model/best.pt"
WEIGHTS_beit_1m_medium = ++trainer.trainer.beit_pretrained_weights="../beit_1M_medium/model/best.pt"
WEIGHTS_beit_2m_small = ++trainer.trainer.beit_pretrained_weights="../beit_2M_small/model/best.pt"
WEIGHTS_beit_2m_medium = ++trainer.trainer.beit_pretrained_weights="../beit_2M_medium/model/best.pt"
LOCK_BEIT4 = ++trainer.trainer.freeze_beit_epoch=4

#
# Models
#

# model type
MODEL_VQVAE = model=vqvae
MODEL_VQVAE_L = $(MODEL_VQVAE) ++model.codebook_tokens=16384 ++model.hidden_dim=512
MODEL_BEIT = model=beit
MODEL_ENCODER_DECODER = model=encoderdecoder

# backbone for input preprocessing: resnet, linear projection, and convstem
IMGCNN = model/model/backbone=imgcnn
IMGLINEAR = model/model/backbone=imglinear
IMGCONVSTEM = model/model/backbone=imgconvstem

# number of layers
E4 = ++model.model.encoder.nlayer=4
E12 = ++model.model.encoder.nlayer=12
E24 = ++model.model.encoder.nlayer=24
D4 = ++model.model.decoder.nlayer=4

# transformer layer: attention heads, hidden size, activation, norm
FF4 = ++model.ff_ratio=4

NHEAD8 = ++model.nhead=8
NHEAD12 = ++model.nhead=12
NHEAD16 = ++model.nhead=16

NORM_FIRST = ++model.norm_first=true
NORM_LAST = ++model.norm_first=false

ACT_RELU = ++model.activation="relu"
ACT_GELU = ++model.activation="gelu"

D_MODEL512 = ++model.d_model=512
D_MODEL768 = ++model.d_model=768
D_MODEL1024 = ++model.d_model=1024

# regularization
REG_d00 = ++model.dropout=0.0
REG_d02 = ++model.dropout=0.2

# linear projection patch size
P16 = ++model.backbone_downsampling_factor=16
P28 = ++model.backbone_downsampling_factor=28
P32 = ++model.backbone_downsampling_factor=32

# cnn backbone
R18 = ++model.model.backbone.backbone._target_=torchvision.models.resnet18 \
	++model.model.backbone.output_channels=512

BEIT_SMALL = $(MODEL_BEIT) $(IMGLINEAR) $(NHEAD8) $(FF4) $(ACT_GELU) \
	$(NORM_FIRST) $(D_MODEL512) $(REG_d02) $(P16) $(E4)
BEIT_MEDIUM = $(MODEL_BEIT) $(IMGLINEAR) $(NHEAD12) $(FF4) $(ACT_GELU) \
	$(NORM_FIRST) $(D_MODEL768) $(REG_d02) $(P16) $(E12)

ARCH_SMALL = $(BEIT_SMALL) $(MODEL_ENCODER_DECODER) $(D4)
ARCH_MEDIUM = $(BEIT_MEDIUM) $(MODEL_ENCODER_DECODER) $(D4)

ARCH_LARGE = $(MODEL_ENCODER_DECODER) $(IMGLINEAR) $(NHEAD16) $(FF4) \
	$(ACT_GELU) $(NORM_FIRST) $(D_MODEL1024) $(REG_d02) $(P16) $(E24) $(D4)

ARCH_R18_SMALL = $(MODEL_ENCODER_DECODER) $(IMGCNN) $(R18) $(NHEAD8) $(FF4) \
	$(ACT_GELU) $(NORM_FIRST) $(D_MODEL512) $(REG_d02) $(P16) $(E4) $(D4)
ARCH_R18_MEDIUM = $(MODEL_ENCODER_DECODER) $(IMGCNN) $(R18) $(NHEAD12) $(FF4) \
	$(ACT_GELU) $(NORM_FIRST) $(D_MODEL768) $(REG_d02) $(P16) $(E12) $(D4)
ARCH_R18_LARGE = $(MODEL_ENCODER_DECODER) $(IMGCNN) $(R18) $(NHEAD16) $(FF4) \
	$(ACT_GELU) $(NORM_FIRST) $(D_MODEL1024) $(REG_d02) $(P16) $(E24) $(D4)

#
# Test
#
HTML_RESULT_FILE = trainer.test.save_to_prefix=html
CELL_RESULT_FILE = trainer.test.save_to_prefix=cell
BBOX_RESULT_FILE = trainer.test.save_to_prefix=bbox


###############
# Experiments #
###############

# cross dataset training
# html
TRAIN_syn_pub_html := $(VOCAB_HTML) \
	$(PUB_SYN) $(LABEL_HTML) $(AUG_RESIZE_NORM) \
	$(TRAINER_TABLE) $(I448) $(SEQ512) \
	$(EPOCH48) $(OPT_ADAMW) $(OPT_WD5e2) $(LR_8e5)

EXP_ssp_2m_syn_pub_html_medium := $(TRAIN_syn_pub_html) $(ARCH_MEDIUM) \
	$(WEIGHTS_beit_2m_medium) $(LOCK_BEIT4) $(BATCH72) $(LR_cosine93k_warm6k)

# cell
TRAIN_syn_pub_pub1m_cell := $(VOCAB_CELL) \
	$(PUB_SYN_PUB1M) $(LABEL_CELL) $(AUG_RESIZE_NORM) \
	$(TRAINER_TABLE) $(I112_448) $(SEQ200) \
	$(EPOCH24) $(OPT_ADAMW) $(OPT_WD5e2) $(LR_8e5) $(GRAD_CLIP12)

EXP_syn_pub_pub1m_cell_medium := $(TRAIN_syn_pub_pub1m_cell) $(ARCH_MEDIUM) \
	$(BATCH24) $(LR_cosine216k_warm27k)

# bbox
TRAIN_syn_pub_bbox := $(VOCAB_BBOX) \
	$(PUB_SYN) $(LABEL_BBOX) $(AUG_RESIZE_NORM) \
	$(TRAINER_TABLE) $(I448) $(SEQ1024) \
	$(EPOCH30) $(OPT_ADAMW) $(OPT_WD5e2) $(LR_3e4) $(GRAD_CLIP12)

EXP_ssp_2m_syn_pub_bbox_medium := $(TRAIN_syn_pub_bbox) $(ARCH_MEDIUM) \
	$(WEIGHTS_beit_2m_medium) $(LOCK_BEIT4) $(BATCH48) $(LR_cosine77k_warm8k)

# -------
# train and test synthtabnet
TRAIN_syn_html := $(VOCAB_HTML) \
	$(SYN) $(LABEL_HTML) $(AUG_RESIZE_NORM) \
	$(TRAINER_TABLE) $(I448) $(SEQ512) \
	$(EPOCH24) $(OPT_ADAMW) $(OPT_WD5e2) $(LR_1e4)

# TRAIN_syn_cell := $(VOCAB_WORDPIECE) \
# 	$(SYN) $(LABEL_CELL) $(AUG_RESIZE_NORM) \
# 	$(TRAINER_TABLE) $(I448) $(SEQ896) \
# 	$(EPOCH24) $(OPT_ADAMW) $(OPT_WD5e2) $(LR_1e4)

TRAIN_syn_bbox := $(VOCAB_BBOX) \
	$(SYN) $(LABEL_BBOX) $(AUG_RESIZE_NORM) \
	$(TRAINER_TABLE) $(I448) $(SEQ1024) \
	$(EPOCH24) $(OPT_ADAMW) $(OPT_WD5e2) $(LR_3e4) $(GRAD_CLIP12)
# $(NBBOX_TRAIN_10)

EXP_syn_html_r18_small := $(TRAIN_syn_html) $(ARCH_R18_SMALL) $(BATCH90) $(LR_cosine16k_warm2k)
# EXP_syn_cell_r18_small := $(TRAIN_syn_cell) $(ARCH_R18_SMALL) $(BATCH48) $(LR_cosine30k_warm4k)
EXP_syn_bbox_r18_small := $(TRAIN_syn_bbox) $(ARCH_R18_SMALL) $(BATCH48) $(LR_cosine30k_warm3k)

EXP_syn_html_r18_medium := $(TRAIN_syn_html) $(ARCH_R18_MEDIUM) $(BATCH64) $(LR_cosine23k_warm3k)
EXP_syn_bbox_r18_medium := $(TRAIN_syn_bbox) $(ARCH_R18_MEDIUM) $(BATCH48) $(LR_cosine30k_warm3k)

EXP_syn_html_small := $(TRAIN_syn_html) $(ARCH_SMALL) $(BATCH90) $(LR_cosine16k_warm2k)
# EXP_syn_cell_small := $(TRAIN_syn_cell) $(ARCH_SMALL) $(BATCH36) $(LR_cosine40k_warm5k)
EXP_syn_bbox_small := $(TRAIN_syn_bbox) $(ARCH_SMALL) $(BATCH48) $(LR_cosine30k_warm3k)

EXP_syn_html_medium := $(TRAIN_syn_html) $(ARCH_MEDIUM) $(BATCH72) $(LR_cosine20k_warm3k)
# EXP_syn_cell_medium := $(TRAIN_syn_cell) $(ARCH_MEDIUM) $(BATCH24) $(LR_cosine60k_warm8k)
# change the lr schedule to 8 gpus, since this one is only for cosmo 7 gpus
EXP_syn_bbox_medium := $(TRAIN_syn_bbox) $(ARCH_MEDIUM) $(BATCH48) $(LR_cosine30k_warm3k)

# use .done_finetune
EXP_ssp_1m_syn_html_small := $(EXP_syn_html_small) $(WEIGHTS_beit_1m_small) $(LOCK_BEIT4)
# EXP_ssp_1m_syn_cell_small := $(EXP_syn_cell_small) $(WEIGHTS_beit_1m_small) $(LOCK_BEIT4)
EXP_ssp_1m_syn_bbox_small := $(EXP_syn_bbox_small) $(WEIGHTS_beit_1m_small) $(LOCK_BEIT4)

EXP_ssp_1m_syn_html_medium := $(EXP_syn_html_medium) $(WEIGHTS_beit_1m_medium) $(LOCK_BEIT4)
EXP_ssp_1m_syn_bbox_medium := $(EXP_syn_bbox_medium) $(WEIGHTS_beit_1m_medium) $(LOCK_BEIT4)

EXP_ssp_2m_syn_html_small := $(EXP_syn_html_small) $(WEIGHTS_beit_2m_small) $(LOCK_BEIT4)
EXP_ssp_2m_syn_bbox_small := $(EXP_syn_bbox_small) $(WEIGHTS_beit_2m_small) $(LOCK_BEIT4)

EXP_ssp_2m_syn_html_medium := $(EXP_syn_html_medium) $(WEIGHTS_beit_2m_medium) $(LOCK_BEIT4)
EXP_ssp_2m_syn_bbox_medium := $(EXP_syn_bbox_medium) $(WEIGHTS_beit_2m_medium) $(LOCK_BEIT4)


TEST_syn_html := $(VOCAB_HTML) \
	$(LABEL_HTML) $(AUG_RESIZE_NORM) \
	$(TRAINER_TABLE) $(I448) $(SEQ512) $(HTML_RESULT_FILE)

TEST_syn_cell := $(VOCAB_WORDPIECE) \
	$(LABEL_CELL) $(AUG_RESIZE_NORM) \
	$(TRAINER_TABLE) $(I448) $(SEQ896) $(CELL_RESULT_FILE)

TEST_syn_bbox := $(VOCAB_BBOX) \
	$(LABEL_BBOX) $(AUG_RESIZE_NORM) \
	$(TRAINER_TABLE) $(I448) $(SEQ1024) $(BBOX_RESULT_FILE)

EXP_syn_html_r18_small-html := $(TEST_syn_html) $(ARCH_R18_SMALL) $(BATCH90)
# EXP_syn_cell_r18_small-html+cell := $(TEST_syn_cell) $(ARCH_R18_SMALL) $(BATCH24)
EXP_syn_bbox_r18_small-bbox := $(TEST_syn_bbox) $(ARCH_R18_SMALL) $(BATCH48)
EXP_syn_html_r18_medium-html := $(TEST_syn_html) $(ARCH_R18_MEDIUM) $(BATCH64)
EXP_syn_bbox_r18_medium-bbox := $(TEST_syn_bbox) $(ARCH_R18_MEDIUM) $(BATCH48)

EXP_syn_html_small-html := $(TEST_syn_html) $(ARCH_SMALL) $(BATCH90)
# EXP_syn_cell_small-html+cell := $(TEST_syn_cell) $(ARCH_SMALL) $(BATCH36)
EXP_syn_bbox_small-bbox := $(TEST_syn_bbox) $(ARCH_SMALL) $(BATCH48)
EXP_syn_html_medium-html := $(TEST_syn_html) $(ARCH_MEDIUM) $(BATCH72)
EXP_syn_bbox_medium-bbox := $(TEST_syn_bbox) $(ARCH_MEDIUM) $(BATCH48)

EXP_ssp_1m_syn_html_small-html := $(EXP_syn_html_small-html)
EXP_ssp_1m_syn_bbox_small-bbox := $(EXP_syn_bbox_small-bbox)
EXP_ssp_1m_syn_html_medium-html := $(EXP_syn_html_medium-html)
EXP_ssp_1m_syn_bbox_medium-bbox := $(EXP_syn_bbox_medium-bbox)

EXP_ssp_2m_syn_html_small-html := $(EXP_ssp_1m_syn_html_small-html)
EXP_ssp_2m_syn_bbox_small-bbox := $(EXP_ssp_1m_syn_bbox_small-bbox)
EXP_ssp_2m_syn_html_medium-html := $(EXP_ssp_1m_syn_html_medium-html)
EXP_ssp_2m_syn_bbox_medium-bbox := $(EXP_ssp_1m_syn_bbox_medium-bbox)

# --------- pubtabnet
TRAIN_pub_html := $(VOCAB_HTML) \
	$(PUBTABNET) $(LABEL_HTML) $(AUG_RESIZE_NORM) \
	$(TRAINER_TABLE) $(I448) $(SEQ512) \
	$(EPOCH24) $(OPT_ADAMW) $(OPT_WD5e2) $(LR_1e4)

TRAIN_pub_bbox := $(VOCAB_BBOX) \
	$(PUBTABNET) $(LABEL_BBOX) $(AUG_RESIZE_NORM) \
	$(TRAINER_TABLE) $(I448) $(SEQ1024) \
	$(EPOCH24) $(OPT_ADAMW) $(OPT_WD5e2) $(LR_3e4) $(GRAD_CLIP12)

TRAIN_pub_cell := $(VOCAB_CELL) \
	$(PUBTABNET) $(LABEL_CELL) $(AUG_RESIZE_NORM) \
	$(TRAINER_TABLE) $(SEQ200) \
	$(EPOCH24) $(OPT_ADAMW) $(OPT_WD5e2) $(LR_1e4)

EXP_pub_cell_r18_small := $(TRAIN_pub_cell) $(ARCH_R18_SMALL) $(BATCH24) $(LR_cosine63k_warm8k) $(I112_336)
EXP_pub_cell_small := $(TRAIN_pub_cell) $(ARCH_SMALL) $(BATCH24) $(LR_cosine63k_warm8k) $(I112_336)
EXP_ssp_2m_pub_cell_small := $(EXP_pub_cell_small) $(WEIGHTS_beit_2m_small) $(LOCK_BEIT4) $(I112_336)
EXP_pub_cell_small_4 := $(TRAIN_pub_cell) $(ARCH_SMALL) $(BATCH24) $(LR_cosine63k_warm8k) $(I112_448)
EXP_pub_cell_small_2 := $(TRAIN_pub_cell) $(ARCH_SMALL) $(BATCH24) $(LR_cosine63k_warm8k) $(I112_224)

# use .done_finetune
EXP_ssp_2m_pub_html_small := $(TRAIN_pub_html) $(ARCH_SMALL) \
	$(WEIGHTS_beit_2m_small) $(LOCK_BEIT4) $(BATCH96) $(LR_cosine16k_warm2k)
EXP_ssp_2m_pub_html_medium := $(TRAIN_pub_html) $(ARCH_MEDIUM) \
	$(WEIGHTS_beit_2m_medium) $(LOCK_BEIT4) $(BATCH72) $(LR_cosine21k_warm3k)

EXP_ssp_2m_pub_bbox_small := $(TRAIN_pub_bbox) $(ARCH_SMALL) \
	$(WEIGHTS_beit_2m_small) $(LOCK_BEIT4) $(BATCH48) $(LR_cosine31k_warm4k)
EXP_ssp_2m_pub_bbox_medium := $(TRAIN_pub_bbox) $(ARCH_MEDIUM) \
	$(WEIGHTS_beit_2m_medium) $(LOCK_BEIT4) $(BATCH48) $(LR_cosine31k_warm4k)

TEST_pub_html := $(VOCAB_HTML) \
	$(LABEL_HTML) $(AUG_RESIZE_NORM) \
	$(TRAINER_TABLE) $(I448) $(SEQ512) $(HTML_RESULT_FILE)

TEST_pub_bbox := $(VOCAB_BBOX) \
	$(LABEL_BBOX) $(AUG_RESIZE_NORM) \
	$(TRAINER_TABLE) $(I448) $(SEQ1024) $(BBOX_RESULT_FILE)

EXP_ssp_2m_pub_html_small-html := $(TEST_pub_html) $(ARCH_SMALL) $(BATCH96)
EXP_ssp_2m_pub_bbox_small-bbox := $(TEST_pub_bbox) $(ARCH_SMALL) $(BATCH48)

EXP_ssp_2m_pub_html_medium-html := $(TEST_pub_html) $(ARCH_MEDIUM) $(BATCH72)
EXP_ssp_2m_pub_bbox_medium-bbox := $(TEST_pub_bbox) $(ARCH_MEDIUM) $(BATCH48)

# vqvae
SET_vqvae := $(VOCAB_NONE) \
	$(LABEL_IMAGE) $(AUG_VQVAE) $(I224) \
	$(TRAINER_VQVAE) $(OPT_ADAMW) $(LR_1e4) $(EPOCH24)

EXP_vqvae_1M := $(SET_vqvae) $(DATA_VQVAE_1M) $(VQVAE_TEMP_1M) $(BATCH80) $(MODEL_VQVAE) $(LR_cosine32k)

EXP_vqvae_2M := $(SET_vqvae) $(DATA_VQVAE_2M) $(VQVAE_TEMP_2M) $(BATCH48) $(MODEL_VQVAE_L) $(LR_cosine118k)

# beit pretraining
SET_beit := $(VOCAB_NONE) \
	$(LABEL_IMAGE) $(AUG_BEIT) \
	$(TRAINER_BEIT) $(OPT_ADAMW) $(OPT_WD5e2) $(LR_5e4) \
	$(TRANS448_VQVAE224_GRID28_MASK300)

EXP_beit_1M_small := $(SET_beit) $(PUB_SYN) $(VQVAE1M_WEIGHTS) $(BEIT_SMALL) \
	$(BATCH384) $(LR_cosine8k_warm1k) $(EPOCH24)
EXP_beit_1M_medium := $(SET_beit) $(PUB_SYN) $(VQVAE1M_WEIGHTS) $(BEIT_MEDIUM) \
	$(BATCH96) $(LR_cosine30k_warm4k) $(EPOCH24)
EXP_beit_2M_small := $(SET_beit) $(DATA_VQVAE_2M) $(VQVAE2M_WEIGHTS) $(BEIT_SMALL) \
	$(BATCH256) $(LR_cosine44k_warm6k) $(EPOCH48)
EXP_beit_2M_medium := $(SET_beit) $(DATA_VQVAE_2M) $(VQVAE2M_WEIGHTS) $(BEIT_MEDIUM) \
	$(BATCH96) $(LR_cosine118k_warm15k) $(EPOCH48)