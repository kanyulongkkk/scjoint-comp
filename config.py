import torch
import os


class Config(object):
    def __init__(self):
        DB = 'data_atlas'
        self.use_cuda = False

        self.threads = 1

        if not self.use_cuda:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:0')

        if DB == 'data_atlas':
            # DB info
            # label_to_idx.txt 的文件行数
            self.number_of_class = 73
            # 用sparse.load_npz("data_atlas/atlas_rna_dropseq.npz").shape 返回的第二个数字
            self.input_size = 15519
            self.rna_paths = ['data_atlas/atlas_rna_dropseq.npz',
                              'data_atlas/atlas_rna_facs.npz']
            self.rna_labels = ['data_atlas/atlas_rna_dropseq_cellTypes.txt',
                               'data_atlas/atlas_rna_facs_cellTypes.txt']
            self.atac_paths = ['data_atlas/atlas_atac.npz']
            # Optional. If atac_labels are provided, accuracy after knn would be provided.
            self.atac_labels = ['data_atlas/atlas_atac_cellTypes.txt']
            self.rna_protein_paths = []
            self.atac_protein_paths = []

            # Training config
            self.batch_size = 256
            self.lr_stage1 = 0.01
            self.lr_stage3 = 0.01
            self.lr_decay_epoch = 20
            self.epochs_stage1 = 20
            self.epochs_stage3 = 20
            self.p = 0.8
            self.embedding_size = 64
            self.momentum = 0.9
            self.center_weight = 1
            self.with_crossentorpy = True
            self.seed = 1
            self.checkpoint = ''

        elif DB == "MOp":
            self.number_of_class = 21
            self.input_size = 18603
            self.rna_paths = ['data_MOp/YaoEtAl_RNA_snRNA_10X_v3_B_exprs.npz',
                              'data_MOp/YaoEtAl_RNA_snRNA_10X_v3_A_exprs.npz',
                              'data_MOp/YaoEtAl_RNA_snRNA_10X_v2_exprs.npz',
                              'data_MOp/YaoEtAl_RNA_snRNA_SMARTer_exprs.npz',
                              'data_MOp/YaoEtAl_RNA_scRNA_10X_v3_exprs.npz',
                              'data_MOp/YaoEtAl_RNA_scRNA_10X_v2_exprs.npz',
                              'data_MOp/YaoEtAl_RNA_scRNA_SMARTer_exprs.npz']
            self.rna_labels = ['data_MOp/YaoEtAl_RNA_snRNA_10X_v3_B_cellTypes.txt',
                               'data_MOp/YaoEtAl_RNA_snRNA_10X_v3_A_cellTypes.txt',
                               'data_MOp/YaoEtAl_RNA_snRNA_10X_v2_cellTypes.txt',
                               'data_MOp/YaoEtAl_RNA_snRNA_SMARTer_cellTypes.txt',
                               'data_MOp/YaoEtAl_RNA_scRNA_10X_v3_cellTypes.txt',
                               'data_MOp/YaoEtAl_RNA_scRNA_10X_v2_cellTypes.txt',
                               'data_MOp/YaoEtAl_RNA_scRNA_SMARTer_cellTypes.txt']
            self.atac_paths = ['data_MOp/YaoEtAl_ATAC_exprs.npz',
                               'data_MOp/YaoEtAl_snmC_exprs.npz']
            self.atac_labels = ['data_MOp/YaoEtAl_ATAC_cellTypes.txt',
                                'data_MOp/YaoEtAl_snmC_cellTypes.txt']
            self.rna_protein_paths = []
            self.atac_protein_paths = []

            # Training config
            self.batch_size = 256
            self.lr_stage1 = 0.01
            self.lr_stage3 = 0.001
            self.lr_decay_epoch = 20
            self.epochs_stage1 = 10
            self.epochs_stage3 = 10
            self.p = 0.8
            self.embedding_size = 64
            self.momentum = 0.9
            self.center_weight = 20
            self.with_crossentorpy = True
            self.seed = 1
            self.checkpoint = ''
        elif DB == "data_atlas11":
            self.number_of_class = 19
            self.input_size = 18603
            self.rna_paths = ['data_MOp/YaoEtAl_RNA_snRNA_10X_v3_B_exprs.npz',
                              'data_MOp/YaoEtAl_RNA_snRNA_10X_v3_A_exprs.npz',
                              'data_MOp/YaoEtAl_RNA_snRNA_10X_v2_exprs.npz',
                              'data_MOp/YaoEtAl_RNA_snRNA_SMARTer_exprs.npz',
                              'data_MOp/YaoEtAl_RNA_scRNA_10X_v3_exprs.npz',
                              'data_MOp/YaoEtAl_RNA_scRNA_10X_v2_exprs.npz',
                              'data_MOp/YaoEtAl_RNA_scRNA_SMARTer_exprs.npz']
            self.rna_labels = ['data_MOp/YaoEtAl_RNA_snRNA_10X_v3_B_cellTypes.txt',
                               'data_MOp/YaoEtAl_RNA_snRNA_10X_v3_A_cellTypes.txt',
                               'data_MOp/YaoEtAl_RNA_snRNA_10X_v2_cellTypes.txt',
                               'data_MOp/YaoEtAl_RNA_snRNA_SMARTer_cellTypes.txt',
                               'data_MOp/YaoEtAl_RNA_scRNA_10X_v3_cellTypes.txt',
                               'data_MOp/YaoEtAl_RNA_scRNA_10X_v2_cellTypes.txt',
                               'data_MOp/YaoEtAl_RNA_scRNA_SMARTer_cellTypes.txt']
            self.atac_paths = ['data_MOp/YaoEtAl_ATAC_exprs.npz',
                               'data_MOp/YaoEtAl_snmC_exprs.npz']
            self.atac_labels = ['data_MOp/YaoEtAl_ATAC_cellTypes.txt',
                                'data_MOp/YaoEtAl_snmC_cellTypes.txt']
            self.rna_protein_paths = []
            self.atac_protein_paths = []

            # Training config
            self.batch_size = 256
            self.lr_stage1 = 0.01
            self.lr_stage3 = 0.001
            self.lr_decay_epoch = 20
            self.epochs_stage1 = 10
            self.epochs_stage3 = 10
            self.p = 0.8
            self.embedding_size = 64
            self.momentum = 0.9
            self.center_weight = 20
            self.with_crossentorpy = True
            self.seed = 1
            self.checkpoint = ''
        elif DB == "db4_control":
            self.number_of_class = 7  # Number of cell types in CITE-seq data
            # Number of common genes and proteins between CITE-seq data and ASAP-seq
            self.input_size = 17668
            # RNA gene expression from CITE-seq data
            self.rna_paths = ['data/citeseq_control_rna.npz']
            # CITE-seq data cell type labels (coverted to numeric)
            self.rna_labels = ['data/citeseq_control_cellTypes.txt']
            # ATAC gene activity matrix from ASAP-seq data
            self.atac_paths = ['data/asapseq_control_atac.npz']
            # ASAP-seq data cell type labels (coverted to numeric)
            self.atac_labels = ['data/asapseq_control_cellTypes.txt']
            # Protein expression from CITE-seq data
            self.rna_protein_paths = ['data/citeseq_control_adt.npz']
            # Protein expression from ASAP-seq data
            self.atac_protein_paths = ['data/asapseq_control_adt.npz']

            # Training config
            #self.batch_size = 256
            #self.lr_stage1 = 0.01
            #self.lr_stage3 = 0.01
            #self.lr_decay_epoch = 20
            #self.epochs_stage1 = 20
            #self.epochs_stage3 = 20
            #self.p = 0.8
            #self.embedding_size = 64
            #self.momentum = 0.9
            #self.center_weight = 1
            #self.with_crossentorpy = True
            #self.seed = 1
            #self.checkpoint = ''
            self.batch_size = 256
            self.lr_stage1 = 0.02
            self.lr_stage3 = 0.01
            self.lr_decay_epoch = 11
            self.epochs_stage1 = 40
            self.epochs_stage3 = 40
            self.p = 0.9999
            self.embedding_size = 224
            self.momentum = 0.9
            self.center_weight = 1
            self.with_crossentorpy = True
            self.seed = 1
            self.checkpoint = ''
