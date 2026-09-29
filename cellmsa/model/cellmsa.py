import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F

from .msa_module import MSAModule, PairformerStack

class ContinuousValueEncoder(nn.Module):
    """
    Encode real number values to a vector using neural nets projection.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_value: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        """
        x = x.unsqueeze(-1)
        x = torch.clamp(x, max=self.max_value)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)
        return self.dropout(x)


class CategoryValueEncoder(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = x.long()
        x = self.embedding(x)
        return self.norm(x)


class BinDecoder(nn.Module):
    def __init__(self, d_model: int, n_bins: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_bins),
        )

    def forward(self, x):
        return self.net(x)


class ReconstructionDecoder(nn.Module):
    def __init__(self, d_model: int, n_bins: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_bins),
        )

    def forward(self, cell_emb, gene_emb):
        # cell_emb: (b, d), gene_emb: (b, n, d) -> output: (b, n, n_bins)
        cell_context = cell_emb.unsqueeze(1).expand(-1, gene_emb.size(1), -1)
        return self.net(torch.cat([cell_context, gene_emb], dim=-1)) # (b, n, 2d) -> (b, n, n_bins)


class Similarity(nn.Module):
    """
    Cosine similarity with temperature, same style as SimCSE/scGPT.
    """

    def __init__(self, temp: float):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class CellMSAModule(nn.Module):
    def __init__(self, gene_vocab_size, embedding_dim, pair_embedding_dim, padding_gene_id,
                 msa_depth=4, pairformer_depth=8, value_emb_style="continuous", n_bins=51,
                 msa_dim=64, msa_outer_product_mean_dim_hidden=8, msa_pwa_dropout_row_prob=0.15,
                 msa_pwa_heads=8, msa_pwa_dim_head=32,
                 pairformer_pair_bias_attn_dim_head=64, pairformer_pair_bias_attn_heads=16,
                 pairformer_dropout_row_prob=0.25, cls_gene_id: int = 1, cce_temp: float = 0.5,
                 neighbor_relation_vocab_size: int = 4, neighbor_relation_unknown_id: int = 3):
        super().__init__()
        self.padding_gene_id = padding_gene_id
        self.cls_gene_id = cls_gene_id
        self.value_emb_style = value_emb_style
        self.n_bins = n_bins
        self.mask_bin_id = n_bins
        self.neighbor_relation_unknown_id = neighbor_relation_unknown_id

        self.gene_encoder = nn.Embedding(gene_vocab_size, embedding_dim, padding_idx=padding_gene_id)
        self.gene_norm = nn.LayerNorm(embedding_dim)
        self.neighbor_relation_encoder = nn.Embedding(neighbor_relation_vocab_size, embedding_dim)

        if value_emb_style == "category":
            self.value_encoder = CategoryValueEncoder(
                num_embeddings=n_bins + 1,  # include one extra mask bin token
                embedding_dim=embedding_dim,
                padding_idx=0,
            )
        else:
            assert False, "Continuous value encoder is currently not supported since we switched to binned input. Please set value_emb_style to 'category'."

        # Pairwise embeddings
        self.pair_proj_left = nn.Linear(embedding_dim, pair_embedding_dim)
        self.pair_proj_right = nn.Linear(embedding_dim, pair_embedding_dim)
        self.pair_proj = nn.Linear(pair_embedding_dim, pair_embedding_dim)

        self.msa_module = MSAModule(
                dim_single = embedding_dim,
                dim_pairwise = pair_embedding_dim,
                depth = msa_depth,
                dim_msa = msa_dim,
                dim_msa_input=embedding_dim,
                dim_additional_msa_feats=0,
                outer_product_mean_dim_hidden=msa_outer_product_mean_dim_hidden,
                msa_pwa_dropout_row_prob=msa_pwa_dropout_row_prob,
                msa_pwa_heads=msa_pwa_heads,
                msa_pwa_dim_head=msa_pwa_dim_head,
        )

        self.pairformer = PairformerStack(
            dim_single=embedding_dim,
            dim_pairwise=pair_embedding_dim,
            depth=pairformer_depth,
            pair_bias_attn_dim_head=pairformer_pair_bias_attn_dim_head,
            pair_bias_attn_heads=pairformer_pair_bias_attn_heads,
            dropout_row_prob=pairformer_dropout_row_prob,
        )

        self.cell_proj = nn.Linear(embedding_dim, embedding_dim)
        self.bin_decoder = BinDecoder(d_model=embedding_dim, n_bins=n_bins)
        self.recon_decoder = ReconstructionDecoder(d_model=embedding_dim, n_bins=n_bins)
        self.sim = Similarity(temp=cce_temp)
        self.creterion_cce = nn.CrossEntropyLoss()
        self.multi_device = False

    def _run_backbone(self, count_matrix, neighbor_matrix, gene_ids, neighbor_relation_ids=None,
                      use_embedding=False, wo_msa_module=False):
        """
        Run CellMSA backbone and return token-wise representations.

        Inputs:
            count_matrix: (b, n)
            neighbor_matrix: (b, s, n)
            gene_ids: (b, n)
            neighbor_relation_ids: (b, s)
        Returns:
            single: (b, n_with_cls, d)
            pairwise: (b, n_with_cls, n_with_cls, dp)
            input_gene_tokens: (b, n, d) gene-side embeddings from model input
            has_cls_input: whether input already has cls at position 0
        """
        has_cls_input = bool(torch.all(gene_ids[:, 0] == self.cls_gene_id).item())
        neighbor_presence = neighbor_matrix[:, :, 0].ne(-1)

        if not has_cls_input:
            b, s, _ = neighbor_matrix.shape
            cls_ids = torch.full((b, 1), self.cls_gene_id, device=gene_ids.device, dtype=gene_ids.dtype)
            cls_values = torch.zeros((b, 1), device=count_matrix.device, dtype=count_matrix.dtype)
            cls_neighbor_values = torch.zeros((b, s, 1), device=neighbor_matrix.device, dtype=neighbor_matrix.dtype)

            gene_ids = torch.cat([cls_ids, gene_ids], dim=1)
            count_matrix = torch.cat([cls_values, count_matrix], dim=1)
            neighbor_matrix = torch.cat([cls_neighbor_values, neighbor_matrix], dim=2)

        count_matrix = torch.cat([count_matrix.unsqueeze(1), neighbor_matrix], dim=1)  # (b, 1 + s, n_with_cls)

        gene_ids_ = gene_ids.unsqueeze(1).expand(-1, count_matrix.size(1), -1)  # (b, 1 + s, n_with_cls)
        gene_embeddings = self.gene_encoder(gene_ids_)  # (b, 1 + s, n_with_cls, d)
        gene_embeddings = self.gene_norm(gene_embeddings)
        value_embeddings = self.value_encoder(count_matrix)  # (b, 1 + s, n_with_cls, d)
        msa_input_embeddings = gene_embeddings + value_embeddings  # (b, 1 + s, n_with_cls, d)
        cell_embeddings = msa_input_embeddings[:, 0, :, :]  # (b, n_with_cls, d)
        msa_embeddings = msa_input_embeddings[:, 1:, :, :]  # (b, s, n_with_cls, d)

        if neighbor_relation_ids is None:
            neighbor_relation_ids = torch.full(
                (neighbor_matrix.size(0), neighbor_matrix.size(1)),
                self.neighbor_relation_unknown_id,
                device=neighbor_matrix.device,
                dtype=torch.long,
            )
        else:
            neighbor_relation_ids = neighbor_relation_ids.to(device=neighbor_matrix.device, dtype=torch.long)
        relation_embeddings = self.neighbor_relation_encoder(neighbor_relation_ids).unsqueeze(2)
        msa_embeddings = msa_embeddings + relation_embeddings

        mask = gene_ids_[:, 0, :].ne(self.padding_gene_id)  # (b, n_with_cls)
        msa_mask = neighbor_presence  # (b, s)

        gene_emb_for_pair = self.gene_encoder(gene_ids)  # (b, n_with_cls, d) --- use the same embedding as single tokens for pairwise representation
        left_emb = self.pair_proj_left(gene_emb_for_pair)   # (b, n_with_cls, dp)
        right_emb = self.pair_proj_right(gene_emb_for_pair) # (b, n_with_cls, dp)
        pair_repr = left_emb[:, :, None, :] + right_emb[:, None, :, :]  # (b, n_with_cls, n_with_cls, dp)
        pair_repr = self.pair_proj(pair_repr) # (b, n_with_cls, n_with_cls, dp)

        msa_module_input = {
            'single_repr': cell_embeddings,
            'pairwise_repr': pair_repr,
            'msa': msa_embeddings,
            'mask': mask,
            'msa_mask': msa_mask,
            'additional_msa_feats': None
        }

        if self.multi_device and self.msa_device != self.io_device:
            for k in msa_module_input:
                msa_module_input[k] = msa_module_input[k].to(self.msa_device) if msa_module_input[k] is not None else None
        if not wo_msa_module:
            pairwise = self.msa_module(**msa_module_input)  # (b, n_with_cls, n_with_cls, dp)
        else:
            pairwise = torch.zeros_like(pair_repr)  # dummy pairwise output if not using msa module
        if self.multi_device and self.msa_device != self.io_device:
            cell_embeddings = cell_embeddings.to(self.io_device)
            pairwise = pairwise.to(self.io_device)
            mask = mask.to(self.io_device)

        input_gene_tokens = gene_embeddings[:, 0, 1:, :]
        single, pairwise = self.pairformer(single_repr=cell_embeddings, pairwise_repr=pairwise, mask=mask, use_embedding=use_embedding)
        return single, pairwise, input_gene_tokens, has_cls_input

    def forward(
        self,
        count_matrix,
        neighbor_matrix,
        gene_ids,
        neighbor_relation_ids=None,
        use_embedding=False,
        CCE=False,
        cce_count_matrix=None,
        cce_gene_ids=None,
        cce_neighbor_matrix=None,
        cce_neighbor_relation_ids=None,
        wo_msa_module=False
    ):
        '''
        input:
            count_matrix: (b, n)
            neighbor_matrix: (b, s, n)
            gene_ids: (b, n)
        output:
            output: {mlm_output: (b, n)}
        '''
        single, pairwise, input_gene_tokens, _ = self._run_backbone(
            count_matrix,
            neighbor_matrix,
            gene_ids,
            neighbor_relation_ids=neighbor_relation_ids,
            use_embedding=use_embedding,
            wo_msa_module=wo_msa_module
        )

        output = {}
        gene_tokens = single[:, 1:, :]
        cell_tokens = single[:, 0, :]
        if self.value_emb_style == "category":
            output["mlm_logits"] = self.bin_decoder(gene_tokens)  # (b, n, n_bins)
        else:
            assert False, "Continuous value encoder is currently not supported since we switched to binned input. Please set value_emb_style to 'category'."
        output["cell_emb_raw"] = self.cell_proj(cell_tokens)  # cls embedding before normalization
        output["cell_emb"] = F.normalize(output["cell_emb_raw"], p=2, dim=-1)  # cls embedding
        output["recon_logits"] = self.recon_decoder(output["cell_emb_raw"], input_gene_tokens)
        output['single_repr'] = gene_tokens
        output['pairwise_repr'] = pairwise[:, 1:, 1:, :]

        if CCE:
            cce_count_matrix = count_matrix if cce_count_matrix is None else cce_count_matrix
            cce_gene_ids = gene_ids if cce_gene_ids is None else cce_gene_ids
            cce_neighbor_matrix = neighbor_matrix if cce_neighbor_matrix is None else cce_neighbor_matrix
            cce_neighbor_relation_ids = neighbor_relation_ids if cce_neighbor_relation_ids is None else cce_neighbor_relation_ids
            single2, _, _, _ = self._run_backbone(
                cce_count_matrix,
                cce_neighbor_matrix,
                cce_gene_ids,
                neighbor_relation_ids=cce_neighbor_relation_ids,
                use_embedding=use_embedding,
            )

            cell1 = output["cell_emb"]  # (b, d)
            cell2 = F.normalize(self.cell_proj(single2[:, 0, :]), p=2, dim=-1)  # (b, d)

            if dist.is_initialized() and self.training:
                cls2_list = [torch.zeros_like(cell2) for _ in range(dist.get_world_size())]  # world_size * (b, d)
                dist.all_gather(tensor_list=cls2_list, tensor=cell2.contiguous())
                cls2_list[dist.get_rank()] = cell2
                cell2_all = torch.cat(cls2_list, dim=0)  # (global_b, d)

                cls1_list = [torch.zeros_like(cell1) for _ in range(dist.get_world_size())]  # world_size * (b, d)
                dist.all_gather(tensor_list=cls1_list, tensor=cell1.contiguous())
                cls1_list[dist.get_rank()] = cell1
                cell1_all = torch.cat(cls1_list, dim=0)  # (global_b, d)

                # Local positives point to the corresponding global indices in cell2_all.
                local_bs = torch.tensor([cell1.size(0)], device=cell1.device, dtype=torch.long)  # (1,)
                all_bs = [torch.zeros_like(local_bs) for _ in range(dist.get_world_size())]  # world_size * (1,)
                dist.all_gather(all_bs, local_bs)
                all_bs = torch.cat(all_bs, dim=0)  # (world_size,)
                rank = dist.get_rank()
                offset = all_bs[:rank].sum()  # scalar
                labels = torch.arange(cell1.size(0), device=cell1.device, dtype=torch.long) + offset  # (b,)
            else:
                cell2_all = cell2  # (b, d)
                cell1_all = cell1  # (b, d)
                labels = torch.arange(cell1.size(0), device=cell1.device, dtype=torch.long)  # (b,)

            # Compute CE only for local cell1 (with grad) against no-grad cell2 list.
            cos_sim1 = self.sim(cell1.unsqueeze(1), cell2_all.unsqueeze(0))  # (b, global_b) or (b, b)
            cos_sim2 = self.sim(cell2.unsqueeze(1), cell1_all.unsqueeze(0))  # (b, global_b) or (b, b)
            loss_cce1 = self.creterion_cce(cos_sim1, labels)
            loss_cce2 = self.creterion_cce(cos_sim2, labels)
            output["loss_cce"] = (loss_cce1 + loss_cce2) / 2

        return output
