import torch
import copy

class ConceptualSuppression():
    def __init__(self, embeddings, similarity_threshold: float, modify_suppression_factor_based_on_cossim: bool):
        ## NOTE that the embeddings tensor should NOT contain the last element of the target
        self.embeddings = embeddings
        self.similarity_threshold = similarity_threshold
        self.modify_suppression_factor_based_on_cossim = modify_suppression_factor_based_on_cossim

        assert embeddings.ndim == 3, 'Expected embeddings to be a tensor with 3 dims (batch, seq, embedding_dim)'
        assert embeddings.shape[0] == 1, 'Expected embeddings to have a batch size 1'

        """
        - prepare similarity matrix from embeddings
        """
        self.similarity_matrix = self.get_embedding_similarity_matrix(
            embeddings_batch = embeddings
        ).squeeze(0) ## removing batch dim because batch size is 1 anyways




    def modify_configs(self, configs:list):

        new_configs = configs.copy()

        for i in range(len(configs)):
            item = configs[i]

            assert len(item['suppression_token_index']) == 1
            assert len(item['suppression_factor']) == 1

            ## skip for first element where no modifications are needed
            if item['suppression_token_index'] == [-1]:
                continue

            suppression_token_index = item['suppression_token_index'][0]
            suppression_factor = item['suppression_factor'][0]
            similarity_scores = self.similarity_matrix[suppression_token_index]
            assert (
                    similarity_scores.ndim == 1
                ), f"Expected similarity_scores.ndim to be 1 but got: {similarity_scores.ndim}"

            additional_indices_bool = similarity_scores >= self.similarity_threshold
            additional_indices = additional_indices_bool.nonzero().view(-1).tolist()

            ## remove the index w.r.t which we calculated the scores (cossim 1) from the additional indices
            if suppression_token_index in additional_indices:
                additional_indices.remove(suppression_token_index)

            ## -1 offset because first iter is skipped because item['suppression_token_index'] == [-1] is True
            additional_suppression_factors = [
                self.get_suppression_factor_from_cosine_similarity(
                    suppression_factor = suppression_factor,
                    cosine_similarity = similarity_scores[additional_indices][i]
                ).item()
                for i in range(len(additional_indices))
            ]
            new_configs[i]['original_token_index'] = [i-1]
            new_configs[i]['suppression_token_index'].extend(additional_indices)
            new_configs[i]['suppression_factor'].extend(
                additional_suppression_factors
            )

        return new_configs


    def get_similarity_matrix(self, a, b, eps=1e-8):
        """
        finds the cosine similarity matrix between each item of a w.r.t each item of b
        a and b are expected to be 2 dimensional (seq, hidden_dim)
        added eps for numerical stability
        source: https://stackoverflow.com/a/58144658
        """
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt

    def get_embedding_similarity_matrix(self, embeddings_batch):
        """
        Returns a similarity matrix of shape (batch_size, seq, seq) containing the cosine
        similarities of each token w.r.t every other token.
        """
        assert (
            embeddings_batch.ndim == 3
        ), f"Expected embeddings_batch to have 3 dimensions but got {embeddings_batch.ndim}"
        batch_size, seq_len = embeddings_batch.shape[0], embeddings_batch.shape[1]
        cossim_matrices = torch.zeros(
            embeddings_batch.shape[0],  # batch
            embeddings_batch.shape[1],  # seq
            embeddings_batch.shape[1],  # seq
        )

        with torch.no_grad():
            for batch_idx in range(batch_size):

                source_embeddings = embeddings_batch[batch_idx].float()
                sim_matrix = self.get_similarity_matrix(
                    a=source_embeddings, b=source_embeddings
                )
                cossim_matrices[batch_idx] = sim_matrix

        assert cossim_matrices.shape[0] == 1, f'Expected batch size to be 1 but got: {cossim_matrices.shape[0]}'
        assert cossim_matrices.shape[1] == cossim_matrices.shape[2], 'Expected a square matrix :('

        return cossim_matrices.clip(-1, 1)

    def get_suppression_factor_from_cosine_similarity(
        self, suppression_factor, cosine_similarity
    ):
        if self.modify_suppression_factor_based_on_cossim == False:
            return torch.tensor(suppression_factor)
        else:
            ## the formula we use for calculating the suppression factor for a conceptually similar token
            ## given a suppresion factor and the cossim of the similar token w.r.t the input token
            if 0 <= cosine_similarity.item() <= 1.0:
                x = (1 - suppression_factor) * (1 - cosine_similarity) + suppression_factor
            else:
                x = torch.tensor([1.0])
            return x
