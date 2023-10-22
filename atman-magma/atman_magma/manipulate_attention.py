import torch

def manipulate_attention_scores(
        attention_scores,
        attention_mask,
        modified_causal_attention_mask,
        multiplicative = True,
        apply_softmax = False
    ):
        """
        Manipulates attention scores on each transformer layer as per the values of suppression_factors and suppression_token_indices.

        Arguments:
            `attention_scores`: the attention scores from the attention module
            `attention_mask`: the tril attention mask
            `suppression_factors` (`tensor` or `list`): factors by which we'll scale the attention scores of the tokens at the given indices
            `suppression_token_indices` (`tensor` or `list`): token indices where we'll scale the attention scores, should be the same shape as that of suppression_factors

        For example:
        ```
        suppression_token_indices = [
            [0,2,3],
            [4,5]
        ]
        suppression_factors = [
            [0.1, 0.2, 0.9],
            [0.2, 0.3]
        ]
        ```

        would scale the attention of the first, 3rd and the 4th input token on the first batch item by factors
        0.1, 0.2, 0.9 respectively, and the 5th and 6th input token on the 2nd batch item by factors 0.2 and 0.3 respectively

        - `0. < suppression_factors < 1.` = suppression
        - `suppression_factors > 1.` = amplification
        """

        batch_size = attention_scores.shape[0]

        """
        Note that we might have to manipulate attention masks differently for each batch item
        given below is a simpler version of what's happening below that

        let us assume that we have batch size = 2:
        - for batch item 0, we want to replace indices 1 and 2 in the sequence with 0.33 and 0.69 respectively
        - for batch item 1, we want to replace indices 0 and 1 in the sequence with 0.22 and 0.99 respectively

        Note that different batch items can have different lengths of indices and factors.
        But len(indices[i]) should always match len(factors[i]) for i in range(batch_size)

        We expect indices like:
        indices = [
            [1, 2], ## batch item 0
            [0, 1],  ## batch item 1
        ]

        and factors like:
        factors = [
            [0.33, 0.69], ## batch item 0
            [0.22, 0.99]  ## batch item 1
        ]
        """
        assert (
            modified_causal_attention_mask.shape[0] == batch_size
        ), f"Expected modified_causal_attention_mask to have a batch size of {batch_size} but got shape: {modified_causal_attention_mask.shape}"

        ## apply modified mask after repeating it along seq dim i.e attention_scores.shape[1]
        if multiplicative:
            attention_scores = attention_scores - attention_scores.min(-1).values.unsqueeze(
            3
            )

            ## apply modified mask
            attention_scores = attention_scores * modified_causal_attention_mask

            ## set all values where mask == True to a very low value so that they become 0 after softmax
            attention_scores.masked_fill_(
                ~attention_mask.to(attention_scores.device), -10000.0
            )
        else:
            attention_scores = attention_scores + modified_causal_attention_mask

        return attention_scores
