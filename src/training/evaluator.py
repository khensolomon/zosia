# Contains functions for evaluation metrics like BLEU.

import sacrebleu
from typing import List, Union

def calculate_bleu(candidate_corpus: List[str], references_corpus: List[List[str]]) -> float:
    """
    Calculates the BLEU score using sacrebleu.

    Args:
        candidate_corpus (List[str]): List of translated sentences (strings).
        references_corpus (List[List[str]]): List of lists of reference sentences.
                                            Each inner list contains one or more reference translations for a candidate.

    Returns:
        float: The BLEU score.
    """
    if not candidate_corpus or not references_corpus:
        return 0.0

    # sacrebleu expects references to be a list of lists where each inner list
    # corresponds to a single reference set (e.g., all 1st references, all 2nd references)
    # If each candidate has multiple references, you need to transpose the reference list.
    # For now, assuming `references_corpus` is already in the correct format for sacrebleu:
    # [['ref1_sent1', 'ref1_sent2'], ['ref2_sent1', 'ref2_sent2']]
    # OR, if only one reference per sentence: [['ref_sent1'], ['ref_sent2']]
    
    # If references_corpus is a list of [single_ref_string], convert to [[single_ref_string]] for sacrebleu
    # The `references_corpus` param in this function expects List[List[str]], where each inner list
    # contains reference(s) for a *single* candidate sentence.
    # sacrebleu.corpus_bleu(candidates, [references_for_candidate_1, references_for_candidate_2])
    # The `corpus_bleu` function expects references as a list of lists of references, where each inner list
    # corresponds to the references for a *single* sentence in the `candidates` list.
    # The format is `references=[[ref1_s1, ref1_s2,...], [ref2_s1, ref2_s2,...], ...]`
    
    # Let's adjust `references_corpus` to fit `sacrebleu.corpus_bleu`'s common usage:
    # where it expects `[reference1_list, reference2_list, ...]` if there are multiple reference sets.
    # If you have one reference per translated sentence, it's `[[ref_s1], [ref_s2], ...]`

    # Simple check for the most common case: one reference per candidate sentence
    # If references_corpus is already in the format [[ref_s1], [ref_s2], ...], it's fine.
    # If it's [ref_s1, ref_s2, ...], it needs to be wrapped.
    if isinstance(references_corpus[0], str): # This means it's [ref_s1, ref_s2, ...]
        references_corpus = [[ref] for ref in references_corpus]
    elif not isinstance(references_corpus[0], list):
        raise ValueError("references_corpus must be List[List[str]] or List[str]")


    # For sacrebleu, the reference format is:
    # list_of_references = [
    #   ['ref_sen_1', 'ref_sen_2', ...],  # First reference set
    #   ['ref2_sen_1', 'ref2_sen_2', ...], # Second reference set (optional)
    # ]
    # However, for a single reference per sentence as is common in NMT evaluation:
    # references = [[ref1_for_candidate1], [ref1_for_candidate2], ...]
    
    # sacrebleu's corpus_bleu expects: candidates (list of strings), list of lists of references.
    # The inner list contains the references for *each sentence*.
    # So if you have one reference per sentence: [[ref1_sent1], [ref1_sent2], ...]
    # If you have two references per sentence: [[ref1_sent1, ref2_sent1], [ref1_sent2, ref2_sent2], ...]

    # Ensure references_corpus is formatted as expected by sacrebleu:
    # a list of reference lists, where each inner list contains references for a single hypothesis.
    # Example: corpus_bleu(["The cat.", "The dog."], [["Le chat."], ["Le chien."]]) -> WRONG
    # Correct: corpus_bleu(["The cat.", "The dog."], [["Le chat."], ["Le chien."]]) is WRONG if `references_corpus` is `[["Le chat.", "Le chien."]]`
    # Correct: corpus_bleu(["The cat.", "The dog."], [[ "Le chat."], ["Le chien."]] -> this is the format we get from `all_target_tokens` if we pass `[[trg_text]]`

    # Ensure `references_corpus` has the correct shape for sacrebleu:
    # It must be a list of lists of references, where each inner list corresponds to *one* reference set across all candidates.
    # i.e., `references_corpus = [ ['ref_1_sent_1', 'ref_1_sent_2', ...], ['ref_2_sent_1', 'ref_2_sent_2', ...] ]`
    # where `ref_1_sent_N` is the Nth reference for *all* candidates.
    # My current `all_target_tokens` creates `[[trg_text_1], [trg_text_2], ...]`
    # which means it's a list of single-reference lists for each candidate.
    # SacreBLEU takes a list of candidate strings and a list of *lists of reference strings*.
    # Each list of reference strings represents one of the sets of references.
    # So if you have one reference per sentence: `[[ref1_s1, ref1_s2, ...]]`
    # If you have two references per sentence: `[[ref1_s1, ref1_s2, ...], [ref2_s1, ref2_s2, ...]]`

    # Let's reshape `references_corpus` assuming `references_corpus` is `[[ref_s1], [ref_s2], ...]`
    # and converting it to `[[ref_s1, ref_s2, ...]]`
    transposed_references = [[] for _ in range(len(references_corpus[0]))] # assuming each inner list has same length
    for item in references_corpus:
        for i, ref in enumerate(item):
            transposed_references[i].append(ref)
            
    # Now call sacrebleu
    bleu = sacrebleu.corpus_bleu(candidate_corpus, transposed_references)
    return bleu.score